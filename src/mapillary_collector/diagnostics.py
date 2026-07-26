"""Read-only inspection commands, plus the one destructive reset."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone

from .config import Config, resolve_token
from .constants import SHARD_LOCAL, SHARD_UPLOADED, SHARD_UPLOADING
from .logging_setup import get_logger
from .quota import compute_quota, quota_table
from .staging import StagingArea, inspect_tar
from .state import StateDB
from .upload import HfStore
from .utils import disk_free_gb, human_count


def _open_db(cfg: Config) -> StateDB:
    if not cfg.db_path.exists():
        raise SystemExit(f"no database yet at {cfg.db_path} -- run the collector first")
    return StateDB(cfg.db_path)


def status(cfg: Config) -> None:
    """What has the collector done so far?"""
    db = _open_db(cfg)
    try:
        totals = db.totals()
        print(f"data dir     : {cfg.data_dir}")
        print(f"images       : {human_count(totals['images'])}")
        print(f"  staged     : {totals['staged']} (waiting for a full shard)")
        print(f"shards       : {totals['shards']}")
        print(f"countries    : {totals['countries']}")
        print(f"candidates   : {human_count(totals['candidates'])}")
        print(f"disk free    : {disk_free_gb(cfg.data_dir):.1f} GB")
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        used = int(db.kv_get(f"tile_budget:{day}", 0) or 0)
        print(f"tiles today  : {used}/{cfg.daily_tile_budget}")
        rows = db.countries_by_status("in_progress")
        for name, quota, tiles in rows:
            have = db.images_in_country(name)
            print(f"in progress  : {name} {have}/{quota} (tiles={tiles})")
    finally:
        db.close()


def repair(cfg: Config) -> None:
    """Undo the damage an outage leaves behind.

    Tiles that failed while the server was refusing us were recorded as dead by
    older versions, and countries whose discovery failed got marked finished
    with zero images. Both are reversible; this reverses them.
    """
    db = _open_db(cfg)
    try:
        revived = db.reset_error_tiles()
        print(f"returned {revived} errored tile(s) to the queue")

        stale = db.countries_missing_images(("completed", "exhausted"))
        for name in stale:
            db.upsert_country(name, "in_progress")
            db.clear_base_scan(name, cfg.tile_base_zoom)
        print(f"reopened {len(stale)} country(ies) that collected nothing"
              + (": " + ", ".join(sorted(stale)[:10]) if stale else ""))

        reopened = 0
        for name, quota, tiles in db.countries_by_status("completed"):
            have = db.images_in_country(name)
            fresh = compute_quota(tiles or 0, cfg)
            if fresh > have:
                db.upsert_country(name, "in_progress", quota=fresh)
                reopened += 1
        print(f"reopened {reopened} country(ies) whose quota grew")
    finally:
        db.close()


def countries(cfg: Config, limit: int = 40) -> None:
    """Per-country results, biggest first."""
    db = _open_db(cfg)
    try:
        printed = 0
        for state in ("completed", "in_progress", "exhausted", "failed"):
            rows = db.countries_by_status(state)
            if not rows:
                continue
            print(f"\n== {state} ({len(rows)}) ==")
            for name, quota, tiles in rows:
                have = db.images_in_country(name)
                row = db.country_row(name)
                started = row["started_at"] if row else None
                finished = row["finished_at"] if row else None
                elapsed = ""
                if started and finished:
                    try:
                        delta = (datetime.fromisoformat(finished)
                                 - datetime.fromisoformat(started))
                        elapsed = f"  {delta.total_seconds() / 60:.1f} min"
                    except ValueError:
                        pass
                print(f"  {name:<34} {have:>6}/{quota or 0:<6} "
                      f"tiles={tiles or 0:<7}{elapsed}")
                printed += 1
                if printed >= limit:
                    print("  ...")
                    return
    finally:
        db.close()


def show_quota(cfg: Config) -> None:
    """Preview the quota curve without touching the network."""
    print(f"quota = clamp({cfg.quota_min}, {cfg.quota_k:g} * tiles^{cfg.quota_alpha:g}, "
          f"{cfg.quota_max})\n")
    print(f"{'leaf tiles':>12}  {'images':>8}")
    for tiles, quota in quota_table(cfg):
        print(f"{tiles:>12}  {quota:>8}")


def verify_local(cfg: Config) -> None:
    """Do local tars agree with the database?"""
    db = _open_db(cfg)
    try:
        problems = 0
        for state in (SHARD_LOCAL, SHARD_UPLOADING, SHARD_UPLOADED):
            for shard_idx, filename, n in db.shards_with_status(state):
                path = cfg.shards_dir / (filename or "")
                if not path.exists():
                    if state != SHARD_UPLOADED:
                        print(f"shard {shard_idx:06d} [{state}]: file missing")
                        problems += 1
                    continue
                keys = inspect_tar(path)
                ids = db.image_ids_in_shard(shard_idx)
                if keys is None:
                    print(f"shard {shard_idx:06d}: tar unreadable")
                    problems += 1
                elif keys != ids:
                    print(f"shard {shard_idx:06d}: MISMATCH tar={len(keys)} "
                          f"db={len(ids)}")
                    problems += 1
        print(f"{problems} problem(s)" if problems else "all local shards consistent")
    finally:
        db.close()


def verify_remote(cfg: Config) -> None:
    """Is every shard marked uploaded actually on the Hub?"""
    db = _open_db(cfg)
    store = HfStore(cfg, resolve_token(cfg.hf_token_env), get_logger())
    try:
        remote = set(store.list_shard_files())
        missing = 0
        for shard_idx, filename, _ in db.shards_with_status(SHARD_UPLOADED):
            target = store.remote_path(filename or "")
            if target not in remote:
                print(f"shard {shard_idx:06d}: marked uploaded but MISSING on hub")
                missing += 1
        print(f"{missing} missing" if missing else
              f"all {len(remote)} uploaded shard(s) verified on hub")
    finally:
        db.close()


def doctor(cfg: Config) -> None:
    """Pre-flight check: everything a run needs, without starting one."""
    def report(label: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}{'  -- ' + detail if detail else ''}")

    report("data dir writable", _writable(cfg), str(cfg.data_dir))
    report("mapillary token", bool(resolve_token(cfg.mapillary_token_env)),
           cfg.mapillary_token_env)
    report("hugging face token",
           bool(resolve_token(cfg.hf_token_env)) or cfg.dry_run_uploads,
           cfg.hf_token_env)
    report("disk headroom", disk_free_gb(cfg.data_dir) >= cfg.min_free_gb,
           f"{disk_free_gb(cfg.data_dir):.1f} GB free")
    if cfg.db_path.exists():
        db = StateDB(cfg.db_path)
        try:
            report("database integrity", db.integrity_ok())
        finally:
            db.close()
    else:
        report("database", True, "none yet (fresh start)")

    staging = StagingArea(cfg, get_logger())
    orphans = staging.orphan_ids()
    report("staging clean", not orphans,
           f"{len(orphans)} incomplete file(s), cleaned on next run" if orphans else "")


def _writable(cfg: Config) -> bool:
    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        probe = cfg.data_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def reset(cfg: Config, confirm: str) -> None:
    """Back up state to a timestamped folder, then wipe it.

    Never touches anything already uploaded to Hugging Face.
    """
    if confirm != "RESET":
        print('refusing: pass --confirm RESET')
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = cfg.data_dir / "backups" / stamp
    dest.mkdir(parents=True, exist_ok=True)

    if cfg.db_path.exists():
        shutil.copy2(cfg.db_path, dest / cfg.db_path.name)
    if cfg.log_path.exists():
        shutil.copy2(cfg.log_path, dest / cfg.log_path.name)

    for path in (cfg.db_path, cfg.db_path.with_suffix(".sqlite-wal"),
                 cfg.db_path.with_suffix(".sqlite-shm")):
        if path.exists():
            path.unlink()
    for directory in (cfg.staging_dir, cfg.shards_dir):
        if directory.exists():
            shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
    print(f"backed up to {dest}; local state reset (hub untouched)")
