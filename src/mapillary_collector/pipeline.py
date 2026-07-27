"""The run loop.

Per country:
  discovery -> quota -> walk candidates in round-robin order -> prefilter (free)
  -> fetch metadata + image in a thread pool -> validate -> stage -> pack shard

Only the pool step is concurrent. Registration, staging and packing stay on the
main thread, which is what keeps the database and the filesystem in lockstep and
makes resume exact.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from shapely.prepared import prep

from .client import MapillaryClient, MapillaryError, TileUnavailableError
from .config import Config, resolve_token
from .constants import (
    COUNTRY_COMPLETED,
    COUNTRY_EXHAUSTED,
    COUNTRY_FAILED,
    COUNTRY_IN_PROGRESS,
    SHARD_LOCAL,
    TILE_PROBE_LNGLAT,
)
from .discovery import TileDiscovery, TileQuotaExhausted
from .filters import evaluate_image, prefilter_candidate, validate_image_bytes
from .geo import CountryRec, load_countries, lnglat_to_tile
from .logging_setup import log_config_summary
from .quota import compute_quota
from .recovery import RecoveryManager
from .staging import StagingArea
from .state import StateDB
from .upload import HfStore, UploadManager
from .utils import disk_free_gb, ensure_dirs, human_count, utc_now_iso


class ShutdownRequested(Exception):
    """Raised internally when SIGINT/SIGTERM arrives, to unwind cleanly."""


class Pipeline:
    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self.db: Optional[StateDB] = None
        self.client: Optional[MapillaryClient] = None
        self.staging: Optional[StagingArea] = None
        self.store: Optional[HfStore] = None
        self.uploader: Optional[UploadManager] = None
        self._stop = threading.Event()
        self._finalized = False
        self._tiles_blocked = False
        self._pool: Optional[ThreadPoolExecutor] = None

    # ---- lifecycle ----------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Ctrl-C and `kill` set a flag instead of killing mid-write.

        The loop checks it between images, so shutdown always happens at a
        point where the database and the staging directory agree.
        """
        def handler(signum, _frame):
            if self._stop.is_set():
                self.log.warning("[SHUTDOWN] second signal, exiting immediately")
                raise KeyboardInterrupt
            self.log.warning("[SHUTDOWN] signal %s received, finishing current "
                             "work then stopping", signum)
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not on the main thread, or unsupported platform

    def startup(self) -> None:
        ensure_dirs(self.cfg.data_dir, self.cfg.staging_dir,
                    self.cfg.shards_dir, self.cfg.cache_dir)

        mly_token = resolve_token(self.cfg.mapillary_token_env)
        if not mly_token:
            raise RuntimeError(
                f"Mapillary token missing. Put {self.cfg.mapillary_token_env}=... "
                "in your .env file or export it."
            )
        hf_token = resolve_token(self.cfg.hf_token_env)
        if not hf_token and not self.cfg.dry_run_uploads:
            raise RuntimeError(
                f"Hugging Face token missing. Put {self.cfg.hf_token_env}=... "
                "in your .env file, or run with --dry-run."
            )

        self.db = StateDB(self.cfg.db_path, cache_mb=self.cfg.sqlite_cache_mb)
        if not self.db.integrity_ok():
            raise RuntimeError(
                f"State database is corrupt: {self.cfg.db_path}. "
                "Move it aside and rerun to start fresh."
            )
        self.client = MapillaryClient(self.cfg, mly_token, self.log)
        self.staging = StagingArea(self.cfg, self.log)
        self.store = HfStore(self.cfg, hf_token, self.log)
        self.store.ensure_repo()
        self.uploader = UploadManager(self.cfg, self.db, self.store, self.log)
        self.uploader.start()
        # one pool for the entire run: creating and tearing one down per country
        # costs thread churn for no benefit, and the workers are stateless
        self._pool = ThreadPoolExecutor(max_workers=self.cfg.workers,
                                        thread_name_prefix="fetch")
        RecoveryManager(self.cfg, self.db, self.staging, self.store,
                        self.uploader, self.log).reconcile()

    def run(self) -> None:
        self._install_signal_handlers()
        self.startup()
        log_config_summary(self.cfg, self.log)
        countries = load_countries(self.cfg, self.log)
        try:
            self._supervise(countries)
            self._finalize("complete")
        except (KeyboardInterrupt, ShutdownRequested):
            self.log.warning("[SHUTDOWN] interrupted, saving state")
            self._finalize("interrupt")
        except Exception:
            self.log.exception("[ERROR] fatal error, saving what is salvageable")
            self._finalize("error")
            raise

    def _supervise(self, countries: list) -> None:
        """Keep working until everything is done or the user stops us.

        A blocked tile server is not a reason to exit. Tile discovery and image
        collection draw on separate quotas -- tiles are 50k/day, the graph API
        is 60k/minute -- so when tiles are refused there is usually a large
        backlog of already-discovered candidates that can still be collected.
        Only when that backlog is also empty do we actually wait.
        """
        while not self._stop.is_set():
            collected = self._run_countries(countries)

            if self._all_terminal(countries):
                self.log.info("[DONE] every country processed")
                return
            if self._stop.is_set():
                return
            if not self.cfg.forever:
                self.log.info("[DONE] pass finished (--once); rerun to continue")
                return

            if self._tiles_blocked:
                if collected:
                    self.log.info("[TILES] blocked, but collected %d image(s) from "
                                  "banked candidates this pass", collected)
                self._wait_for_tiles()
                continue

            if collected == 0:
                # tiles are fine and a full pass produced nothing: everything
                # reachable under the current limits is already collected
                self.log.info("[DONE] no further progress possible with the "
                              "current settings; raise the quota or the tile "
                              "cap to collect more")
                return

    def _run_countries(self, countries: list) -> int:
        """One pass over every country. Returns how many images it collected."""
        self._reopen_empty_countries()
        skip = {COUNTRY_COMPLETED, COUNTRY_FAILED}
        if not self.cfg.retry_exhausted:
            skip.add(COUNTRY_EXHAUSTED)
        collected = 0
        for rec in countries:
            if self._stop.is_set():
                return collected
            if self.db.country_status(rec.name) in skip:
                continue
            if rec.geometry is None or rec.geometry.is_empty:
                self.db.upsert_country(rec.name, COUNTRY_FAILED,
                                       iso3=rec.iso3, continent=rec.continent)
                self.log.warning("[COUNTRY] %s has no usable polygon", rec.name)
                continue
            try:
                collected += self._collect_country(
                    rec, allow_discovery=not self._tiles_blocked)
            except TileQuotaExhausted as exc:
                # not fatal: other countries may still have banked candidates,
                # and collecting those does not touch the tile server at all
                if not self._tiles_blocked:
                    self.log.warning("[TILES] %s", exc)
                    self.log.warning("[TILES] discovery paused; continuing with "
                                     "candidates already on disk")
                self._tiles_blocked = True
        return collected

    def _reopen_empty_countries(self) -> None:
        """Undo any country that got marked finished without collecting anything.

        A country only earns a terminal status by actually being walked to the
        end. Zero images plus a terminal status means an outage wrote it off, so
        reopen it -- otherwise one bad hour silently removes a country from the
        dataset forever.
        """
        stale = self.db.countries_missing_images(
            (COUNTRY_COMPLETED, COUNTRY_EXHAUSTED))
        for name in stale:
            self.db.upsert_country(name, COUNTRY_IN_PROGRESS)
            self.db.clear_base_scan(name, self.cfg.tile_base_zoom)
        if stale:
            self.log.warning("[RECOVERY] reopened %d country(ies) marked finished "
                             "with zero images: %s", len(stale),
                             ", ".join(sorted(stale)[:8])
                             + (" ..." if len(stale) > 8 else ""))
        revived = self.db.reset_error_tiles()
        if revived:
            self.log.info("[RECOVERY] returned %d errored tile(s) to the queue",
                          revived)

    # ---- per country --------------------------------------------------

    def _collect_country(self, rec: CountryRec,
                         allow_discovery: bool = True) -> int:
        country_started = time.monotonic()
        prepared = prep(rec.geometry)
        discovery = TileDiscovery(self.cfg, self.client, self.db, rec,
                                  prepared, self.log)

        self.db.upsert_country(rec.name, COUNTRY_IN_PROGRESS, iso3=rec.iso3,
                               continent=rec.continent, started_at=utc_now_iso())

        if allow_discovery:
            try:
                scan_started = time.monotonic()
                leaf_tiles = discovery.base_scan()
                scan_s = time.monotonic() - scan_started
                if scan_s > 5:
                    self.log.info("[TILES] %s: base scan took %.1fs -> %d leaf tiles",
                                  rec.name, scan_s, leaf_tiles)
            except TileQuotaExhausted as exc:
                # refused mid-scan. drop to collect-only rather than abandoning
                # the country: banked candidates are still perfectly collectable
                self.log.warning("[TILES] %s", exc)
                self._tiles_blocked = True
                allow_discovery = False
                if not discovery.base_scan_complete():
                    return 0
                leaf_tiles = self.db.count_leaf_tiles(
                    rec.name, self.cfg.tile_leaf_zoom)
        else:
            # tile server is refused: use whatever the last complete scan found.
            # a country we never scanned has no candidates to collect either,
            # so it simply waits for tiles to come back
            if not discovery.base_scan_complete():
                return 0
            leaf_tiles = self.db.count_leaf_tiles(rec.name, self.cfg.tile_leaf_zoom)
        quota = compute_quota(leaf_tiles, self.cfg)
        already = self.db.images_in_country(rec.name)
        self.db.upsert_country(rec.name, COUNTRY_IN_PROGRESS,
                               quota=quota, leaf_tiles=leaf_tiles)

        if quota == 0:
            # zero coverage is only believable when discovery actually finished.
            # otherwise the tile server was unavailable and this country still
            # has data waiting -- keep it open rather than writing it off
            if discovery.discovery_complete():
                self.db.upsert_country(rec.name, COUNTRY_EXHAUSTED,
                                       finished_at=utc_now_iso())
                self.log.info("[COUNTRY] %s: no coverage on Mapillary", rec.name)
            else:
                self.log.warning("[COUNTRY] %s: discovery incomplete, leaving open "
                                 "for a later run", rec.name)
            return 0
        if already >= quota:
            self.db.upsert_country(rec.name, COUNTRY_COMPLETED,
                                   finished_at=utc_now_iso())
            return 0

        self.log.info("[COUNTRY] %s: %d leaf tiles -> quota %d (have %d, "
                      "%d tiles used today)", rec.name, leaf_tiles, quota, already,
                      discovery.tiles_used_today())

        collected = already
        skips: Counter = Counter()
        processed = 0
        walked_out = False

        # alternate between walking known candidates and discovering more, so a
        # country is never abandoned while it still has unexplored coverage
        while collected < quota and not self._stop.is_set():
            target = quota * self.cfg.candidate_multiplier
            before_candidates = self.db.candidates_count(rec.name)
            if before_candidates < target and allow_discovery:
                try:
                    discovery.harvest(target)
                except TileQuotaExhausted as exc:
                    # keep whatever was discovered before the refusal and carry
                    # on collecting; discovery resumes once tiles come back
                    self.log.warning("[TILES] %s", exc)
                    self._tiles_blocked = True
                    allow_discovery = False
                after_candidates = self.db.candidates_count(rec.name)
                self.log.info("[TILES] %s: %d candidates, %d leaf fetches this run",
                              rec.name, after_candidates, discovery.leaf_fetches)
            else:
                after_candidates = before_candidates

            before = collected
            collected, processed, walked_out = self._walk_candidates(
                rec, quota, collected, processed, skips)
            if collected >= quota or self._stop.is_set():
                break
            # stop once a full pass adds nothing and discovery found no new
            # candidates either -- otherwise a fully-walked, fully-deduped list
            # gets rescanned from position zero forever with zero progress
            gained_candidates = after_candidates > before_candidates
            if collected == before and not gained_candidates:
                break
            if walked_out and not discovery.can_harvest_more() and not gained_candidates:
                break

        if collected >= quota:
            status = COUNTRY_COMPLETED
        elif allow_discovery and discovery.discovery_complete() and walked_out:
            # every tile fetched, every candidate seen, still short: this is the
            # honest "that is all Mapillary has" case
            status = COUNTRY_EXHAUSTED
        else:
            status = COUNTRY_IN_PROGRESS
        self.db.upsert_country(rec.name, status, finished_at=utc_now_iso())
        elapsed = time.monotonic() - country_started
        gained = collected - already
        self.log.info(
            "[COUNTRY] %s %s: %d/%d images (+%d this run) from %d candidates "
            "in %.1f min at %.1f img/s; tiles used today %d/%d; skips=%s",
            rec.name, status, collected, quota, gained, processed,
            elapsed / 60, gained / max(elapsed, 1e-6),
            discovery.tiles_used_today(), self.cfg.daily_tile_budget, dict(skips))
        return gained

    def _walk_candidates(self, rec: CountryRec, quota: int, collected: int,
                         processed: int, skips: Counter) -> tuple:
        """Walk the candidate list once. Returns (collected, processed, walked_out).

        walked_out is True only when the list was consumed to the end, which is
        what distinguishes "ran out of data" from "stopped early".
        """
        consecutive_errors = 0
        walked_out = True
        group_size = max(1, self.cfg.entity_batch_size)
        started = time.monotonic()
        start_count = collected

        for batch in self.db.iter_candidates(
                rec.name,
                batch=max(group_size * 2, self.cfg.workers * 8),
                exclude_collected=True,
                exclude_panos=not self.cfg.include_panoramas,
                min_quality=self.cfg.min_quality_score):
            if collected >= quota or self._stop.is_set():
                walked_out = False
                break

            # live checks only: duplicates, panoramas and quality are already
            # excluded by the query, so this loop is short
            pending = []
            for (_rank, _trank, image_id, lat, lng,
                 sequence, quality, is_pano) in batch:
                processed += 1
                ok, reason = prefilter_candidate(
                    image_id, lat, lng, sequence, quality, is_pano,
                    self.cfg, self.db)
                if ok:
                    pending.append(image_id)
                else:
                    skips[reason] += 1
            if not pending:
                continue

            for start in range(0, len(pending), group_size):
                if collected >= quota or self._stop.is_set():
                    walked_out = False
                    break
                group = pending[start:start + group_size]
                for kind, payload in self._fetch_group(group, rec, self._pool):
                    if kind == "skip":
                        skips[payload] += 1
                        continue
                    if kind == "error":
                        skips["api_error"] += 1
                        consecutive_errors += 1
                        if consecutive_errors >= self.cfg.max_consecutive_api_errors:
                            raise RuntimeError(
                                f"{consecutive_errors} consecutive API failures. "
                                "Check the Mapillary token, then rerun to resume."
                            )
                        continue
                    consecutive_errors = 0
                    row, data = payload
                    if self._commit_image(row, data):
                        collected += 1
                    else:
                        skips["raced_duplicate"] += 1
                    if collected >= quota:
                        walked_out = False
                        break

            if self._stop.is_set():
                walked_out = False
                break
            if processed % self.cfg.status_every < len(batch):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = (collected - start_count) / elapsed
                self.log.info(
                    "[COUNTRY] %s: %d/%d images, %d candidates seen, staged=%d, "
                    "%.1f img/s, throttle=%.2fs",
                    rec.name, collected, quota, processed, self.staging.count(),
                    rate, self.client.graph_limiter.interval)
        return collected, processed, walked_out

    def _fetch_group(self, image_ids: list, rec: CountryRec, pool) -> list:
        """Metadata for a group in one request, then thumbnails in parallel.

        Splitting the two phases is what makes batching worthwhile: metadata is
        cheap per image once batched, while thumbnails are large and unbatchable,
        so they stay on the worker pool where concurrency actually helps.

        Returns a list of ("ok", (row, bytes)) / ("skip", reason) /
        ("error", detail) so all bookkeeping stays on the main thread.
        """
        try:
            metas = self.client.get_images_batch(image_ids)
        except Exception as exc:
            # a group is only as reliable as its worst member, so a failed group
            # is retried one id at a time. that keeps a single unlucky image
            # from discarding the other forty-nine, and confines the damage of
            # any future API change to one image rather than a whole batch
            if len(image_ids) > 1:
                self.log.warning("[API] group of %d failed (%s: %s); retrying "
                                 "individually", len(image_ids),
                                 type(exc).__name__, exc)
                outcomes = []
                for image_id in image_ids:
                    outcomes.extend(self._fetch_group([image_id], rec, pool))
                return outcomes
            self.log.error("[API] metadata for image %s failed: %s: %s",
                           image_ids[0], type(exc).__name__, exc)
            return [("error", f"{type(exc).__name__}: {exc}")]

        outcomes = []
        downloads = []
        for image_id in image_ids:
            meta = metas.get(str(image_id))
            if meta is None:
                outcomes.append(("skip", "image_gone"))
                continue
            row, reason = evaluate_image(meta, self.cfg, self.db, rec)
            if row is None:
                outcomes.append(("skip", reason))
                continue
            url = meta.get(self.cfg.thumb_field)
            if not url:
                outcomes.append(("skip", "no_thumb_url"))
                continue
            downloads.append((row, url))

        if downloads:
            outcomes.extend(pool.map(
                lambda item: self._download_and_validate(item, rec), downloads))
        return outcomes

    def _download_and_validate(self, item, rec: CountryRec):
        """Worker body. Touches no shared mutable state beyond the HTTP client."""
        row, url = item
        try:
            data = self.client.fetch_image_bytes(url, f"{rec.name} id={row['id']}")
        except MapillaryError as exc:
            self.log.warning("[API] thumbnail failed (%s id=%s): %s",
                             rec.name, row["id"], exc)
            return "skip", "download_failed"
        except Exception as exc:
            self.log.error("[API] unexpected error downloading %s: %s: %s",
                           row["id"], type(exc).__name__, exc)
            return "error", f"{type(exc).__name__}: {exc}"

        try:
            ok, reason, dims = validate_image_bytes(data, self.cfg)
        except Exception as exc:
            return "skip", f"invalid:{type(exc).__name__}"
        if not ok:
            return "skip", f"invalid:{reason}"
        if dims is not None:
            row["width"], row["height"] = dims
        return "ok", (row, data)

    def _commit_image(self, row: dict, data: bytes) -> bool:
        """Register, stage, and pack when a full shard is ready. Main thread only."""
        self._check_disk()
        # workers filter in parallel, so several images from one sequence or one
        # coordinate cell can pass prefilter before any of them reaches the db.
        # recheck here, on the single-threaded commit path, where it is race-free
        seq = row.get("sequence")
        if seq and self.db.sequence_count(seq) >= self.cfg.max_per_sequence:
            return False
        if self.db.coord_taken(row["lat_r"], row["lng_r"]):
            return False
        if not self.db.register_image(row):
            return False
        try:
            self.staging.add(row["id"], data, row)
        except OSError:
            # never leave a row claiming a file that is not on disk
            self.db.remove_image(row["id"])
            raise
        if self.staging.count() >= self.cfg.shard_size:
            self._pack_shard()
        return True

    def _pack_shard(self) -> None:
        """Pack exactly shard_size samples so every shard is uniform."""
        ids = self.staging.complete_ids()[: self.cfg.shard_size]
        if len(ids) < self.cfg.shard_size:
            return
        offset = int(self.db.kv_get("shard_offset", 0) or 0)
        shard_idx = self.db.next_shard_idx(offset)
        path = self.staging.pack_shard(shard_idx, ids)
        if path is None:
            return
        self.db.assign_shard(ids, shard_idx)
        self.db.upsert_shard(shard_idx, SHARD_LOCAL, n_samples=len(ids),
                             filename=path.name)
        self.uploader.submit(shard_idx, path)
        self.log.info("[SHARD] %06d packed with %d samples, queued for upload",
                      shard_idx, len(ids))

    def _check_disk(self) -> None:
        free = disk_free_gb(self.cfg.data_dir)
        if free >= self.cfg.min_free_gb:
            return
        self.log.warning("[DISK] %.1f GB free, waiting for uploads to clear space",
                         free)
        self.uploader.drain(300.0)
        if disk_free_gb(self.cfg.data_dir) < self.cfg.min_free_gb:
            raise RuntimeError(
                f"Disk critically low ({free:.1f} GB free). Free space or lower "
                "min_free_gb, then rerun to resume."
            )

    # ---- waiting for the tile server ----------------------------------

    def _all_terminal(self, countries: list) -> bool:
        terminal = {COUNTRY_COMPLETED, COUNTRY_FAILED, COUNTRY_EXHAUSTED}
        return all(self.db.country_status(rec.name) in terminal
                   for rec in countries)

    def _own_budget_spent(self) -> bool:
        """True when our self-imposed daily cap, not the server, is the blocker."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        used = int(self.db.kv_get(f"tile_budget:{day}", 0) or 0)
        return used >= self.cfg.daily_tile_budget

    def _tiles_available(self) -> bool:
        """One cheap request against a known-covered tile.

        Mapillary documents a 4xx for tile rate limiting but in practice serves
        an HTML page with a 200, so this checks whether a real tile comes back
        rather than trusting any status code.
        """
        lng, lat = TILE_PROBE_LNGLAT
        x, y = lnglat_to_tile(lng, lat, self.cfg.tile_leaf_zoom)
        try:
            self.client.fetch_coverage_tile(self.cfg.tile_leaf_zoom, x, y)
            return True
        except TileUnavailableError:
            return False
        except MapillaryError:
            return False

    def _sleep_interruptibly(self, seconds: float) -> None:
        """Sleep in short slices so Ctrl-C stays responsive."""
        deadline = time.monotonic() + seconds
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(5.0, remaining))

    def _wait_for_tiles(self) -> None:
        """Block until the tile server answers again, then resume discovery.

        The daily allowance is documented but its reset time is not, and the
        error is indistinguishable from an outage, so this probes rather than
        assuming a schedule. When our own cap is what tripped, the UTC day
        boundary is known exactly and we can wait for it directly.
        """
        started = time.monotonic()
        last_heartbeat = 0.0

        if self._own_budget_spent():
            now = datetime.now(timezone.utc)
            midnight = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0)
            wait_s = (midnight - now).total_seconds() + 60
            self.log.info("[TILES] own daily budget spent; sleeping %.1f h until "
                          "the UTC day rolls over, then resuming automatically",
                          wait_s / 3600)
            self._sleep_interruptibly(wait_s)
            if not self._stop.is_set():
                self._tiles_blocked = False
                self.log.info("[TILES] new UTC day, discovery resumed")
            return

        self.log.info("[TILES] server is refusing tiles; probing every %.0f min "
                      "and resuming on its own -- nothing for you to do",
                      self.cfg.tile_retry_interval_s / 60)
        while not self._stop.is_set():
            self._sleep_interruptibly(self.cfg.tile_retry_interval_s)
            if self._stop.is_set():
                return
            if self._tiles_available():
                self._tiles_blocked = False
                self.log.info("[TILES] server responding again after %.1f h, "
                              "resuming discovery", (time.monotonic() - started) / 3600)
                return
            waited = time.monotonic() - started
            if waited - last_heartbeat >= self.cfg.idle_heartbeat_s:
                last_heartbeat = waited
                self.log.info("[TILES] still blocked after %.1f h, still waiting",
                              waited / 3600)

    # ---- shutdown -----------------------------------------------------

    def _finalize(self, reason: str) -> None:
        """Leave a consistent state. Partial staging is kept, never shipped short.

        The staged images stay as loose files and become the start of the next
        shard on the next run, which is what keeps shard sizes uniform.
        """
        if self._finalized:
            return
        self._finalized = True

        if self.staging is not None:
            staged = self.staging.count()
            if staged:
                self.log.info("[SHUTDOWN] %d staged image(s) kept for the next "
                              "run (need %d for a full shard)",
                              staged, self.cfg.shard_size)
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self.uploader is not None:
            if not self.uploader.drain(self.cfg.drain_timeout_s):
                self.log.warning("[UPLOAD] queue not drained; recovery will "
                                 "re-queue on the next run")
            self.uploader.stop()
        if self.client is not None:
            st = getattr(self.client, "stats", None)
            if st:
                per_request = (st["entity_images"] / st["entity_requests"]
                               if st["entity_requests"] else 0)
                self.log.info(
                    "[METRICS] entity: %d request(s) for %d image(s) "
                    "(%.1f per request); tiles: %d; thumbnails: %d (%.2f GB); "
                    "retries: %d",
                    st["entity_requests"], st["entity_images"], per_request,
                    st["tile_requests"], st["thumb_requests"],
                    st["thumb_bytes"] / 1024 ** 3, st["retries"])
        if self.db is not None:
            totals = self.db.totals()
            self.log.info("[SUMMARY] images=%s staged=%d shards=%s countries=%s",
                          human_count(totals["images"]), totals["staged"],
                          totals["shards"], totals["countries"])
            self.db.close()
        self.log.info("[SHUTDOWN] complete (%s)", reason)

    def finalize_partial_shard(self) -> None:
        """Pack whatever is staged into a short final shard.

        Only for `mapillary finalize`, when the user has decided the collection
        is done and wants the remainder shipped.
        """
        ids = self.staging.complete_ids()
        if not ids:
            self.log.info("[SHARD] nothing staged")
            return
        offset = int(self.db.kv_get("shard_offset", 0) or 0)
        shard_idx = self.db.next_shard_idx(offset)
        path = self.staging.pack_shard(shard_idx, ids)
        self.db.assign_shard(ids, shard_idx)
        self.db.upsert_shard(shard_idx, SHARD_LOCAL, n_samples=len(ids),
                             filename=path.name)
        self.uploader.submit(shard_idx, path)
        self.log.info("[SHARD] %06d packed with %d samples (final, short)",
                      shard_idx, len(ids))