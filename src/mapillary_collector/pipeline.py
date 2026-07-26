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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from shapely.prepared import prep

from .client import MapillaryClient, MapillaryError
from .config import Config, resolve_token
from .constants import (
    COUNTRY_COMPLETED,
    COUNTRY_EXHAUSTED,
    COUNTRY_FAILED,
    COUNTRY_IN_PROGRESS,
    SHARD_LOCAL,
)
from .discovery import TileDiscovery, TileQuotaExhausted
from .filters import evaluate_image, prefilter_candidate, validate_image_bytes
from .geo import CountryRec, load_countries
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

        self.db = StateDB(self.cfg.db_path)
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
        RecoveryManager(self.cfg, self.db, self.staging, self.store,
                        self.uploader, self.log).reconcile()

    def run(self) -> None:
        self._install_signal_handlers()
        self.startup()
        log_config_summary(self.cfg, self.log)
        countries = load_countries(self.cfg, self.log)
        try:
            self._run_countries(countries)
            if self._stop.is_set():
                self.log.info("[SHUTDOWN] stopped by request")
            else:
                self.log.info("[DONE] all countries processed")
            self._finalize("complete")
        except TileQuotaExhausted as exc:
            self.log.error("[TILES] %s", exc)
            self.log.error("[TILES] stopping cleanly. No country was marked empty; "
                           "rerun later and discovery resumes where it stopped.")
            self._finalize("tile_quota")
        except (KeyboardInterrupt, ShutdownRequested):
            self.log.warning("[SHUTDOWN] interrupted, saving state")
            self._finalize("interrupt")
        except Exception:
            self.log.exception("[ERROR] fatal error, saving what is salvageable")
            self._finalize("error")
            raise

    def _run_countries(self, countries: list) -> None:
        self._reopen_empty_countries()
        skip = {COUNTRY_COMPLETED, COUNTRY_FAILED}
        if not self.cfg.retry_exhausted:
            skip.add(COUNTRY_EXHAUSTED)
        for rec in countries:
            if self._stop.is_set():
                return
            if self.db.country_status(rec.name) in skip:
                continue
            if rec.geometry is None or rec.geometry.is_empty:
                self.db.upsert_country(rec.name, COUNTRY_FAILED,
                                       iso3=rec.iso3, continent=rec.continent)
                self.log.warning("[COUNTRY] %s has no usable polygon", rec.name)
                continue
            self._collect_country(rec)

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

    def _collect_country(self, rec: CountryRec) -> None:
        prepared = prep(rec.geometry)
        discovery = TileDiscovery(self.cfg, self.client, self.db, rec,
                                  prepared, self.log)

        self.db.upsert_country(rec.name, COUNTRY_IN_PROGRESS, iso3=rec.iso3,
                               continent=rec.continent, started_at=utc_now_iso())

        leaf_tiles = discovery.base_scan()
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
            return
        if already >= quota:
            self.db.upsert_country(rec.name, COUNTRY_COMPLETED,
                                   finished_at=utc_now_iso())
            return

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
            if self.db.candidates_count(rec.name) < target:
                discovery.harvest(target)
                self.log.info("[TILES] %s: %d candidates, %d leaf fetches this run",
                              rec.name, self.db.candidates_count(rec.name),
                              discovery.leaf_fetches)

            before = collected
            collected, processed, walked_out = self._walk_candidates(
                rec, quota, collected, processed, skips)
            if collected >= quota or self._stop.is_set():
                break
            if not discovery.can_harvest_more():
                break
            if collected == before and not walked_out:
                break  # neither collecting nor discovering: stop rather than spin

        if collected >= quota:
            status = COUNTRY_COMPLETED
        elif discovery.discovery_complete() and walked_out:
            # every tile fetched, every candidate seen, still short: this is the
            # honest "that is all Mapillary has" case
            status = COUNTRY_EXHAUSTED
        else:
            status = COUNTRY_IN_PROGRESS
        self.db.upsert_country(rec.name, status, finished_at=utc_now_iso())
        self.log.info("[COUNTRY] %s %s: %d/%d images from %d candidates; skips=%s",
                      rec.name, status, collected, quota, processed, dict(skips))

    def _walk_candidates(self, rec: CountryRec, quota: int, collected: int,
                         processed: int, skips: Counter) -> tuple:
        """Walk the candidate list once. Returns (collected, processed, walked_out).

        walked_out is True only when the list was consumed to the end, which is
        what distinguishes "ran out of data" from "stopped early".
        """
        consecutive_errors = 0
        walked_out = True
        with ThreadPoolExecutor(max_workers=self.cfg.workers,
                                thread_name_prefix="fetch") as pool:
            for batch in self.db.iter_candidates(rec.name,
                                                 batch=self.cfg.workers * 8):
                if collected >= quota or self._stop.is_set():
                    walked_out = False
                    break

                # prefilter on the main thread: pure indexed lookups, no network,
                # so rejected candidates never occupy a worker
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

                for outcome in pool.map(lambda i: self._fetch_one(i, rec), pending):
                    if outcome is None:
                        continue
                    kind, payload = outcome
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
                    self.log.info("[COUNTRY] %s: %d/%d images, %d candidates seen, "
                                  "staged=%d, interval=%.2fs",
                                  rec.name, collected, quota, processed,
                                  self.staging.count(),
                                  self.client.graph_limiter.interval)
        return collected, processed, walked_out

    def _fetch_one(self, image_id: str, rec: CountryRec):
        """Worker body: metadata, filters, download, validation. No shared state.

        Returns ("ok", (row, bytes)) / ("skip", reason) / ("error", detail) so
        the main thread can do all the bookkeeping.
        """
        try:
            meta = self.client.get_image(image_id)
        except MapillaryError as exc:
            self.log.error("[API] %s", exc)
            return "error", str(exc)
        except Exception as exc:
            # a worker raising anything unexpected would propagate out of
            # pool.map and end the whole run; one bad candidate is not worth that
            self.log.error("[API] unexpected error on %s: %s: %s",
                           image_id, type(exc).__name__, exc)
            return "error", f"{type(exc).__name__}: {exc}"
        if meta is None:
            return "skip", "image_gone"  # deleted since the tile was built

        row, reason = evaluate_image(meta, self.cfg, self.db, rec)
        if row is None:
            return "skip", reason

        url = meta.get(self.cfg.thumb_field)
        if not url:
            return "skip", "no_thumb_url"
        try:
            data = self.client.fetch_image_bytes(url, f"{rec.name} id={image_id}")
        except MapillaryError as exc:
            self.log.warning("[API] thumb failed (%s): %s", image_id, exc)
            return "skip", "download_failed"

        try:
            ok, vreason, dims = validate_image_bytes(data, self.cfg)
        except Exception as exc:
            return "skip", f"invalid:{type(exc).__name__}"
        if not ok:
            return "skip", f"invalid:{vreason}"
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
        if self.uploader is not None:
            if not self.uploader.drain(self.cfg.drain_timeout_s):
                self.log.warning("[UPLOAD] queue not drained; recovery will "
                                 "re-queue on the next run")
            self.uploader.stop()
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
