"""Startup reconciliation.

Makes the database, the local filesystem and the Hub agree before any new work
starts. Idempotent, so it is safe to run at the start of every session.
"""

from __future__ import annotations

import logging

from .config import Config
from .constants import SHARD_LOCAL, SHARD_NAME_FMT, SHARD_UPLOADED, SHARD_UPLOADING
from .staging import StagingArea
from .state import StateDB
from .upload import HfStore, UploadManager


class RecoveryManager:
    def __init__(self, cfg: Config, db: StateDB, staging: StagingArea,
                 store: HfStore, uploader: UploadManager, log: logging.Logger):
        self.cfg = cfg
        self.db = db
        self.staging = staging
        self.store = store
        self.uploader = uploader
        self.log = log

    def reconcile(self) -> None:
        self._clean_orphan_staging_files()
        self._reconcile_staged_rows()
        self._resolve_pending_uploads()
        self._align_shard_numbering()

    def _clean_orphan_staging_files(self) -> None:
        """A jpg with no json (or vice versa) means a crash mid-write."""
        orphans = self.staging.orphan_ids()
        for image_id in orphans:
            self.staging.remove(image_id)
        if orphans:
            self.log.info("[RECOVERY] removed %d incomplete staging file(s)",
                          len(orphans))
        # leftover .tmp files from an interrupted atomic write
        stale = list(self.staging.dir.glob("*.tmp"))
        for path in stale:
            path.unlink()
        if stale:
            self.log.info("[RECOVERY] removed %d stale temp file(s)", len(stale))

    def _reconcile_staged_rows(self) -> None:
        """Database rows and staging files must describe the same set.

        A row without files can never be packed, so it is dropped and its image
        becomes collectable again. Files without a row are unreferenced, so they
        are deleted rather than smuggled into a shard with no metadata.
        """
        db_ids = set(self.db.staged_ids())
        disk_ids = set(self.staging.complete_ids())

        missing_files = db_ids - disk_ids
        if missing_files:
            self.db.remove_images(missing_files)
            self.log.warning("[RECOVERY] dropped %d staged row(s) with no files "
                             "(images returned to the pool)", len(missing_files))

        unreferenced = disk_ids - db_ids
        for image_id in unreferenced:
            self.staging.remove(image_id)
        if unreferenced:
            self.log.warning("[RECOVERY] removed %d staging file(s) with no row",
                             len(unreferenced))

        remaining = self.staging.count()
        if remaining:
            self.log.info("[RECOVERY] %d staged image(s) carried into this run "
                          "(shard will be filled to %d)",
                          remaining, self.cfg.shard_size)

    def _resolve_pending_uploads(self) -> None:
        """Anything not confirmed on the Hub is re-queued or rolled back."""
        pending = (self.db.shards_with_status(SHARD_UPLOADING)
                   + self.db.shards_with_status(SHARD_LOCAL))
        for shard_idx, filename, n_samples in pending:
            filename = filename or SHARD_NAME_FMT.format(idx=shard_idx)
            local_path = self.cfg.shards_dir / filename
            if self.store.exists(filename):
                # crashed mid-upload but it landed: do not send it twice
                self.db.upsert_shard(shard_idx, SHARD_UPLOADED, filename=filename)
                self.log.info("[RECOVERY] shard %06d already on hub", shard_idx)
                if self.cfg.delete_local_after_upload and local_path.exists():
                    local_path.unlink()
            elif local_path.exists():
                self.uploader.submit(shard_idx, local_path)
                self.log.info("[RECOVERY] re-queued shard %06d (%s samples)",
                              shard_idx, n_samples)
            else:
                # the tar is gone and it never reached the hub: free the ids so
                # those images can be collected again
                ids = self.db.image_ids_in_shard(shard_idx)
                self.db.remove_images(ids)
                self.db.upsert_shard(shard_idx, "lost")
                self.log.warning("[RECOVERY] shard %06d lost before upload; "
                                 "released %d image(s)", shard_idx, len(ids))

    def _align_shard_numbering(self) -> None:
        """Start numbering after whatever is already in the repo.

        Lets a fresh local database share a Hugging Face repo with shards from
        an earlier run without overwriting them.
        """
        if self.db.kv_get("shard_offset") is not None:
            return
        highest = self.store.max_existing_shard_index()
        offset = highest + 1
        self.db.kv_set("shard_offset", offset)
        if offset:
            self.log.info("[RECOVERY] repo already holds shards up to %06d; "
                          "numbering continues from %06d", highest, offset)
