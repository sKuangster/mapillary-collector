"""Hugging Face upload, isolated from collection.

Uploads run on a background thread so a slow shard upload never stalls
fetching. Every upload checks the Hub first, so a shard is never sent twice --
which is what makes a crash mid-upload safe.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from .config import Config
from .constants import SHARD_LOCAL, SHARD_NAME_FMT, SHARD_UPLOADED, SHARD_UPLOADING
from .state import StateDB
from .utils import backoff_sleep


class HfStore:
    """Thin wrapper around the Hub. Also where dry-run lives."""

    def __init__(self, cfg: Config, token: Optional[str], log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self.api = HfApi(token=token)

    def remote_path(self, filename: str) -> str:
        return f"{self.cfg.hf_images_prefix}/{filename}"

    def ensure_repo(self) -> None:
        if self.cfg.dry_run_uploads:
            return
        self.api.create_repo(
            repo_id=self.cfg.hf_repo_id,
            repo_type="dataset",
            private=self.cfg.hf_repo_private,
            exist_ok=True,
        )

    def list_shard_files(self) -> list:
        if self.cfg.dry_run_uploads:
            return []
        try:
            files = self.api.list_repo_files(self.cfg.hf_repo_id, repo_type="dataset")
        except Exception as exc:
            self.log.warning("[UPLOAD] could not list repo files: %s", exc)
            return []
        prefix = f"{self.cfg.hf_images_prefix}/"
        return [f for f in files if f.startswith(prefix) and f.endswith(".tar")]

    def max_existing_shard_index(self) -> int:
        """Highest shard number already on the Hub, or -1 when none.

        New runs start numbering after this so a fresh local database can share
        a repo with shards uploaded by an earlier run.
        """
        best = -1
        for path in self.list_shard_files():
            stem = Path(path).stem
            if not stem.startswith("shard-"):
                continue
            try:
                best = max(best, int(stem.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
        return best

    def exists(self, filename: str) -> bool:
        if self.cfg.dry_run_uploads:
            return False
        target = self.remote_path(filename)
        try:
            return bool(self.api.file_exists(
                repo_id=self.cfg.hf_repo_id, filename=target, repo_type="dataset"))
        except (AttributeError, TypeError):
            return target in set(self.api.list_repo_files(
                self.cfg.hf_repo_id, repo_type="dataset"))

    def upload(self, local_path: Path, filename: str) -> None:
        if self.cfg.dry_run_uploads:
            self.log.info("[UPLOAD] dry-run: would upload %s", filename)
            return
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=self.remote_path(filename),
            repo_id=self.cfg.hf_repo_id,
            repo_type="dataset",
        )


class UploadManager:
    """Background uploader with retry, verification and idempotency."""

    def __init__(self, cfg: Config, db: StateDB, store: HfStore,
                 log: logging.Logger):
        self.cfg = cfg
        self.db = db
        self.store = store
        self.log = log
        self._queue: "queue.Queue" = queue.Queue()
        self._pending = 0
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="shard-uploader")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def submit(self, shard_idx: int, path: Path) -> None:
        with self._cond:
            self._pending += 1
        self._queue.put((shard_idx, path))

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            shard_idx, path = item
            try:
                self._upload_one(shard_idx, path)
            except Exception as exc:  # the worker must survive anything
                self.log.error("[UPLOAD] shard %06d unexpected failure: %s",
                               shard_idx, exc)
                self.db.upsert_shard(shard_idx, SHARD_LOCAL)
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()
                self._queue.task_done()

    def _upload_one(self, shard_idx: int, path: Path) -> None:
        filename = path.name
        self.db.upsert_shard(shard_idx, SHARD_UPLOADING, filename=filename)
        for attempt in range(self.cfg.upload_retries):
            try:
                if self.store.exists(filename):
                    break  # a previous attempt landed; never upload twice
                self.store.upload(path, filename)
                if self.cfg.dry_run_uploads or self.store.exists(filename):
                    break
                raise RuntimeError("post-upload verification failed")
            except Exception as exc:
                self.log.warning("[UPLOAD] shard %06d attempt %d/%d failed: %s",
                                 shard_idx, attempt + 1, self.cfg.upload_retries, exc)
                backoff_sleep(attempt, 2.0, 120.0)
        else:
            # left as local so the next run's recovery re-queues it
            self.db.upsert_shard(shard_idx, SHARD_LOCAL, filename=filename)
            self.log.error("[UPLOAD] shard %06d gave up after %d attempts; "
                           "will retry next run", shard_idx, self.cfg.upload_retries)
            return

        self.db.upsert_shard(shard_idx, SHARD_UPLOADED, filename=filename)
        self.log.info("[UPLOAD] shard %06d on hub (%s)", shard_idx, filename)
        if self.cfg.delete_local_after_upload and not self.cfg.dry_run_uploads:
            try:
                path.unlink()  # verified on the hub; the local copy just eats disk
            except OSError as exc:
                self.log.warning("[UPLOAD] could not delete %s: %s", path, exc)

    def drain(self, timeout_s: float) -> bool:
        """Wait for the queue to empty. False when the timeout hit first."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(min(5.0, remaining))
        return True

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)
