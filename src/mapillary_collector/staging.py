"""Staging area and shard packing.

Every validated image lands in the staging directory as two loose files
(`{id}.jpg`, `{id}.json`), each written temp-then-rename so a file on disk is
never partial. Only when staging holds a full `shard_size` do those files get
packed into a tar.

This is what makes shard sizes uniform. Writing straight into an open tar means
a crash produces a short, torn shard; here a crash just leaves N complete loose
files, and the next run keeps filling toward a full shard. Every shard is
exactly shard_size samples except the final one of a completed run.
"""

from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path
from typing import Optional

from .config import Config
from .constants import SHARD_NAME_FMT
from .utils import atomic_write_bytes, atomic_write_text


class StagingArea:
    """Loose validated samples waiting to be packed."""

    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self.dir = cfg.staging_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg.shards_dir.mkdir(parents=True, exist_ok=True)

    def add(self, image_id: str, img_bytes: bytes, metadata: dict) -> None:
        """Write the image first, then its metadata.

        Order matters for recovery: the json is the commit marker, so a crash
        between the two leaves a stray jpg that reconcile() cleans up, never a
        json claiming an image that is not there.
        """
        atomic_write_bytes(self.dir / f"{image_id}.jpg", img_bytes)
        atomic_write_text(self.dir / f"{image_id}.json",
                          json.dumps(metadata, ensure_ascii=False))

    def remove(self, image_id: str) -> None:
        for suffix in (".jpg", ".json"):
            path = self.dir / f"{image_id}{suffix}"
            if path.exists():
                path.unlink()

    def complete_ids(self) -> list:
        """Ids that have both files present, sorted for deterministic packing."""
        jpgs = {p.stem for p in self.dir.glob("*.jpg")}
        jsons = {p.stem for p in self.dir.glob("*.json")}
        return sorted(jpgs & jsons)

    def orphan_ids(self) -> list:
        """Ids missing one of the two files -- a crash mid-write."""
        jpgs = {p.stem for p in self.dir.glob("*.jpg")}
        jsons = {p.stem for p in self.dir.glob("*.json")}
        return sorted(jpgs ^ jsons)

    def count(self) -> int:
        return len(self.complete_ids())

    def pack_shard(self, shard_idx: int, image_ids: list) -> Optional[Path]:
        """Pack exactly these ids into a tar, then delete the loose files.

        The tar is built under a .tmp name and renamed on success, so a crash
        mid-pack cannot leave a half-written shard that looks finished.
        """
        if not image_ids:
            return None
        filename = SHARD_NAME_FMT.format(idx=shard_idx)
        final_path = self.cfg.shards_dir / filename
        tmp_path = final_path.with_suffix(".tar.tmp")

        with tarfile.open(tmp_path, "w") as tar:
            # sorted so an image and its json are adjacent: WebDataset pairs
            # consecutive members sharing a basename
            for image_id in sorted(image_ids):
                for suffix in (".jpg", ".json"):
                    source = self.dir / f"{image_id}{suffix}"
                    tar.add(source, arcname=f"{image_id}{suffix}")
        tmp_path.replace(final_path)

        for image_id in image_ids:
            self.remove(image_id)
        return final_path


def inspect_tar(path: Path) -> Optional[set]:
    """Sample keys inside a tar, or None when unreadable. Used by diagnostics."""
    try:
        with tarfile.open(path) as tar:
            names = tar.getnames()
    except (tarfile.TarError, OSError, EOFError):
        return None
    return {name.rsplit(".", 1)[0] for name in names}
