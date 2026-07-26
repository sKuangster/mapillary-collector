"""Small shared helpers. Everything here is pure or filesystem-local."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_key(*parts: Any) -> float:
    """Deterministic pseudo-random float in [0, 1) from any parts.

    Python's hash() is salted per process, so it cannot produce the same
    ordering across restarts. This can, which is what makes tile order and
    therefore sampling reproducible session to session.
    """
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") / 2 ** 64


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via temp file then rename, so a crash never leaves a partial file.

    This is the guarantee the staging directory depends on: every file in it is
    either absent or complete.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def backoff_sleep(attempt: int, base: float, cap: float,
                  retry_after: Optional[float] = None) -> float:
    """Exponential backoff with jitter, honoring Retry-After when present.

    Half the delay is fixed and half randomized: avoids synchronized retry
    storms across worker threads without ever retrying sooner than intended.
    """
    delay = min(cap, base * (2 ** attempt))
    delay = delay / 2 + random.uniform(0, delay / 2)
    if retry_after is not None:
        delay = max(delay, retry_after)
    time.sleep(delay)
    return delay


def human_count(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
