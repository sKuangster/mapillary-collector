"""Adaptive global throttling, shared by every worker thread."""

from __future__ import annotations

import threading
import time
from typing import Optional

import requests


class AdaptiveRateLimiter:
    """Paces requests globally: gentle when the API is happy, cautious when not.

    A fixed sleep cannot react to the server. This shrinks the interval on
    success and doubles it on 429/5xx, so one rate-limit response slows the
    entire pipeline rather than just the request that tripped it.
    """

    def __init__(self, min_interval: float, max_interval: float,
                 decay: float, growth: float):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.decay = decay
        self.growth = growth
        self.interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until it is polite to send the next request."""
        with self._lock:
            now = time.monotonic()
            sleep_for = self._last + self.interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()

    def on_success(self) -> None:
        with self._lock:
            self.interval = max(self.min_interval, self.interval * self.decay)

    def penalize(self) -> None:
        with self._lock:
            self.interval = min(
                self.max_interval,
                max(self.interval, self.min_interval) * self.growth,
            )


def parse_retry_after(resp: "requests.Response") -> Optional[float]:
    """Seconds from a Retry-After header; None when absent or an HTTP-date."""
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
