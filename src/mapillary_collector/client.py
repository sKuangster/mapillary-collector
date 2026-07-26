"""All Mapillary network traffic. Nothing else in the package calls requests."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import requests
from vt2geojson.tools import vt_bytes_to_geojson

from .config import Config
from .constants import MAPILLARY_ENTITY_URL, MAPILLARY_FIELDS, MAPILLARY_TILE_URL
from .ratelimit import AdaptiveRateLimiter, parse_retry_after
from .utils import backoff_sleep


class MapillaryError(Exception):
    """Base class. Messages always carry enough context to debug from a log."""


class ApiRequestError(MapillaryError):
    """Retries exhausted, unexpected status, or network failure."""


class BadResponseError(MapillaryError):
    """HTTP said OK but the body was not the shape we expect."""


class MapillaryClient:
    """Thread-safe. One instance is shared by every worker."""

    def __init__(self, cfg: Config, token: str, log: logging.Logger):
        self.cfg = cfg
        self.token = token
        self.log = log
        # graph API and tile server have separate rate budgets, so a slowdown on
        # one must not throttle the other
        self.graph_limiter = AdaptiveRateLimiter(
            cfg.graph_min_interval_s, cfg.graph_max_interval_s,
            cfg.throttle_decay, cfg.throttle_growth)
        self.tile_limiter = AdaptiveRateLimiter(
            cfg.tile_min_interval_s, cfg.tile_max_interval_s,
            cfg.throttle_decay, cfg.throttle_growth)
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        """One Session per thread: connection pooling without cross-thread sharing."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=4, pool_maxsize=8, max_retries=0)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return session

    def get_image(self, image_id: str) -> Optional[dict]:
        """Full metadata for one image, or None when it no longer exists.

        Tile-discovered ids can be stale (images do get deleted), so a 404 here
        is an ordinary outcome, not an error worth retrying.
        """
        ctx = f"entity id={image_id}"
        resp = self._request(
            MAPILLARY_ENTITY_URL.format(image_id=image_id),
            {"access_token": self.token, "fields": ",".join(MAPILLARY_FIELDS)},
            ctx, self.graph_limiter, allow_404=True,
        )
        if resp is None:
            return None
        try:
            payload = resp.json()
        except ValueError as exc:
            raise BadResponseError(f"malformed json ({ctx})") from exc
        if not isinstance(payload, dict) or "id" not in payload:
            raise BadResponseError(f"malformed image record ({ctx})")
        return payload

    def fetch_coverage_tile(self, z: int, x: int, y: int) -> Optional[dict]:
        """Decoded coverage tile as GeoJSON, or None when the tile holds nothing."""
        ctx = f"tile z={z} x={x} y={y}"
        resp = self._request(
            MAPILLARY_TILE_URL.format(z=z, x=x, y=y),
            {"access_token": self.token}, ctx, self.tile_limiter, allow_404=True,
        )
        if resp is None or not resp.content:
            return None
        try:
            return vt_bytes_to_geojson(resp.content, x, y, z)
        except Exception as exc:  # protobuf decode failures come in many flavors
            raise BadResponseError(
                f"tile decode failed ({ctx}): {type(exc).__name__}") from exc

    def fetch_image_bytes(self, url: str, ctx: str) -> bytes:
        """Download thumbnail bytes from the CDN. Retried, but not rate-limited:
        CDN traffic does not count against the graph API budget."""
        last = "unknown"
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.get(url, timeout=self.cfg.request_timeout_s)
            except requests.RequestException as exc:
                last = f"network:{type(exc).__name__}"
                backoff_sleep(attempt, self.cfg.backoff_base_s, self.cfg.backoff_cap_s)
                continue
            if resp.status_code == 429:
                last = "429"
                backoff_sleep(attempt, self.cfg.backoff_base_s,
                              self.cfg.backoff_cap_s, parse_retry_after(resp))
                continue
            if resp.status_code >= 500:
                last = f"http:{resp.status_code}"
                backoff_sleep(attempt, self.cfg.backoff_base_s, self.cfg.backoff_cap_s)
                continue
            if not resp.ok:
                # thumbnail URLs expire; a fresh entity call mints a new one
                raise ApiRequestError(f"thumb http {resp.status_code} ({ctx})")
            if not resp.content:
                last = "empty_body"
                backoff_sleep(attempt, self.cfg.backoff_base_s, self.cfg.backoff_cap_s)
                continue
            return resp.content
        raise ApiRequestError(f"thumb retries exhausted ({ctx}); last={last}")

    def _request(self, url: str, params: dict, ctx: str,
                 limiter: AdaptiveRateLimiter,
                 allow_404: bool = False) -> Optional[requests.Response]:
        """Throttled GET with retries. Returns None only for an allowed 404."""
        last = "unknown"
        for attempt in range(self.cfg.max_retries):
            limiter.wait()
            try:
                resp = self.session.get(url, params=params,
                                        timeout=self.cfg.request_timeout_s)
            except requests.RequestException as exc:
                last = f"network:{type(exc).__name__}"
                limiter.penalize()
                backoff_sleep(attempt, self.cfg.backoff_base_s, self.cfg.backoff_cap_s)
                continue
            if resp.status_code == 404 and allow_404:
                limiter.on_success()  # server answered fine, the thing is just gone
                return None
            if resp.status_code == 429:
                last = "429"
                retry_after = parse_retry_after(resp)
                self.log.warning("[RATELIMIT] 429 attempt %d (%s) retry_after=%s",
                                 attempt + 1, ctx, retry_after)
                limiter.penalize()
                backoff_sleep(attempt, self.cfg.backoff_base_s,
                              self.cfg.backoff_cap_s, retry_after)
                continue
            if resp.status_code >= 500:
                last = f"http:{resp.status_code}"
                limiter.penalize()
                backoff_sleep(attempt, self.cfg.backoff_base_s, self.cfg.backoff_cap_s)
                continue
            if not resp.ok:
                raise ApiRequestError(
                    f"http {resp.status_code} ({ctx}): {resp.text[:200]}")
            limiter.on_success()
            return resp
        raise ApiRequestError(f"retries exhausted ({ctx}); last={last}")
