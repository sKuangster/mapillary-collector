"""All Mapillary network traffic. Nothing else in the package calls requests."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import requests
from vt2geojson.tools import vt_bytes_to_geojson

from .config import Config
from .constants import (
    MAPILLARY_ENTITY_URL,
    MAPILLARY_FIELDS,
    MAPILLARY_IMAGES_URL,
    MAPILLARY_TILE_URL,
)
from .ratelimit import AdaptiveRateLimiter, parse_retry_after
from .utils import backoff_sleep


class MapillaryError(Exception):
    """Base class. Messages always carry enough context to debug from a log."""


class ApiRequestError(MapillaryError):
    """Retries exhausted, unexpected status, or network failure."""


class BadResponseError(MapillaryError):
    """HTTP said OK but the body was not the shape we expect."""


class TileUnavailableError(MapillaryError):
    """A tile could not be read *right now* -- quota, throttling, server hiccup.

    Deliberately distinct from a permanent failure. The caller must leave such a
    tile pending so a later run retries it; recording it as dead is how a bad
    hour turns into a country permanently marked as having no coverage.
    """


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
        self._fields = ",".join(MAPILLARY_FIELDS)
        # batching is documented but its exact behaviour is not guaranteed, so
        # it is treated as an optimisation that can switch itself off
        self._batch_ok = cfg.use_entity_batching
        self._batch_failures = 0
        self._lock = threading.Lock()
        # cheap counters for the end-of-run diagnostics
        self.stats = {"entity_requests": 0, "entity_images": 0,
                      "tile_requests": 0, "thumb_requests": 0,
                      "thumb_bytes": 0, "retries": 0}

    @property
    def session(self) -> requests.Session:
        """One Session per thread: connection pooling without cross-thread sharing."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            # one pooled connection per worker, plus headroom: a pool smaller
            # than the worker count silently serialises requests
            pool = max(8, self.cfg.workers + 4)
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=pool, pool_maxsize=pool, max_retries=0)
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
            {"access_token": self.token, "fields": self._fields},
            ctx, self.graph_limiter, allow_404=True,
        )
        if resp is None:
            return None
        with self._lock:
            self.stats["entity_requests"] += 1
            self.stats["entity_images"] += 1
        try:
            payload = resp.json()
        except ValueError as exc:
            raise BadResponseError(f"malformed json ({ctx})") from exc
        if not isinstance(payload, dict) or "id" not in payload:
            raise BadResponseError(f"malformed image record ({ctx})")
        return payload

    def get_images_batch(self, image_ids: list) -> dict:
        """Metadata for many images at once, as {image_id: meta}.

        One request replaces up to entity_batch_size single-image requests.
        Ids that no longer exist are simply absent from the result, which is
        the same outcome as a 404 on the single-image path.

        If the endpoint ever misbehaves this degrades to individual lookups
        permanently for the rest of the run, so a batching regression can never
        turn into a collection outage.
        """
        if not image_ids:
            return {}
        if not self._batch_ok or len(image_ids) == 1:
            return self._get_images_individually(image_ids)

        ctx = f"batch of {len(image_ids)}"
        try:
            resp = self._request(
                MAPILLARY_IMAGES_URL,
                {"access_token": self.token, "fields": self._fields,
                 "image_ids": ",".join(image_ids)},
                ctx, self.graph_limiter, allow_404=True,
            )
            if resp is None:
                raise BadResponseError(f"404 on {ctx}")
            payload = resp.json()
            records = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(records, list):
                raise BadResponseError(f"no data list in {ctx}")
        except Exception as exc:
            with self._lock:
                self._batch_failures += 1
                give_up = self._batch_failures >= 2
                if give_up and self._batch_ok:
                    self._batch_ok = False
            self.log.warning("[API] batched lookup failed (%s): %s%s", ctx, exc,
                             "; falling back to single lookups for this run"
                             if not self._batch_ok else "")
            return self._get_images_individually(image_ids)

        with self._lock:
            self.stats["entity_requests"] += 1
            self.stats["entity_images"] += len(records)
        out = {}
        for record in records:
            if isinstance(record, dict) and "id" in record:
                out[str(record["id"])] = record
        return out

    def _get_images_individually(self, image_ids: list) -> dict:
        """Per-image lookups with per-image isolation.

        Batching means a single malformed record could otherwise take down the
        whole group, so failures are contained here: one bad id is dropped and
        the rest still come back. Only a total failure is escalated, which is
        what the caller's consecutive-error breaker is meant to catch.
        """
        out = {}
        failures = 0
        last_error: Optional[Exception] = None
        for image_id in image_ids:
            try:
                meta = self.get_image(image_id)
            except MapillaryError as exc:
                failures += 1
                last_error = exc
                continue
            except Exception as exc:
                failures += 1
                last_error = exc
                self.log.error("[API] unexpected error on image %s: %s: %s",
                               image_id, type(exc).__name__, exc)
                continue
            if meta is not None:
                out[str(image_id)] = meta
        if failures and not out:
            raise ApiRequestError(
                f"all {failures} lookup(s) in this group failed; "
                f"last error: {last_error}")
        if failures:
            self.log.warning("[API] %d of %d lookup(s) failed; continuing with "
                             "the rest", failures, len(image_ids))
        return out

    def fetch_coverage_tile(self, z: int, x: int, y: int,
                            layer: Optional[str] = None) -> Optional[dict]:
        """Decoded coverage tile as GeoJSON, or None when the tile holds nothing."""
        ctx = f"tile z={z} x={x} y={y}"
        resp = self._request(
            MAPILLARY_TILE_URL.format(z=z, x=x, y=y),
            {"access_token": self.token}, ctx, self.tile_limiter, allow_404=True,
        )
        with self._lock:
            self.stats["tile_requests"] += 1
        if resp is None or not resp.content:
            return None
        body = resp.content
        # a vector tile is protobuf. an HTML or JSON body here means the server
        # answered with an error page or a quota notice while still saying 200,
        # so treat it as transient rather than as an empty tile
        if body[:1] in (b"<", b"{"):
            self.tile_limiter.penalize()
            snippet = body[:80].decode("utf-8", "replace").replace("\n", " ")
            raise TileUnavailableError(f"non-tile body ({ctx}): {snippet!r}")
        try:
            # decoding one named layer skips parsing the others entirely
            if layer is not None:
                return vt_bytes_to_geojson(body, x, y, z, layer=layer)
            return vt_bytes_to_geojson(body, x, y, z)
        except Exception as exc:  # protobuf decode failures come in many flavors
            self.tile_limiter.penalize()
            raise TileUnavailableError(
                f"tile undecodable ({ctx}): {type(exc).__name__}") from exc

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
            with self._lock:
                self.stats["thumb_requests"] += 1
                self.stats["thumb_bytes"] += len(resp.content)
            return resp.content
        raise ApiRequestError(f"thumb retries exhausted ({ctx}); last={last}")

    def _request(self, url: str, params: dict, ctx: str,
                 limiter: AdaptiveRateLimiter,
                 allow_404: bool = False) -> Optional[requests.Response]:
        """Throttled GET with retries. Returns None only for an allowed 404."""
        last = "unknown"
        for attempt in range(self.cfg.max_retries):
            if attempt:
                with self._lock:
                    self.stats["retries"] += 1
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