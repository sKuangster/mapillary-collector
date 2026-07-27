"""Coverage-first discovery.

Instead of guessing coordinates and hoping an image is nearby, read Mapillary's
own coverage tiles to learn where images actually are, then spend graph API
calls only on images already known to exist.

Two cached phases per country:
  1. base scan (coarse zoom) -- which parts of the country have coverage at all
  2. leaf harvest (fine zoom) -- the actual image ids, with enough metadata to
     pre-filter before spending an API call

The central invariant: a tile that failed *transiently* is never recorded.
Recording it would remove it from the pending set forever, so one bad hour on
the tile server would permanently mark a country as having no coverage. Only
genuinely permanent failures get written down.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .client import MapillaryClient, MapillaryError, TileUnavailableError
from .config import Config
from .constants import (
    PARENT_TILE_ZOOM,
    TILE_EMPTY,
    TILE_ERROR,
    TILE_FETCHED,
    TILE_LAYER_IMAGE,
    TILE_LAYER_SEQUENCE,
)
from .geo import (
    CountryRec,
    clip_tiles_to_geometry,
    contains_point,
    tiles_over_geometry,
    tiles_touching_geometry,
)
from .state import StateDB
from .utils import stable_key


class TileQuotaExhausted(Exception):
    """The tile server is refusing us, or we hit our own daily budget.

    Raised to stop the whole run rather than grind through thousands of doomed
    requests. Everything collected so far is saved; the next run resumes.
    """


class TileDiscovery:
    """Builds a country's candidate list from coverage tiles.

    Determinism: tiles are ranked by stable_key(seed, country, tile), so the
    same seed produces the same tile order in every session -- which is what
    makes the resulting sample reproducible.
    """

    def __init__(self, cfg: Config, client: MapillaryClient, db: StateDB,
                 rec: CountryRec, prepared, log: logging.Logger):
        self.cfg = cfg
        self.client = client
        self.db = db
        self.rec = rec
        self.prepared = prepared
        self.log = log
        self.leaf_fetches = 0
        self.unavailable = 0          # transient failures this session
        self._attempted: set = set()  # tiles tried this session, so we do not respin
        # how many candidates each sequence has already contributed. a sequence
        # at its cap can never yield another image, so storing more of it wastes
        # the tile's candidate slots
        self._seq_budget: dict = {}
        self._base_done_key = f"base_done:{rec.name}"
        # half a leaf tile per interpolation step, so no leaf lying between two
        # simplified vertices of a coverage line is skipped
        self._step_deg = 360.0 / (2 ** cfg.tile_leaf_zoom) / 2.0

    # ---- daily budget -------------------------------------------------

    def _budget_key(self) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"tile_budget:{day}"

    def _spend_budget(self) -> None:
        """Count one tile request against today's self-imposed budget.

        Mapillary's tile allowance is daily. Stopping ourselves a little short
        of it beats discovering the limit by having every request answered with
        an HTML error page.
        """
        key = self._budget_key()
        used = int(self.db.kv_get(key, 0) or 0) + 1
        self.db.kv_set(key, used)
        if used > self.cfg.daily_tile_budget:
            raise TileQuotaExhausted(
                f"self-imposed daily tile budget reached ({used} requests today). "
                "Resumes automatically after UTC midnight."
            )

    def tiles_used_today(self) -> int:
        return int(self.db.kv_get(self._budget_key(), 0) or 0)

    def _note_unavailable(self) -> None:
        """A transient tile failure. Enough of them means stop, not skip."""
        self.unavailable += 1
        if self.unavailable >= self.cfg.max_tile_failures:
            raise TileQuotaExhausted(
                f"{self.unavailable} unreadable tile responses. The tile server is "
                "refusing us (usually the daily allowance). Nothing was marked "
                "empty; rerun later and discovery continues where it stopped."
            )

    # ---- phase 1: base scan -------------------------------------------

    def base_scan(self) -> int:
        """Find every leaf tile with coverage. Returns the leaf tile count.

        base_done is only set when every base tile was answered. A partial scan
        stays unfinished, because trusting a partial scan is exactly how a
        country with real coverage ends up with a quota of zero.
        """
        if self.db.kv_get(self._base_done_key):
            return self.db.count_leaf_tiles(self.rec.name, self.cfg.tile_leaf_zoom)

        zoom = self.cfg.tile_base_zoom
        base_tiles = tiles_over_geometry(self.rec.geometry, zoom, self.prepared)
        base_tiles.sort(key=lambda t: stable_key(
            self.cfg.rng_seed, self.rec.name, zoom, t[0], t[1]))

        todo = [(x, y) for x, y in base_tiles
                if self.db.tile_status(self.rec.name, zoom, x, y) is None]
        if todo:
            self.log.info("[TILES] %s: base scan, %d/%d tiles at z%d remaining",
                          self.rec.name, len(todo), len(base_tiles), zoom)

        incomplete = False
        for x, y in todo:
            rank = int(stable_key(self.cfg.rng_seed, self.rec.name, zoom, x, y)
                       * 1_000_000_000)
            self._spend_budget()
            try:
                collection = self.client.fetch_coverage_tile(
                    zoom, x, y, layer=TILE_LAYER_SEQUENCE)
            except TileUnavailableError as exc:
                # deliberately NOT recorded: it must stay eligible for retry
                self.log.warning("[TILES] base z%d/%d/%d unavailable: %s",
                                 zoom, x, y, exc)
                incomplete = True
                self._note_unavailable()
                continue
            except MapillaryError as exc:
                self.log.warning("[TILES] base z%d/%d/%d permanent failure: %s",
                                 zoom, x, y, exc)
                self.db.record_tile(self.rec.name, zoom, x, y, TILE_ERROR, rank)
                continue
            leaves = self._leaves_from_collection(collection)
            if leaves:
                self.db.add_pending_tiles(self.rec.name, leaves)
                self.db.record_tile(self.rec.name, zoom, x, y, TILE_FETCHED,
                                    rank, len(leaves))
            else:
                self.db.record_tile(self.rec.name, zoom, x, y, TILE_EMPTY, rank)

        if incomplete:
            self.log.warning("[TILES] %s: base scan incomplete, leaving it open "
                             "for the next run", self.rec.name)
        else:
            self.db.kv_set(self._base_done_key, True)
        return self.db.count_leaf_tiles(self.rec.name, self.cfg.tile_leaf_zoom)

    def base_scan_complete(self) -> bool:
        return bool(self.db.kv_get(self._base_done_key))

    def _leaves_from_collection(self, collection) -> list:
        """Coverage geometry -> the leaf tiles it touches, clipped to this country.

        Two things make this fast enough for large countries. Tiles are derived
        by walking each segment in tile space instead of interpolating at a
        fixed angular step, and containment is then decided hierarchically on
        coarse parent tiles so that interior tiles are admitted in bulk.
        Together these measured ~7x faster than the point-by-point approach,
        which matters because a single z6 base tile can imply tens of thousands
        of leaf tiles.

        Leaf tiles are ranked across the country's whole leaf set, not per base
        tile, so even a partial harvest is spread over the entire country.
        """
        if not collection or not collection.get("features"):
            return []
        leaf_zoom = self.cfg.tile_leaf_zoom

        touched: set = set()
        for feature in collection["features"]:
            geometry = feature.get("geometry") or {}
            touched |= tiles_touching_geometry(geometry, leaf_zoom)
        if not touched:
            return []

        kept = clip_tiles_to_geometry(touched, leaf_zoom, self.prepared,
                                      parent_zoom=PARENT_TILE_ZOOM)
        return [
            (leaf_zoom, x, y,
             int(stable_key(self.cfg.rng_seed, self.rec.name, leaf_zoom, x, y)
                 * 1_000_000_000))
            for x, y in kept
        ]

    # ---- phase 2: leaf harvest ----------------------------------------

    def can_harvest_more(self) -> bool:
        """Is there any tile left that this session has not already tried?"""
        if self.leaf_fetches >= self.cfg.max_leaf_tiles_per_country:
            return False
        return bool(self._next_pending_batch(limit=1))

    def harvest(self, target_candidates: int) -> int:
        """Fetch leaf tiles until we hold `target_candidates`, or coverage runs out.

        Tiles are taken in rank order, and rank is a stable hash over the whole
        country, so a partial harvest still samples the entire country rather
        than one corner of it.
        """
        added = 0
        while self.db.candidates_count(self.rec.name) < target_candidates:
            if self.leaf_fetches >= self.cfg.max_leaf_tiles_per_country:
                self.log.info("[TILES] %s: reached max_leaf_tiles_per_country (%d)",
                              self.rec.name, self.cfg.max_leaf_tiles_per_country)
                break
            batch = self._next_pending_batch(limit=16)
            if not batch:
                break
            for z, x, y, tile_rank in batch:
                # remember every attempt: a transient failure leaves the tile
                # pending, and without this we would re-fetch the same 16 tiles
                # forever
                self._attempted.add((z, x, y))
                added += self._harvest_tile(z, x, y, tile_rank)
                self.leaf_fetches += 1
        return added

    def _next_pending_batch(self, limit: int) -> list:
        """Pending tiles not already attempted this session."""
        collected: list = []
        offset = 0
        page = max(limit * 4, 64)
        while len(collected) < limit:
            rows = self.db.pending_tiles(self.rec.name, limit=page, offset=offset)
            if not rows:
                break
            offset += len(rows)
            for z, x, y, rank in rows:
                if (z, x, y) not in self._attempted:
                    collected.append((z, x, y, rank))
                    if len(collected) >= limit:
                        break
        return collected

    def _harvest_tile(self, z: int, x: int, y: int, tile_rank: int) -> int:
        self._spend_budget()
        try:
            collection = self.client.fetch_coverage_tile(
                z, x, y, layer=TILE_LAYER_IMAGE)
        except TileUnavailableError as exc:
            # not recorded, so a bad hour cannot permanently kill this tile
            self.log.warning("[TILES] leaf z%d/%d/%d unavailable: %s", z, x, y, exc)
            self._note_unavailable()
            return 0
        except MapillaryError as exc:
            self.log.warning("[TILES] leaf z%d/%d/%d permanent failure: %s",
                             z, x, y, exc)
            self.db.record_tile(self.rec.name, z, x, y, TILE_ERROR, tile_rank)
            return 0

        rows = []
        for feature in (collection or {}).get("features", []):
            geometry = feature.get("geometry") or {}
            props = feature.get("properties") or {}
            # only image points carry ids; sequence lines are a different layer
            if geometry.get("type") != "Point" or "id" not in props:
                continue
            coords = geometry.get("coordinates") or []
            if len(coords) < 2:
                continue
            lng, lat = float(coords[0]), float(coords[1])
            if not contains_point(self.prepared, lng, lat):
                continue  # border tiles bleed into the neighbouring country
            rows.append({
                "image_id": str(props["id"]),
                "lat": lat,
                "lng": lng,
                "sequence": props.get("sequence_id"),
                "quality": props.get("quality_score"),
                "is_pano": props.get("is_pano"),
            })

        rows = self._select_candidates(self._drop_saturated(rows))
        for i, row in enumerate(rows):
            row["tile_rank"] = tile_rank
            row["rank_in_tile"] = i  # position within its tile, for round-robin

        inserted = self.db.add_candidates(self.rec.name, rows) if rows else []
        # count only rows that actually landed: a duplicate that was ignored
        # never consumed any of its sequence's budget
        self._note_selected(inserted)
        self.db.record_tile(self.rec.name, z, x, y,
                            TILE_FETCHED if rows else TILE_EMPTY,
                            tile_rank, len(inserted))
        return len(inserted)

    def _drop_saturated(self, rows: list) -> list:
        """Discard candidates whose sequence can no longer produce an image.

        Sequences are road traces that run across many tiles, while the
        per-sequence cap is global. Once a sequence reaches its cap every
        further candidate from it is dead on arrival -- yet it still occupies
        one of the tile's few candidate slots, and tile requests are the
        scarcest resource in the pipeline. Filtering here hands those slots to
        sequences that can still yield.

        The count combines images already collected with candidates banked this
        session, so the filter works on a country's very first pass rather than
        only after collection has begun.
        """
        cap = self.cfg.max_per_sequence
        if cap <= 0:
            return rows

        kept = []
        for row in rows:
            sequence = row.get("sequence")
            if not sequence:
                kept.append(row)      # no sequence means the cap cannot apply
                continue
            used = self._seq_budget.get(sequence)
            if used is None:
                used = self.db.sequence_count(sequence)
                self._seq_budget[sequence] = used
            if used < cap:
                kept.append(row)
        return kept

    def _note_selected(self, rows: list) -> None:
        for row in rows:
            sequence = row.get("sequence")
            if sequence:
                self._seq_budget[sequence] = self._seq_budget.get(sequence, 0) + 1

    def _select_candidates(self, rows: list) -> list:
        """Choose which of a tile's images to keep, favouring distinct sequences.

        A z14 tile is roughly two kilometres across, so most of its images tend
        to come from a handful of capture runs down the same streets. Taking
        them in id order therefore tends to take several frames of one drive,
        which the per-sequence cap then rejects downstream -- observed rejecting
        79% of candidates in Afghanistan, wasting tile budget that is the
        scarcest resource in the whole pipeline.

        Interleaving across sequences takes the first image of every sequence
        before the second of any, so a tile yields varied viewpoints and far
        more of its candidates survive filtering. Ordering stays fully
        deterministic: sequences sort by id, images sort by id within them.
        """
        cap = self.cfg.max_candidates_per_tile
        if not self.cfg.prefer_distinct_sequences:
            rows.sort(key=lambda r: r["image_id"])
            return rows[:cap]

        buckets: dict = {}
        for row in rows:
            buckets.setdefault(row.get("sequence") or "", []).append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda r: r["image_id"])

        order = sorted(buckets)
        picked: list = []
        depth = 0
        while len(picked) < cap:
            added = False
            for key in order:
                bucket = buckets[key]
                if depth < len(bucket):
                    picked.append(bucket[depth])
                    added = True
                    if len(picked) >= cap:
                        break
            if not added:
                break   # every sequence exhausted
            depth += 1
        return picked

    # ---- completion ---------------------------------------------------

    def discovery_complete(self) -> bool:
        """True only when we can honestly say this country has no more coverage.

        Every condition must hold: the base scan finished, no leaf tiles are
        pending, and nothing failed transiently this session. Anything less and
        the country stays open -- marking it finished is irreversible in
        practice, since finished countries are skipped on every later run.
        """
        return (self.base_scan_complete()
                and self.db.count_pending_tiles(self.rec.name) == 0
                and self.unavailable == 0)