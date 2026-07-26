"""Coverage-first discovery.

Instead of guessing coordinates and hoping an image is nearby, read Mapillary's
own coverage tiles to learn where images actually are, then spend graph API
calls only on images already known to exist.

Two cached phases per country:
  1. base scan (coarse zoom) -- which parts of the country have coverage at all
  2. leaf harvest (fine zoom) -- the actual image ids, with enough metadata to
     pre-filter before spending an API call

Every tile fetched and every candidate found is persisted, so discovery is
resumable and never repeats a fetch.
"""

from __future__ import annotations

import logging

from .client import MapillaryClient, MapillaryError
from .config import Config
from .constants import TILE_EMPTY, TILE_ERROR, TILE_FETCHED
from .geo import (
    CountryRec,
    contains_point,
    iter_geometry_coords,
    lnglat_to_tile,
    tiles_over_geometry,
)
from .state import StateDB
from .utils import stable_key


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
        self.error_fetches = 0
        self._base_done_key = f"base_done:{rec.name}"
        # half a leaf tile per interpolation step, so no leaf lying between two
        # simplified vertices of a coverage line is skipped
        self._step_deg = 360.0 / (2 ** cfg.tile_leaf_zoom) / 2.0

    # ---- phase 1: base scan -------------------------------------------

    def base_scan(self) -> int:
        """Find every leaf tile with coverage. Returns the leaf tile count.

        Runs once per country; the result is cached in the tiles table, so a
        rerun after a crash costs nothing.
        """
        if self.db.kv_get(self._base_done_key):
            return self.db.count_leaf_tiles(self.rec.name, self.cfg.tile_leaf_zoom)

        zoom = self.cfg.tile_base_zoom
        base_tiles = tiles_over_geometry(self.rec.geometry, zoom, self.prepared)
        base_tiles.sort(key=lambda t: stable_key(
            self.cfg.rng_seed, self.rec.name, zoom, t[0], t[1]))
        self.log.info("[TILES] %s: scanning %d base tiles at z%d",
                      self.rec.name, len(base_tiles), zoom)

        for rank, (x, y) in enumerate(base_tiles):
            if self.db.tile_status(self.rec.name, zoom, x, y) is not None:
                continue  # a previous session already scanned this one
            try:
                collection = self.client.fetch_coverage_tile(zoom, x, y)
            except MapillaryError as exc:
                self.log.warning("[TILES] base z%d/%d/%d failed: %s", zoom, x, y, exc)
                self.db.record_tile(self.rec.name, zoom, x, y, TILE_ERROR, rank)
                continue
            leaves = self._leaves_from_collection(collection)
            if leaves:
                self.db.add_pending_tiles(self.rec.name, leaves)
                self.db.record_tile(self.rec.name, zoom, x, y, TILE_FETCHED,
                                    rank, len(leaves))
            else:
                self.db.record_tile(self.rec.name, zoom, x, y, TILE_EMPTY, rank)

        self.db.kv_set(self._base_done_key, True)
        return self.db.count_leaf_tiles(self.rec.name, self.cfg.tile_leaf_zoom)

    def _leaves_from_collection(self, collection) -> list:
        """Coverage geometry -> the leaf tiles it touches, clipped to this country.

        Leaf tiles get a stable rank here. Ranking across the country's whole
        leaf set (not per base tile) is what lets a partial harvest still be
        spread over the entire country.
        """
        if not collection or not collection.get("features"):
            return []
        leaf_zoom = self.cfg.tile_leaf_zoom
        seen = set()
        for feature in collection["features"]:
            geometry = feature.get("geometry") or {}
            for lng, lat in iter_geometry_coords(geometry, self._step_deg):
                if not contains_point(self.prepared, lng, lat):
                    continue
                seen.add(lnglat_to_tile(lng, lat, leaf_zoom))
        return [
            (leaf_zoom, x, y,
             int(stable_key(self.cfg.rng_seed, self.rec.name, leaf_zoom, x, y)
                 * 1_000_000_000))
            for x, y in seen
        ]

    # ---- phase 2: leaf harvest ----------------------------------------

    def harvest(self, target_candidates: int) -> int:
        """Fetch leaf tiles until we hold `target_candidates`, or coverage runs out.

        Tiles are taken in rank order, and rank is a stable hash over the whole
        country -- so a partial harvest samples the entire country rather than
        one corner of it.
        """
        added = 0
        while self.db.candidates_count(self.rec.name) < target_candidates:
            if self.leaf_fetches >= self.cfg.max_leaf_tiles_per_country:
                self.log.info("[TILES] %s: hit max_leaf_tiles_per_country (%d)",
                              self.rec.name, self.cfg.max_leaf_tiles_per_country)
                break
            # a long run of undecodable tiles means this region's coverage is
            # broken or absent -- move on instead of grinding all night
            if self.error_fetches > 200:
                self.log.warning("[TILES] %s: %d decode failures, moving on "
                                 "with %d candidates", self.rec.name,
                                 self.error_fetches,
                                 self.db.candidates_count(self.rec.name))
                break
            batch = self.db.pending_tiles(self.rec.name, limit=16)
            error_tiles = self.db.count_leaf_tiles(self.rec.name, self.cfg.tile_leaf_zoom) - self.db.count_pending_tiles(self.rec.name)
            if self.leaf_fetches > 100 and self.db.candidates_count(self.rec.name) == 0:
                self.log.warning("[TILES] %s: 0 candidates after %d fetches, skipping",
                                self.rec.name, self.leaf_fetches)
                break
            if not batch:
                break
            for z, x, y, tile_rank in batch:
                added += self._harvest_tile(z, x, y, tile_rank)
                self.leaf_fetches += 1
        return added

    def _harvest_tile(self, z: int, x: int, y: int, tile_rank: int) -> int:
        try:
            collection = self.client.fetch_coverage_tile(z, x, y)
        except MapillaryError as exc:
            self.log.warning("[TILES] leaf z%d/%d/%d failed: %s", z, x, y, exc)
            self.db.record_tile(self.rec.name, z, x, y, TILE_ERROR, tile_rank)
            self.error_fetches += 1
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

        # deterministic order, then cap: stops one dense city block from
        # contributing hundreds of near-identical frames
        rows.sort(key=lambda r: r["image_id"])
        rows = rows[: self.cfg.max_candidates_per_tile]
        for i, row in enumerate(rows):
            row["tile_rank"] = tile_rank
            row["rank_in_tile"] = i  # position within its tile, for round-robin

        inserted = self.db.add_candidates(self.rec.name, rows) if rows else 0
        self.db.record_tile(self.rec.name, z, x, y,
                            TILE_FETCHED if rows else TILE_EMPTY,
                            tile_rank, inserted)
        return inserted

    def fully_harvested(self) -> bool:
        return self.db.count_pending_tiles(self.rec.name) == 0
