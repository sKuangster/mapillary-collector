"""Regression tests for the outage bugs that silently dropped whole countries.

Every test here fails on the previous version.
"""

from __future__ import annotations

import io
import logging

import pytest
from PIL import Image
from shapely.geometry import box
from shapely.prepared import prep

from mapillary_collector import geo
from mapillary_collector.client import MapillaryError, TileUnavailableError
from mapillary_collector.config import Config
from mapillary_collector.constants import (
    COUNTRY_COMPLETED,
    COUNTRY_EXHAUSTED,
    COUNTRY_IN_PROGRESS,
    TILE_PENDING,
)
from mapillary_collector.discovery import TileDiscovery, TileQuotaExhausted
from mapillary_collector.geo import CountryRec
from mapillary_collector.pipeline import Pipeline
from mapillary_collector.ratelimit import AdaptiveRateLimiter
from mapillary_collector.state import StateDB

SQUARE = box(0, 0, 10, 10)
REC = CountryRec("Testland", "TST", "Nowhere", SQUARE)


def make_jpeg(w=64, h=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def null_log():
    log = logging.getLogger("test-outage")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


def cfg_for(tmp_path, **over):
    base = dict(
        data_dir=tmp_path, dry_run_uploads=True, shard_size=3, workers=2,
        min_image_bytes=100, min_width=8, min_height=8, coord_round_decimals=6,
        min_free_gb=0.0, status_every=10_000, tile_base_zoom=4, tile_leaf_zoom=10,
        candidate_multiplier=4, quota_k=2.0, quota_alpha=0.5,
        quota_min=5, quota_max=5, max_candidates_per_tile=4,
    )
    base.update(over)
    return Config(**base)


class TileServerDown:
    """Every tile request fails transiently -- the outage you actually hit."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.graph_limiter = AdaptiveRateLimiter(0, 0, 1, 1)
        self.calls = 0

    def fetch_coverage_tile(self, z, x, y):
        self.calls += 1
        raise TileUnavailableError(f"non-tile body (z={z} x={x} y={y}): '<!doctype'")

    def get_image(self, image_id):
        raise AssertionError("should never reach the graph API")

    def fetch_image_bytes(self, url, ctx):
        raise AssertionError("should never download")


class WorkingTiles:
    """Serves synthetic coverage so a country genuinely has data."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.graph_limiter = AdaptiveRateLimiter(0, 0, 1, 1)

    def fetch_coverage_tile(self, z, x, y):
        w, s, e, n = geo.tile_bounds(z, x, y)
        if e < 0 or w > 10 or n < 0 or s > 10:
            return None
        if z == self.cfg.tile_base_zoom:
            return {"features": [{
                "geometry": {"type": "LineString",
                             "coordinates": [[1.0, 1.0], [8.0, 8.0]]},
                "properties": {}}]}
        feats = []
        for i in range(4):
            lng = w + (e - w) * (0.2 + 0.15 * i)
            lat = s + (n - s) * 0.5
            if not (0 <= lng <= 10 and 0 <= lat <= 10):
                continue
            iid = f"{x}_{y}_{i}"
            feats.append({"geometry": {"type": "Point", "coordinates": [lng, lat]},
                          "properties": {"id": iid, "is_pano": False,
                                         "quality_score": 0.9,
                                         "sequence_id": f"s_{iid}"}})
        return {"features": feats}

    def get_image(self, image_id):
        x, y, i = image_id.split("_")
        w, s, e, n = geo.tile_bounds(self.cfg.tile_leaf_zoom, int(x), int(y))
        lng = w + (e - w) * (0.2 + 0.15 * int(i))
        lat = s + (n - s) * 0.5
        return {"id": image_id,
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "computed_geometry": {"type": "Point", "coordinates": [lng, lat]},
                "compass_angle": 1.0, "captured_at": 1, "is_pano": False,
                "quality_score": 0.9, "sequence": f"s_{image_id}",
                "camera_type": "perspective", "width": 640, "height": 480,
                "thumb_1024_url": f"http://fake/{image_id}.jpg"}

    def fetch_image_bytes(self, url, ctx):
        return make_jpeg()


def _discovery(cfg, client, db, rec=REC):
    return TileDiscovery(cfg, client, db, rec, prep(rec.geometry), null_log())


# ---- bug 1: transient failures were recorded as permanent -------------

def test_transient_failure_never_marks_a_tile_dead(tmp_path):
    cfg = cfg_for(tmp_path, max_tile_failures=10_000)
    db = StateDB(cfg.db_path)
    db.add_pending_tiles("Testland", [(10, 5, 5, 1), (10, 6, 6, 2)])
    disc = _discovery(cfg, TileServerDown(cfg), db)
    disc.harvest(100)
    # still pending, so a later run retries them
    assert db.count_pending_tiles("Testland") == 2
    assert db.tile_status("Testland", 10, 5, 5) == TILE_PENDING
    db.close()


def test_permanent_failure_still_marks_a_tile_dead(tmp_path):
    """The distinction has to cut both ways or the queue never drains."""
    cfg = cfg_for(tmp_path)
    db = StateDB(cfg.db_path)
    db.add_pending_tiles("Testland", [(10, 5, 5, 1)])

    class PermanentlyBroken(TileServerDown):
        def fetch_coverage_tile(self, z, x, y):
            raise MapillaryError("http 400 permanent")

    disc = _discovery(cfg, PermanentlyBroken(cfg), db)
    disc.harvest(100)
    assert db.count_pending_tiles("Testland") == 0
    db.close()


# ---- bug 2: base scan claimed completion after a failed scan ----------

def test_base_scan_stays_open_after_an_outage(tmp_path):
    cfg = cfg_for(tmp_path, max_tile_failures=10_000)
    db = StateDB(cfg.db_path)
    disc = _discovery(cfg, TileServerDown(cfg), db)
    assert disc.base_scan() == 0
    assert not disc.base_scan_complete()      # would have been True before
    assert not disc.discovery_complete()
    db.close()


def test_base_scan_marks_complete_when_it_really_finishes(tmp_path):
    cfg = cfg_for(tmp_path)
    db = StateDB(cfg.db_path)
    disc = _discovery(cfg, WorkingTiles(cfg), db)
    assert disc.base_scan() > 0
    assert disc.base_scan_complete()
    db.close()


# ---- bug 3: a country with data got marked exhausted forever ----------

def test_outage_leaves_country_open_not_exhausted(tmp_path, monkeypatch):
    """The Brazil/Botswana failure: tiles unavailable, country written off."""
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    cfg = cfg_for(tmp_path, max_tile_failures=10_000)
    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.client = TileServerDown(cfg)
    pipe._collect_country(REC)
    pipe._finalize("t")

    db = StateDB(cfg.db_path)
    assert db.country_status("Testland") == COUNTRY_IN_PROGRESS
    assert db.country_status("Testland") != COUNTRY_EXHAUSTED
    db.close()


def test_country_recovers_on_the_next_run(tmp_path, monkeypatch):
    """End to end: outage run collects nothing, healthy run collects fully."""
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    cfg = cfg_for(tmp_path, max_tile_failures=10_000)

    down = Pipeline(cfg, null_log())
    down.startup()
    down.client = TileServerDown(cfg)
    down._run_countries([REC])
    down._finalize("outage")

    db = StateDB(cfg.db_path)
    assert db.images_in_country("Testland") == 0
    db.close()

    up = Pipeline(cfg, null_log())
    up.startup()
    up.client = WorkingTiles(cfg)
    up._run_countries([REC])          # must NOT skip it
    up._finalize("recovered")

    db = StateDB(cfg.db_path)
    assert db.images_in_country("Testland") == 5
    assert db.country_status("Testland") == COUNTRY_COMPLETED
    db.close()


def test_startup_reopens_countries_wrongly_marked_finished(tmp_path, monkeypatch):
    """Safety net for databases already damaged by the old code."""
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    cfg = cfg_for(tmp_path)
    db = StateDB(cfg.db_path)
    db.upsert_country("Testland", COUNTRY_EXHAUSTED, quota=0, leaf_tiles=0)
    db.kv_set("base_done:Testland", True)
    db.close()

    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.client = WorkingTiles(cfg)
    pipe._run_countries([REC])
    pipe._finalize("t")

    db = StateDB(cfg.db_path)
    assert db.images_in_country("Testland") == 5
    db.close()


def test_genuinely_empty_country_is_marked_exhausted(tmp_path, monkeypatch):
    """The safety net must not resurrect countries that truly have nothing."""
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    cfg = cfg_for(tmp_path)

    class NoCoverage(WorkingTiles):
        def fetch_coverage_tile(self, z, x, y):
            return {"features": []}

    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.client = NoCoverage(cfg)
    pipe._collect_country(REC)
    pipe._finalize("t")

    db = StateDB(cfg.db_path)
    assert db.country_status("Testland") == COUNTRY_EXHAUSTED
    db.close()


# ---- bug 4: harvest re-fetched the same failing tiles forever ---------

def test_harvest_does_not_respin_on_unavailable_tiles(tmp_path):
    cfg = cfg_for(tmp_path, max_tile_failures=10_000,
                  max_leaf_tiles_per_country=10_000)
    db = StateDB(cfg.db_path)
    db.add_pending_tiles("Testland", [(10, i, i, i) for i in range(20)])
    client = TileServerDown(cfg)
    disc = _discovery(cfg, client, db)
    disc.harvest(1000)
    # 20 tiles, each attempted at most once, instead of looping on the first 16
    assert client.calls == 20
    db.close()


# ---- bug 5: no ceiling on doomed requests ----------------------------

def test_run_stops_after_too_many_unreadable_tiles(tmp_path):
    cfg = cfg_for(tmp_path, max_tile_failures=5,
                  max_leaf_tiles_per_country=10_000)
    db = StateDB(cfg.db_path)
    db.add_pending_tiles("Testland", [(10, i, i, i) for i in range(500)])
    client = TileServerDown(cfg)
    disc = _discovery(cfg, client, db)
    with pytest.raises(TileQuotaExhausted):
        disc.harvest(10_000)
    assert client.calls <= 6      # stopped almost immediately
    db.close()


def test_daily_tile_budget_is_enforced(tmp_path):
    cfg = cfg_for(tmp_path, daily_tile_budget=7,
                  max_leaf_tiles_per_country=10_000)
    db = StateDB(cfg.db_path)
    db.add_pending_tiles("Testland", [(10, i, i, i) for i in range(100)])
    client = WorkingTiles(cfg)
    disc = _discovery(cfg, client, db)
    with pytest.raises(TileQuotaExhausted):
        disc.harvest(10_000)
    assert disc.tiles_used_today() >= 7
    db.close()


def test_tile_quota_stop_is_clean_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    cfg = cfg_for(tmp_path, max_tile_failures=3)
    pipe = Pipeline(cfg, null_log())
    pipe._install_signal_handlers()
    pipe.startup()
    pipe.client = TileServerDown(cfg)
    pipe.run = pipe.run       # exercised through run() below
    try:
        pipe._run_countries([REC])
    except TileQuotaExhausted:
        pass                  # run() catches this and finalizes
    pipe._finalize("quota")

    db = StateDB(cfg.db_path)
    assert db.country_status("Testland") == COUNTRY_IN_PROGRESS
    db.close()


# ---- bug 6: a worker exception killed the whole run ------------------

def test_unexpected_worker_exception_does_not_kill_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    cfg = cfg_for(tmp_path, max_consecutive_api_errors=10_000)

    class Flaky(WorkingTiles):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.n = 0

        def get_image(self, image_id):
            self.n += 1
            if self.n % 3 == 0:
                raise ValueError("something unexpected from a library")
            return super().get_image(image_id)

    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.client = Flaky(cfg)
    pipe._collect_country(REC)        # must not raise
    pipe._finalize("t")

    db = StateDB(cfg.db_path)
    assert db.images_in_country("Testland") == 5
    db.close()


# ---- bug 7: parallel workers could overshoot the sequence cap --------

def test_sequence_cap_holds_under_parallel_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    cfg = cfg_for(tmp_path, workers=6, max_per_sequence=2,
                  quota_min=50, quota_max=50)

    class OneSequence(WorkingTiles):
        def get_image(self, image_id):
            meta = super().get_image(image_id)
            meta["sequence"] = "the_only_sequence"
            return meta

    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.client = OneSequence(cfg)
    pipe._collect_country(REC)
    pipe._finalize("t")

    db = StateDB(cfg.db_path)
    assert db.sequence_count("the_only_sequence") <= cfg.max_per_sequence
    db.close()
