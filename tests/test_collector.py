"""Smoke tests. Run with: pytest -q"""

from __future__ import annotations

import io
import json
import logging
import random
import tarfile
from pathlib import Path

import pytest
from PIL import Image
from shapely.geometry import box
from shapely.prepared import prep

from mapillary_collector import geo, quota
from mapillary_collector.config import Config
from mapillary_collector.constants import (
    COUNTRY_COMPLETED,
    SHARD_LOCAL,
    SHARD_UPLOADED,
)
from mapillary_collector.filters import (
    evaluate_image,
    prefilter_candidate,
    validate_image_bytes,
)
from mapillary_collector.geo import CountryRec
from mapillary_collector.pipeline import Pipeline
from mapillary_collector.ratelimit import AdaptiveRateLimiter
from mapillary_collector.recovery import RecoveryManager
from mapillary_collector.staging import StagingArea, inspect_tar
from mapillary_collector.state import StateDB
from mapillary_collector.upload import HfStore, UploadManager
from mapillary_collector.utils import stable_key

SQUARE = box(0, 0, 10, 10)
REC = CountryRec("Testland", "TST", "Nowhere", SQUARE)


def make_jpeg(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 30, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def cfg_for(tmp_path: Path, **over) -> Config:
    base = dict(
        data_dir=tmp_path, dry_run_uploads=True, shard_size=3, workers=2,
        min_image_bytes=100, min_width=8, min_height=8, coord_round_decimals=6,
        min_free_gb=0.0, status_every=10_000,
        tile_base_zoom=4, tile_leaf_zoom=10, candidate_multiplier=4,
        quota_k=2.0, quota_alpha=0.5, quota_min=5, quota_max=5,
    )
    base.update(over)
    return Config(**base)


def base_row(image_id: str, **over) -> dict:
    row = {
        "id": image_id, "shard_idx": None, "country": "Testland", "iso3": "TST",
        "continent": "Nowhere", "lat": 1.5, "lng": 2.5, "lat_r": 1.5, "lng_r": 2.5,
        "coord_source": "raw", "compass": 1.0, "computed_compass": None,
        "captured_at": 1, "is_pano": 0, "quality": 0.9, "sequence": "seqA",
        "camera_type": "perspective", "width": 64, "height": 64,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    row.update(over)
    return row


def null_log() -> logging.Logger:
    log = logging.getLogger("test-mapillary")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


# ---- rate limiter ------------------------------------------------------

def test_limiter_grows_and_decays():
    rl = AdaptiveRateLimiter(0.1, 1.0, 0.5, 2.0)
    rl.penalize()
    assert rl.interval == pytest.approx(0.2)
    for _ in range(5):
        rl.penalize()
    assert rl.interval <= 1.0
    rl.on_success()
    assert rl.interval < 1.0


# ---- tile math ---------------------------------------------------------

def test_tile_roundtrip_contains_point():
    rng = random.Random(7)
    for _ in range(300):
        lng = rng.uniform(-179.9, 179.9)
        lat = rng.uniform(-84.0, 84.0)
        for z in (4, 10, 14):
            x, y = geo.lnglat_to_tile(lng, lat, z)
            w, s, e, n = geo.tile_bounds(z, x, y)
            assert w - 1e-9 <= lng <= e + 1e-9
            assert s - 1e-6 <= lat <= n + 1e-6


def test_polar_and_bounds_are_safe():
    x, y = geo.lnglat_to_tile(0, 89.999, 14)
    assert 0 <= y < 2 ** 14
    assert geo.lnglat_to_tile(0.5, 0.5, 1) == (1, 0)


def test_tiles_over_geometry_all_intersect():
    prepared = prep(SQUARE)
    tiles = geo.tiles_over_geometry(SQUARE, 4, prepared)
    assert tiles
    for x, y in tiles:
        assert prepared.intersects(box(*geo.tile_bounds(4, x, y)))


def test_line_interpolation_fills_gaps():
    pts = list(geo.iter_geometry_coords(
        {"type": "LineString", "coordinates": [[0, 0], [1, 0]]}, 0.25))
    assert len(pts) >= 5


def test_stable_key_is_deterministic():
    assert stable_key(1, "a", 2) == stable_key(1, "a", 2)
    assert stable_key(1, "a", 2) != stable_key(1, "a", 3)


# ---- quota -------------------------------------------------------------

def test_quota_is_sublinear_and_clamped():
    cfg = Config(quota_k=35.0, quota_alpha=0.5, quota_min=200, quota_max=5000)
    assert quota.compute_quota(0, cfg) == 0
    assert quota.compute_quota(1, cfg) == 200          # floor
    assert quota.compute_quota(10 ** 9, cfg) == 5000   # ceiling
    small = quota.compute_quota(1_000, cfg)
    big = quota.compute_quota(100_000, cfg)
    # 100x the tiles must not yield 100x the images
    assert big < small * 100
    assert big > small


# ---- state db ----------------------------------------------------------

def test_register_dedupe_and_staging(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    assert db.register_image(base_row("a"))
    assert not db.register_image(base_row("a"))
    assert db.id_exists("a")
    assert db.coord_taken(1.5, 2.5)
    assert db.sequence_count("seqA") == 1
    assert db.staged_ids() == ["a"]
    db.assign_shard(["a"], 4)
    assert db.staged_ids() == []
    assert db.image_ids_in_shard(4) == {"a"}
    db.close()


def test_round_robin_candidate_order(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    rows = []
    for tile_rank in range(3):
        for rank_in_tile in range(3):
            rows.append({
                "image_id": f"t{tile_rank}_i{rank_in_tile}",
                "tile_rank": tile_rank, "rank_in_tile": rank_in_tile,
                "lat": 1.0, "lng": 1.0,
            })
    db.add_candidates("Testland", rows)
    order = [r[2] for batch in db.iter_candidates("Testland", batch=2) for r in batch]
    # first image of every tile before the second image of any tile
    assert order[:3] == ["t0_i0", "t1_i0", "t2_i0"]
    assert order[3:6] == ["t0_i1", "t1_i1", "t2_i1"]
    assert len(order) == 9
    db.close()


def test_shard_offset_numbering(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    assert db.next_shard_idx(offset=0) == 0
    assert db.next_shard_idx(offset=7) == 7      # fresh db, existing repo shards
    db.upsert_shard(7, SHARD_LOCAL, n_samples=3)
    assert db.next_shard_idx(offset=7) == 8
    db.close()


def test_country_fields_roundtrip(tmp_path):
    db = StateDB(tmp_path / "s.sqlite")
    db.upsert_country("Testland", "in_progress", iso3="TST", quota=100,
                      leaf_tiles=42, started_at="t0")
    db.upsert_country("Testland", COUNTRY_COMPLETED, finished_at="t1")
    row = db.country_row("Testland")
    assert row["quota"] == 100 and row["leaf_tiles"] == 42
    assert row["started_at"] == "t0" and row["finished_at"] == "t1"
    assert row["status"] == COUNTRY_COMPLETED
    db.close()


# ---- validation and filters -------------------------------------------

def test_validation_rejects_bad_bytes(tmp_path):
    cfg = cfg_for(tmp_path)
    good = make_jpeg()
    ok, _, dims = validate_image_bytes(good, cfg)
    assert ok and dims == (64, 64)
    assert not validate_image_bytes(good[: len(good) // 2], cfg)[0]
    assert validate_image_bytes(b"x" * 10, cfg)[1] == "too_small_bytes"
    assert validate_image_bytes(make_jpeg(4, 4), cfg)[1].startswith("small_dims")

    rng = random.Random(0)
    noisy = Image.new("RGB", (64, 64))
    noisy.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                   for _ in range(64 * 64)])
    buf = io.BytesIO()
    noisy.save(buf, format="PNG")
    assert validate_image_bytes(buf.getvalue(), cfg)[1].startswith("format")


def _meta(image_id: str, lng: float, lat: float, **over) -> dict:
    meta = {
        "id": image_id,
        "geometry": {"type": "Point", "coordinates": [lng, lat]},
        "computed_geometry": {"type": "Point", "coordinates": [lng, lat + 0.001]},
        "compass_angle": 5.0, "computed_compass_angle": 5.5,
        "captured_at": 1, "is_pano": False, "quality_score": 0.8,
        "sequence": f"seq_{image_id}", "camera_type": "perspective",
        "width": 640, "height": 480,
        "thumb_1024_url": f"http://fake/{image_id}.jpg",
    }
    meta.update(over)
    return meta


def test_evaluate_prefers_computed_geometry(tmp_path):
    cfg = cfg_for(tmp_path)
    db = StateDB(tmp_path / "s.sqlite")
    row, reason = evaluate_image(_meta("x", 2.0, 3.0), cfg, db, REC)
    assert reason == "ok"
    assert row["coord_source"] == "computed"
    assert row["lat"] == pytest.approx(3.001)
    assert row["shard_idx"] is None
    db.close()


def test_filters_reject_each_reason(tmp_path):
    cfg = cfg_for(tmp_path)
    db = StateDB(tmp_path / "s.sqlite")
    assert evaluate_image(_meta("p", 1, 1, is_pano=True), cfg, db, REC)[1] == "panorama"
    assert evaluate_image(_meta("n", 1, 1, geometry=None, computed_geometry=None),
                          cfg, db, REC)[1] == "no_coords"
    strict = cfg_for(tmp_path, min_quality_score=0.5)
    assert evaluate_image(_meta("q", 1, 1, quality_score=0.1),
                          strict, db, REC)[1] == "low_quality"

    db.register_image(base_row("dup", lat_r=9.0, lng_r=9.0, sequence="full"))
    assert prefilter_candidate("dup", 5.0, 5.0, None, 0.9, 0, cfg, db) \
        == (False, "duplicate_id")
    assert prefilter_candidate("new", 9.0, 9.0, None, 0.9, 0, cfg, db) \
        == (False, "coord_dupe")
    capped = cfg_for(tmp_path, max_per_sequence=1)
    assert prefilter_candidate("new2", 5.0, 5.0, "full", 0.9, 0, capped, db) \
        == (False, "sequence_cap")
    assert prefilter_candidate("new3", 5.0, 5.0, "fresh", 0.9, 0, cfg, db)[0]
    db.close()


# ---- staging and packing ----------------------------------------------

def test_staging_pack_produces_paired_tar(tmp_path):
    cfg = cfg_for(tmp_path)
    staging = StagingArea(cfg, null_log())
    for i in range(3):
        staging.add(f"img{i}", make_jpeg(), {"id": f"img{i}", "lat": 1.0})
    assert staging.count() == 3
    path = staging.pack_shard(0, staging.complete_ids())
    assert path.exists()
    assert staging.count() == 0            # loose files cleaned up after packing
    with tarfile.open(path) as tar:
        names = tar.getnames()
    assert len(names) == 6
    # image and its json adjacent, so WebDataset pairs them
    assert names[0].rsplit(".", 1)[0] == names[1].rsplit(".", 1)[0]
    assert inspect_tar(path) == {"img0", "img1", "img2"}


def test_orphan_detection(tmp_path):
    cfg = cfg_for(tmp_path)
    staging = StagingArea(cfg, null_log())
    staging.add("good", make_jpeg(), {"id": "good"})
    (staging.dir / "half.jpg").write_bytes(make_jpeg())   # crash before the json
    assert staging.complete_ids() == ["good"]
    assert staging.orphan_ids() == ["half"]


# ---- recovery ----------------------------------------------------------

def _env(tmp_path, **over):
    cfg = cfg_for(tmp_path, **over)
    log = null_log()
    db = StateDB(cfg.db_path)
    staging = StagingArea(cfg, log)
    store = HfStore(cfg, None, log)
    uploader = UploadManager(cfg, db, store, log)
    uploader.start()
    return cfg, db, staging, store, uploader, log


def test_recovery_drops_rows_without_files(tmp_path):
    cfg, db, staging, store, uploader, log = _env(tmp_path)
    db.register_image(base_row("ghost"))
    RecoveryManager(cfg, db, staging, store, uploader, log).reconcile()
    assert not db.id_exists("ghost")
    uploader.stop()
    db.close()


def test_recovery_removes_files_without_rows(tmp_path):
    cfg, db, staging, store, uploader, log = _env(tmp_path)
    staging.add("stray", make_jpeg(), {"id": "stray"})
    RecoveryManager(cfg, db, staging, store, uploader, log).reconcile()
    assert staging.complete_ids() == []
    uploader.stop()
    db.close()


def test_recovery_keeps_matched_staging(tmp_path):
    cfg, db, staging, store, uploader, log = _env(tmp_path)
    db.register_image(base_row("keep"))
    staging.add("keep", make_jpeg(), {"id": "keep"})
    RecoveryManager(cfg, db, staging, store, uploader, log).reconcile()
    assert staging.complete_ids() == ["keep"]
    assert db.id_exists("keep")
    uploader.stop()
    db.close()


def test_recovery_requeues_local_shard(tmp_path):
    cfg, db, staging, store, uploader, log = _env(tmp_path)
    for i in range(3):
        staging.add(f"s{i}", make_jpeg(), {"id": f"s{i}"})
        db.register_image(base_row(f"s{i}", lat_r=float(i), lng_r=float(i),
                                   sequence=f"q{i}"))
    ids = staging.complete_ids()
    path = staging.pack_shard(0, ids)
    db.assign_shard(ids, 0)
    db.upsert_shard(0, SHARD_LOCAL, n_samples=3, filename=path.name)
    RecoveryManager(cfg, db, staging, store, uploader, log).reconcile()
    uploader.drain(10)
    uploader.stop()
    assert db.shards_with_status(SHARD_UPLOADED)[0][0] == 0
    db.close()


def test_recovery_releases_lost_shard(tmp_path):
    cfg, db, staging, store, uploader, log = _env(tmp_path)
    db.register_image(base_row("lost1", shard_idx=0))
    db.assign_shard(["lost1"], 0)
    db.upsert_shard(0, SHARD_LOCAL, n_samples=1, filename="shard-000000.tar")
    RecoveryManager(cfg, db, staging, store, uploader, log).reconcile()
    assert not db.id_exists("lost1")   # freed for re-collection
    uploader.stop()
    db.close()


# ---- end to end --------------------------------------------------------

class FakeClient:
    """Serves synthetic coverage tiles and entity records, no network."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.graph_limiter = AdaptiveRateLimiter(0, 0, 1, 1)
        self.entity_calls = 0
        self.tile_calls = 0

    def fetch_coverage_tile(self, z, x, y, layer=None):
        self.tile_calls += 1
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
            feats.append({
                "geometry": {"type": "Point", "coordinates": [lng, lat]},
                "properties": {"id": iid, "is_pano": False, "quality_score": 0.8,
                               "sequence_id": f"seq_{iid}"},
            })
        return {"features": feats}

    def get_image(self, image_id):
        self.entity_calls += 1
        x, y, i = image_id.split("_")
        w, s, e, n = geo.tile_bounds(self.cfg.tile_leaf_zoom, int(x), int(y))
        lng = w + (e - w) * (0.2 + 0.15 * int(i))
        lat = s + (n - s) * 0.5
        return _meta(image_id, lng, lat, sequence=f"seq_{image_id}")

    def fetch_image_bytes(self, url, ctx):
        return make_jpeg()

    def get_images_batch(self, image_ids):
        out = {}
        for image_id in image_ids:
            meta = self.get_image(image_id)
            if meta is not None:
                out[str(image_id)] = meta
        return out



def _run_pipeline(cfg, client_factory=FakeClient):
    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.client = client_factory(cfg)
    pipe._collect_country(REC)
    pipe._finalize("test")
    return pipe


def test_end_to_end_collects_quota_and_packs(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = cfg_for(tmp_path)          # quota clamped to 5, shard_size 3
    _run_pipeline(cfg)

    db = StateDB(cfg.db_path)
    assert db.images_in_country("Testland") == 5
    assert db.country_status("Testland") == COUNTRY_COMPLETED
    shards = db.shards_with_status(SHARD_UPLOADED)
    assert len(shards) == 1                # exactly one full shard
    assert shards[0][2] == 3               # of exactly shard_size
    assert db.totals()["staged"] == 2      # remainder stays staged, not shipped short
    db.close()


def test_shards_are_uniform_never_short(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = cfg_for(tmp_path, quota_min=7, quota_max=7, shard_size=3)
    _run_pipeline(cfg)
    db = StateDB(cfg.db_path)
    sizes = [n for _, _, n in db.shards_with_status(SHARD_UPLOADED)]
    assert sizes == [3, 3]                 # 7 images -> two full shards
    assert db.totals()["staged"] == 1
    db.close()


def test_resume_adds_nothing_and_spends_no_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = cfg_for(tmp_path)
    _run_pipeline(cfg)

    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    client = FakeClient(cfg)
    pipe.client = client
    pipe._run_countries([REC])
    pipe._finalize("test2")

    assert client.entity_calls == 0
    assert client.tile_calls == 0
    db = StateDB(cfg.db_path)
    assert db.images_in_country("Testland") == 5
    db.close()


def test_staged_images_survive_restart_and_complete_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = cfg_for(tmp_path)
    _run_pipeline(cfg)                      # leaves 2 staged

    db = StateDB(cfg.db_path)
    staged_before = db.totals()["staged"]
    db.close()
    assert staged_before == 2

    # a second country tops the shard up rather than shipping a short one
    rec2 = CountryRec("Otherland", "OTH", "Nowhere", box(0, 0, 10, 10))
    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.client = FakeClient(cfg)
    pipe._collect_country(rec2)
    pipe._finalize("test3")

    db = StateDB(cfg.db_path)
    sizes = [n for _, _, n in db.shards_with_status(SHARD_UPLOADED)]
    assert all(size == 3 for size in sizes)
    assert db.totals()["images"] == 10
    db.close()


def test_finalize_packs_short_shard_on_demand(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = cfg_for(tmp_path)
    _run_pipeline(cfg)

    pipe = Pipeline(cfg, null_log())
    pipe.startup()
    pipe.finalize_partial_shard()
    pipe._finalize("finalize")

    db = StateDB(cfg.db_path)
    sizes = sorted(n for _, _, n in db.shards_with_status(SHARD_UPLOADED))
    assert sizes == [2, 3]                 # short shard only when asked for
    assert db.totals()["staged"] == 0
    db.close()


def test_sampling_spreads_across_tiles(tmp_path, monkeypatch):
    monkeypatch.setenv("MAPILLARY_TOKEN", "MLY|test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = cfg_for(tmp_path, quota_min=12, quota_max=12)
    _run_pipeline(cfg)

    db = StateDB(cfg.db_path)
    staged = db.staged_ids()
    shard_ids = set()
    for idx, _, _ in db.shards_with_status(SHARD_UPLOADED):
        shard_ids |= db.image_ids_in_shard(idx)
    all_ids = set(staged) | shard_ids
    # ids are "{tilex}_{tiley}_{i}" -- distinct tiles, and no tile used twice
    # before every other tile has been used once
    tiles = [i.rsplit("_", 1)[0] for i in all_ids]
    assert len(set(tiles)) == len(tiles)
    db.close()