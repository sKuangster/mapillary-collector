"""Dataset quality gates: metadata filters and downloaded-image validation."""

from __future__ import annotations

import io
from typing import Optional

from PIL import Image

from .config import Config
from .geo import CountryRec
from .state import StateDB
from .utils import utc_now_iso


def choose_coords(meta: dict, cfg: Config) -> Optional[tuple]:
    """(lat, lng, source), preferring the structure-from-motion corrected
    position -- a materially better label for coordinate regression than raw
    phone GPS. Falls back to raw geometry when computed is missing."""
    candidates = []
    if cfg.use_computed_geometry:
        candidates.append(("computed_geometry", "computed"))
    candidates.append(("geometry", "raw"))

    for field, source in candidates:
        geom = meta.get(field)
        if not isinstance(geom, dict):
            continue
        coords = geom.get("coordinates")
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            continue
        try:
            lng, lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue
        if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
            return lat, lng, source
    return None


def prefilter_candidate(image_id: str, lat: float, lng: float,
                        sequence: Optional[str], quality: Optional[float],
                        is_pano: Optional[int], cfg: Config,
                        db: StateDB) -> tuple:
    """(ok, reason) using only tile-cached metadata, so a reject costs no API call.

    evaluate_image stays authoritative after the entity lookup; this is purely
    a free early exit.
    """
    if db.id_exists(image_id):
        return False, "duplicate_id"
    if not cfg.include_panoramas and is_pano:
        return False, "panorama"
    if (cfg.min_quality_score is not None and quality is not None
            and quality < cfg.min_quality_score):
        return False, "low_quality"
    if db.coord_taken(round(lat, cfg.coord_round_decimals),
                      round(lng, cfg.coord_round_decimals)):
        return False, "coord_dupe"
    if sequence and db.sequence_count(sequence) >= cfg.max_per_sequence:
        return False, "sequence_cap"
    return True, "ok"


def evaluate_image(meta: dict, cfg: Config, db: StateDB,
                   rec: CountryRec) -> tuple:
    """(row, "ok") when the image belongs in the dataset, else (None, reason)."""
    image_id = str(meta.get("id", "")) or None
    if image_id is None:
        return None, "missing_id"
    if db.id_exists(image_id):
        return None, "duplicate_id"
    if not cfg.include_panoramas and meta.get("is_pano"):
        return None, "panorama"

    quality = meta.get("quality_score")
    if (cfg.min_quality_score is not None and quality is not None
            and quality < cfg.min_quality_score):
        return None, "low_quality"

    picked = choose_coords(meta, cfg)
    if picked is None:
        return None, "no_coords"
    lat, lng, coord_source = picked

    lat_r = round(lat, cfg.coord_round_decimals)
    lng_r = round(lng, cfg.coord_round_decimals)
    if db.coord_taken(lat_r, lng_r):
        return None, "coord_dupe"

    sequence = meta.get("sequence")
    if sequence and db.sequence_count(sequence) >= cfg.max_per_sequence:
        return None, "sequence_cap"

    row = {
        "id": image_id,
        "shard_idx": None,          # staged until a full shard is packed
        "country": rec.name,
        "iso3": rec.iso3,
        "continent": rec.continent,
        "lat": lat,
        "lng": lng,
        "lat_r": lat_r,
        "lng_r": lng_r,
        "coord_source": coord_source,
        "compass": meta.get("compass_angle"),
        "computed_compass": meta.get("computed_compass_angle"),
        "captured_at": meta.get("captured_at"),
        "is_pano": 1 if meta.get("is_pano") else 0,
        "quality": quality,
        "sequence": sequence,
        "camera_type": meta.get("camera_type"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "created_at": utc_now_iso(),
    }
    return row, "ok"


def validate_image_bytes(data: bytes, cfg: Config) -> tuple:
    """(ok, reason, (w, h) | None).

    Never trust downloaded bytes: error pages, truncated transfers and 1-pixel
    placeholders all arrive with a 200 status.
    """
    if data is None or len(data) < cfg.min_image_bytes:
        return False, "too_small_bytes", None

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()          # structural check
            fmt = probe.format
    except Exception as exc:        # PIL raises many types; keep the name as context
        return False, f"undecodable:{type(exc).__name__}", None

    if fmt not in cfg.allowed_formats:
        return False, f"format:{fmt}", None

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()              # full decode catches truncation verify() misses
            width, height = img.size
    except Exception as exc:
        return False, f"truncated:{type(exc).__name__}", None

    if width < cfg.min_width or height < cfg.min_height:
        return False, f"small_dims:{width}x{height}", None
    return True, "ok", (width, height)
