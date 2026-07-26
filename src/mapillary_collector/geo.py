"""Country polygons and slippy-map tile math.

Natural Earth is loaded as GeoJSON and parsed with shapely directly. Using
geopandas here would drag in GDAL/fiona, which is the single most common cause
of a failed install on a laptop -- and we only need one shapefile's geometry.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import requests
from shapely.geometry import Point, box, shape
from shapely.prepared import prep

from .config import Config
from .constants import NATURAL_EARTH_GEOJSON_URL, WEB_MERCATOR_MAX_LAT

NE_CACHE_NAME = "ne_110m_admin_0_countries.geojson"


@dataclass(frozen=True)
class CountryRec:
    name: str
    iso3: Optional[str]
    continent: Optional[str]
    geometry: Any


def _download_natural_earth(cache_path: Path, log: logging.Logger) -> dict:
    log.info("[GEO] downloading Natural Earth country boundaries (one time)")
    resp = requests.get(NATURAL_EARTH_GEOJSON_URL, timeout=120)
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    return json.loads(resp.content)


def load_countries(cfg: Config, log: logging.Logger) -> list:
    """Sorted CountryRec list, filtered by the include/exclude config."""
    cache_path = cfg.cache_dir / NE_CACHE_NAME
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("[GEO] cached boundaries unreadable, re-downloading")
            data = _download_natural_earth(cache_path, log)
    else:
        data = _download_natural_earth(cache_path, log)

    records = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        name = props.get("NAME") or props.get("ADMIN") or props.get("name")
        geometry = feature.get("geometry")
        if not name or not geometry:
            continue
        try:
            geom = shape(geometry)
        except Exception:  # a malformed polygon should skip one country, not all
            log.warning("[GEO] unusable geometry for %s, skipping", name)
            continue
        if geom.is_empty:
            continue
        records.append(CountryRec(
            name=str(name),
            iso3=props.get("ISO_A3") or props.get("ADM0_A3"),
            continent=props.get("CONTINENT"),
            geometry=geom,
        ))

    records.sort(key=lambda r: r.name)  # stable order across sessions

    if cfg.countries_include is not None:
        wanted = set(cfg.countries_include)
        missing = wanted - {r.name for r in records}
        if missing:
            log.warning("[GEO] requested countries not found: %s", sorted(missing))
        records = [r for r in records if r.name in wanted]
    if cfg.countries_exclude:
        excluded = set(cfg.countries_exclude)
        records = [r for r in records if r.name not in excluded]

    log.info("[GEO] %d countries queued", len(records))
    return records


def lnglat_to_tile(lng: float, lat: float, zoom: int) -> tuple:
    """Tile (x, y) containing a point. Latitude is clamped to the web-mercator
    limit so polar coordinates cannot blow up the tangent."""
    lat = min(max(lat, -WEB_MERCATOR_MAX_LAT), WEB_MERCATOR_MAX_LAT)
    n = 2 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_bounds(zoom: int, x: int, y: int) -> tuple:
    """(west, south, east, north) in degrees."""
    n = 2 ** zoom

    def row_to_lat(row: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / n))))

    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    return west, row_to_lat(y + 1), east, row_to_lat(y)


def tiles_over_geometry(geometry: Any, zoom: int, prepared: Any) -> list:
    """Every tile at `zoom` whose box intersects the country polygon.

    Testing against the polygon rather than its bounding box is what keeps
    awkward shapes cheap -- Chile's bbox is mostly open Pacific.
    """
    minx, miny, maxx, maxy = geometry.bounds
    x_min, _ = lnglat_to_tile(minx, 0.0, zoom)
    x_max, _ = lnglat_to_tile(maxx, 0.0, zoom)
    _, y_min = lnglat_to_tile(0.0, maxy, zoom)  # north edge maps to the smaller y
    _, y_max = lnglat_to_tile(0.0, miny, zoom)
    out = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if prepared.intersects(box(*tile_bounds(zoom, x, y))):
                out.append((x, y))
    return out


def iter_geometry_coords(geometry: dict, step_deg: float) -> Iterator[tuple]:
    """(lng, lat) points covering a GeoJSON geometry.

    Line segments are interpolated at `step_deg` because coarse-zoom coverage
    lines are simplified: a long highway may be only two vertices, and without
    interpolation every leaf tile between them would be missed.
    """
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "GeometryCollection":
        for sub in geometry.get("geometries", []):
            yield from iter_geometry_coords(sub, step_deg)
        return
    if coords is None:
        return
    if gtype == "Point":
        yield tuple(coords[:2])
    elif gtype == "MultiPoint":
        for c in coords:
            yield tuple(c[:2])
    elif gtype == "LineString":
        yield from _iter_line(coords, step_deg)
    elif gtype == "MultiLineString":
        for line in coords:
            yield from _iter_line(line, step_deg)
    elif gtype == "Polygon":
        for ring in coords:
            yield from _iter_line(ring, step_deg)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                yield from _iter_line(ring, step_deg)


def _iter_line(points: list, step_deg: float) -> Iterator[tuple]:
    for i, point in enumerate(points):
        yield tuple(point[:2])
        if i + 1 >= len(points):
            continue
        x0, y0 = point[:2]
        x1, y1 = points[i + 1][:2]
        steps = int(math.hypot(x1 - x0, y1 - y0) / step_deg)
        for s in range(1, steps + 1):
            t = s / (steps + 1)
            yield (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)


def contains_point(prepared: Any, lng: float, lat: float) -> bool:
    return prepared.contains(Point(lng, lat))
