"""All persistent state in one SQLite database.

SQLite because transactions are atomic: state can never be half-written the way
a JSON checkpoint file can. It is also the duplicate index -- an image id is
"seen" if and only if a row exists, so dedupe and the dataset cannot drift apart.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from .constants import (
    COUNTRY_IN_PROGRESS,
    SHARD_LOCAL,
    TILE_PENDING,
)
from .utils import utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
    id TEXT PRIMARY KEY,
    shard_idx INTEGER,              -- NULL while staged, set when packed
    country TEXT NOT NULL,
    iso3 TEXT,
    continent TEXT,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    lat_r REAL NOT NULL,
    lng_r REAL NOT NULL,
    coord_source TEXT,
    compass REAL,
    computed_compass REAL,
    captured_at INTEGER,
    is_pano INTEGER,
    quality REAL,
    sequence TEXT,
    camera_type TEXT,
    width INTEGER,
    height INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_coord ON images(lat_r, lng_r);
CREATE INDEX IF NOT EXISTS idx_images_seq ON images(sequence);
CREATE INDEX IF NOT EXISTS idx_images_shard ON images(shard_idx);
CREATE INDEX IF NOT EXISTS idx_images_country ON images(country);

CREATE TABLE IF NOT EXISTS shards(
    shard_idx INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    n_samples INTEGER,
    filename TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS countries(
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    iso3 TEXT,
    continent TEXT,
    quota INTEGER,
    leaf_tiles INTEGER,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tiles(
    country TEXT NOT NULL,
    z INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    status TEXT NOT NULL,
    tile_rank INTEGER NOT NULL,
    n_candidates INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (country, z, x, y)
);
CREATE INDEX IF NOT EXISTS idx_tiles_pending ON tiles(country, status, tile_rank);

CREATE TABLE IF NOT EXISTS candidates(
    country TEXT NOT NULL,
    image_id TEXT NOT NULL,
    tile_rank INTEGER NOT NULL,
    rank_in_tile INTEGER NOT NULL,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    sequence TEXT,
    quality REAL,
    is_pano INTEGER,
    PRIMARY KEY (country, image_id)
);
-- rank_in_tile FIRST: this is what makes sampling uniform. Ordering by it means
-- we take image #1 from every covered tile in the country before image #2 from
-- any of them, so any quota is spread across the whole coverage footprint.
CREATE INDEX IF NOT EXISTS idx_candidates_rr
    ON candidates(country, rank_in_tile, tile_rank);

CREATE TABLE IF NOT EXISTS kv(
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class StateDB:
    """Thread-safe state access. One lock, so writers never interleave."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # check_same_thread=False so the upload thread can update shard status;
        # every access goes through self._lock, so there is still one writer.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        with self._lock, self.conn:
            self.conn.executescript(SCHEMA)

    # ---- images -------------------------------------------------------

    def id_exists(self, image_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM images WHERE id = ?", (image_id,)
            ).fetchone()
        return row is not None

    def coord_taken(self, lat_r: float, lng_r: float) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM images WHERE lat_r = ? AND lng_r = ? LIMIT 1",
                (lat_r, lng_r),
            ).fetchone()
        return row is not None

    def sequence_count(self, sequence: Optional[str]) -> int:
        if not sequence:
            return 0
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM images WHERE sequence = ?", (sequence,)
            ).fetchone()
        return int(row[0])

    def register_image(self, row: dict) -> bool:
        """Insert a staged image (shard_idx NULL). False when the id already exists."""
        cols = (
            "id", "shard_idx", "country", "iso3", "continent", "lat", "lng",
            "lat_r", "lng_r", "coord_source", "compass", "computed_compass",
            "captured_at", "is_pano", "quality", "sequence", "camera_type",
            "width", "height", "created_at",
        )
        values = tuple(row.get(c) for c in cols)
        placeholders = ",".join("?" for _ in cols)
        with self._lock, self.conn:
            cur = self.conn.execute(
                f"INSERT OR IGNORE INTO images({','.join(cols)}) VALUES({placeholders})",
                values,
            )
        return cur.rowcount == 1

    def remove_image(self, image_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM images WHERE id = ?", (image_id,))

    def remove_images(self, image_ids: Iterable[str]) -> int:
        ids = list(image_ids)
        if not ids:
            return 0
        with self._lock, self.conn:
            cur = self.conn.executemany(
                "DELETE FROM images WHERE id = ?", [(i,) for i in ids]
            )
        return cur.rowcount

    def staged_ids(self) -> list:
        """Images collected but not yet packed into a shard."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM images WHERE shard_idx IS NULL ORDER BY id"
            ).fetchall()
        return [r[0] for r in rows]

    def assign_shard(self, image_ids: Iterable[str], shard_idx: int) -> int:
        with self._lock, self.conn:
            cur = self.conn.executemany(
                "UPDATE images SET shard_idx = ? WHERE id = ?",
                [(shard_idx, i) for i in image_ids],
            )
        return cur.rowcount

    def image_ids_in_shard(self, shard_idx: int) -> set:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM images WHERE shard_idx = ?", (shard_idx,)
            ).fetchall()
        return {r[0] for r in rows}

    def images_in_country(self, country: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM images WHERE country = ?", (country,)
            ).fetchone()
        return int(row[0])

    def total_images(self) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()
        return int(row[0])

    # ---- shards -------------------------------------------------------

    def upsert_shard(self, shard_idx: int, status: str,
                     n_samples: Optional[int] = None,
                     filename: Optional[str] = None) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO shards(shard_idx, status, n_samples, filename, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(shard_idx) DO UPDATE SET
                    status = excluded.status,
                    n_samples = COALESCE(excluded.n_samples, shards.n_samples),
                    filename = COALESCE(excluded.filename, shards.filename),
                    updated_at = excluded.updated_at
                """,
                (shard_idx, status, n_samples, filename, utc_now_iso()),
            )

    def shards_with_status(self, status: str) -> list:
        with self._lock:
            return self.conn.execute(
                "SELECT shard_idx, filename, n_samples FROM shards "
                "WHERE status = ? ORDER BY shard_idx",
                (status,),
            ).fetchall()

    def next_shard_idx(self, offset: int = 0) -> int:
        with self._lock:
            row = self.conn.execute("SELECT MAX(shard_idx) FROM shards").fetchone()
        current = offset - 1 if row[0] is None else int(row[0])
        return max(current, offset - 1) + 1

    # ---- countries ----------------------------------------------------

    def upsert_country(self, name: str, status: str, **fields) -> None:
        allowed = ("iso3", "continent", "quota", "leaf_tiles",
                   "started_at", "finished_at")
        sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO countries(name, status, updated_at) "
                "VALUES(?, ?, ?)",
                (name, status, utc_now_iso()),
            )
            assignments = ", ".join(f"{k} = ?" for k in sets)
            sql = "UPDATE countries SET status = ?, updated_at = ?"
            params: list = [status, utc_now_iso()]
            if assignments:
                sql += ", " + assignments
                params.extend(sets.values())
            sql += " WHERE name = ?"
            params.append(name)
            self.conn.execute(sql, params)

    def country_row(self, name: str) -> Optional[sqlite3.Row]:
        with self._lock:
            self.conn.row_factory = sqlite3.Row
            try:
                row = self.conn.execute(
                    "SELECT * FROM countries WHERE name = ?", (name,)
                ).fetchone()
            finally:
                self.conn.row_factory = None
        return row

    def country_status(self, name: str) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM countries WHERE name = ?", (name,)
            ).fetchone()
        return None if row is None else row[0]

    def countries_by_status(self, status: str) -> list:
        with self._lock:
            return self.conn.execute(
                "SELECT name, quota, leaf_tiles FROM countries "
                "WHERE status = ? ORDER BY name",
                (status,),
            ).fetchall()

    # ---- tiles --------------------------------------------------------

    def tile_status(self, country: str, z: int, x: int, y: int) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM tiles WHERE country=? AND z=? AND x=? AND y=?",
                (country, z, x, y),
            ).fetchone()
        return None if row is None else row[0]

    def record_tile(self, country: str, z: int, x: int, y: int, status: str,
                    tile_rank: int, n_candidates: int = 0) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO tiles(country, z, x, y, status, tile_rank,
                                  n_candidates, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(country, z, x, y) DO UPDATE SET
                    status = excluded.status,
                    n_candidates = excluded.n_candidates,
                    updated_at = excluded.updated_at
                """,
                (country, z, x, y, status, tile_rank, n_candidates, utc_now_iso()),
            )

    def add_pending_tiles(self, country: str, tiles: Iterable[tuple]) -> int:
        """tiles: (z, x, y, tile_rank). Already-known tiles are left untouched."""
        rows = list(tiles)
        if not rows:
            return 0
        now = utc_now_iso()
        with self._lock, self.conn:
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO tiles(country, z, x, y, status, tile_rank, "
                "updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                [(country, z, x, y, TILE_PENDING, rank, now)
                 for z, x, y, rank in rows],
            )
            return self.conn.total_changes - before

    def pending_tiles(self, country: str, limit: int) -> list:
        with self._lock:
            return self.conn.execute(
                "SELECT z, x, y, tile_rank FROM tiles "
                "WHERE country=? AND status=? ORDER BY tile_rank LIMIT ?",
                (country, TILE_PENDING, limit),
            ).fetchall()

    def count_pending_tiles(self, country: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM tiles WHERE country=? AND status=?",
                (country, TILE_PENDING),
            ).fetchone()
        return int(row[0])

    def count_known_tiles(self, country: str) -> int:
        """Every leaf tile we know has coverage, fetched or not -- the quota input."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM tiles WHERE country=? AND z=? ",
                (country, self.kv_get("leaf_zoom", 14)),
            ).fetchone()
        return int(row[0])

    def count_leaf_tiles(self, country: str, leaf_zoom: int) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM tiles WHERE country=? AND z=?",
                (country, leaf_zoom),
            ).fetchone()
        return int(row[0])

    # ---- candidates ---------------------------------------------------

    def add_candidates(self, country: str, rows: list) -> int:
        """rows: dicts with image_id/tile_rank/rank_in_tile/lat/lng/... .

        Duplicate image ids (border tiles overlap) are ignored rather than
        raising, so re-fetching a tile is always safe.
        """
        if not rows:
            return 0
        with self._lock, self.conn:
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO candidates(country, image_id, tile_rank, "
                "rank_in_tile, lat, lng, sequence, quality, is_pano) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(country, r["image_id"], r["tile_rank"], r["rank_in_tile"],
                  r["lat"], r["lng"], r.get("sequence"), r.get("quality"),
                  1 if r.get("is_pano") else 0) for r in rows],
            )
            return self.conn.total_changes - before

    def iter_candidates(self, country: str, batch: int = 500):
        """Round-robin ordered candidates: image #1 of every tile, then #2, ...

        Yields batches. Uses keyset pagination on (rank_in_tile, tile_rank,
        image_id) so it stays fast no matter how deep into the list we get.
        """
        last = (-1, -1, "")
        while True:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT rank_in_tile, tile_rank, image_id, lat, lng, "
                    "sequence, quality, is_pano FROM candidates "
                    "WHERE country = ? AND (rank_in_tile, tile_rank, image_id) > (?, ?, ?) "
                    "ORDER BY rank_in_tile, tile_rank, image_id LIMIT ?",
                    (country, last[0], last[1], last[2], batch),
                ).fetchall()
            if not rows:
                return
            yield rows
            last = (rows[-1][0], rows[-1][1], rows[-1][2])

    def candidates_count(self, country: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE country=?", (country,)
            ).fetchone()
        return int(row[0])

    # ---- kv -----------------------------------------------------------

    def kv_get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM kv WHERE key = ?", (key,)
            ).fetchone()
        if row is None or row[0] is None:
            return default
        return json.loads(row[0])

    def kv_set(self, key: str, value: Any) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    # ---- health -------------------------------------------------------

    def totals(self) -> dict:
        with self._lock:
            images = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            staged = self.conn.execute(
                "SELECT COUNT(*) FROM images WHERE shard_idx IS NULL"
            ).fetchone()[0]
            shard_rows = self.conn.execute(
                "SELECT status, COUNT(*) FROM shards GROUP BY status"
            ).fetchall()
            country_rows = self.conn.execute(
                "SELECT status, COUNT(*) FROM countries GROUP BY status"
            ).fetchall()
            candidates = self.conn.execute(
                "SELECT COUNT(*) FROM candidates"
            ).fetchone()[0]
        return {
            "images": int(images),
            "staged": int(staged),
            "shards": {s: int(c) for s, c in shard_rows},
            "countries": {s: int(c) for s, c in country_rows},
            "candidates": int(candidates),
        }

    def integrity_ok(self) -> bool:
        with self._lock:
            row = self.conn.execute("PRAGMA integrity_check").fetchone()
        return row is not None and row[0] == "ok"

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()
