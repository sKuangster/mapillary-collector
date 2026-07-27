"""Central configuration. Every tunable lives here, no magic numbers elsewhere."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

DEFAULT_DATA_DIR = Path.home() / ".mapillary_collector"


@dataclass(frozen=True)
class Config:
    """Frozen so nothing can mutate it mid-run. Override via CLI flags."""

    # secrets, read from environment (.env is loaded at startup)
    mapillary_token_env: str = "MAPILLARY_TOKEN"
    hf_token_env: str = "HF_TOKEN"

    # hugging face
    hf_repo_id: str = "skuangster/Mapillary_Dataset"
    hf_repo_private: bool = False
    hf_images_prefix: str = "images"
    dry_run_uploads: bool = False

    # paths, all under one data dir so the whole run is easy to move or back up
    data_dir: Path = DEFAULT_DATA_DIR

    # dataset shape
    shard_size: int = 1000
    thumb_field: str = "thumb_1024_url"

    # proportional per-country quota:
    #   quota = clamp(quota_min, round(quota_k * n_leaf_tiles ** quota_alpha), quota_max)
    # sublinear on purpose -- pure proportional would make coverage-rich countries
    # (US, Japan) most of the dataset and teach the model a country prior instead
    # of teaching it to read the image
    quota_k: float = 50.0
    quota_alpha: float = 0.5
    quota_min: int = 200
    quota_max: int = 20000

    # country selection
    countries_include: Optional[tuple] = None
    countries_exclude: tuple = ("Antarctica", "North Korea", "Turkmenistan",
                                "Western Sahara", "Somaliland",
                                "Fr. S. Antarctic Lands", "S. Geo. and the Is.")

    # quality filters
    include_panoramas: bool = False
    min_quality_score: Optional[float] = None
    coord_round_decimals: int = 4     # ~11 m cell -- rejects near-identical positions
    max_per_sequence: int = 5         # cap frames from any one capture drive
    use_computed_geometry: bool = True

    # coverage discovery
    rng_seed: int = 1337
    tile_base_zoom: int = 6           # coarse pass: where does coverage exist at all
    tile_leaf_zoom: int = 14          # leaf pass: per-image points with ids
    candidate_multiplier: int = 3
    max_candidates_per_tile: int = 4
    max_leaf_tiles_per_country: int = 25000
    daily_tile_budget: int = 45000    # stay under Mapillary's daily tile allowance
    max_tile_failures: int = 40       # unreadable tiles before stopping the run
    retry_exhausted: bool = False     # revisit countries previously found empty

    # a z14 tile's images mostly come from a handful of drives. taking them in id
    # order means every pick lands in one sequence and all but the first few are
    # rejected by max_per_sequence, so picks are interleaved across sequences
    prefer_distinct_sequences: bool = True

    # unattended operation
    forever: bool = True              # on tile block, wait and resume instead of exiting
    tile_retry_interval_s: float = 900.0    # how often to probe a blocked tile server
    idle_heartbeat_s: float = 1800.0        # log a heartbeat while waiting

    # networking
    workers: int = 6                  # parallel image fetches (network-bound, not CPU)
    request_timeout_s: float = 20.0
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 60.0
    graph_min_interval_s: float = 0.05
    graph_max_interval_s: float = 10.0
    tile_min_interval_s: float = 0.05
    tile_max_interval_s: float = 10.0
    throttle_decay: float = 0.95
    throttle_growth: float = 2.0
    max_consecutive_api_errors: int = 150

    # batched metadata: one /images?image_ids=a,b,c call replaces up to
    # entity_batch_size single-image lookups. treated as an optimisation the
    # client can switch off mid-run, so a batching regression degrades to the
    # old path instead of stopping collection
    use_entity_batching: bool = True
    entity_batch_size: int = 50

    # image validation
    min_image_bytes: int = 4096
    min_width: int = 256
    min_height: int = 256
    allowed_formats: tuple = ("JPEG",)

    # operations
    upload_retries: int = 5
    delete_local_after_upload: bool = True
    min_free_gb: float = 2.0
    drain_timeout_s: float = 1800.0
    status_every: int = 200
    log_level: str = "INFO"
    sqlite_cache_mb: int = 64         # passed to StateDB as a negative KiB pragma

    # derived paths

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.sqlite"

    @property
    def staging_dir(self) -> Path:
        """Loose validated images waiting to be packed into a full shard."""
        return self.data_dir / "staging"

    @property
    def shards_dir(self) -> Path:
        return self.data_dir / "shards"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "collector.log"

    def with_overrides(self, **kwargs) -> "Config":
        """Return a copy with fields replaced, ignoring None values.

        CLI flags default to None so that "not passed" never clobbers a default.
        """
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean) if clean else self


def load_dotenv(path: Path) -> None:
    """Minimal .env reader.

    Hand-rolled rather than pulling in python-dotenv: one less dependency to
    install, and the format we need is three lines of parsing.
    Existing environment variables always win.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_token(env_name: str) -> Optional[str]:
    value = os.environ.get(env_name)
    return value.strip() if value else None