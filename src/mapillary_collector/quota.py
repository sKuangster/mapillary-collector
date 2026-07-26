"""Per-country image quota, proportional to coverage but deliberately sublinear."""

from __future__ import annotations

from .config import Config


def compute_quota(n_leaf_tiles: int, cfg: Config) -> int:
    """quota = clamp(quota_min, quota_k * tiles^quota_alpha, quota_max).

    Why not linear: coverage between countries spans roughly three orders of
    magnitude. A linear quota would make a handful of coverage-rich countries
    most of the dataset, and a geolocation model trained on that learns to
    guess the majority country instead of reading the image. The square root
    compresses a 1000x coverage range into roughly a 30x image range, so real
    differences still show through without any one country dominating.
    """
    if n_leaf_tiles <= 0:
        return 0
    raw = cfg.quota_k * (n_leaf_tiles ** cfg.quota_alpha)
    return int(max(cfg.quota_min, min(cfg.quota_max, round(raw))))


def quota_table(cfg: Config, samples=(10, 50, 200, 1000, 5000, 20000, 100000)) -> list:
    """(tiles, quota) pairs -- used by the CLI to preview the curve."""
    return [(n, compute_quota(n, cfg)) for n in samples]
