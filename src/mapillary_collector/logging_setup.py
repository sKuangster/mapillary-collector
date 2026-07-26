"""Logging. Tags like [COUNTRY] [SHARD] [UPLOAD] make an overnight log greppable."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import Config

LOGGER_NAME = "mapillary"


def setup_logging(cfg: Config) -> logging.Logger:
    """Console + rotating-ish file handler. Idempotent across repeated calls."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    try:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg.log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("file logging unavailable (%s); console only", exc)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_config_summary(cfg: Config, log: logging.Logger) -> None:
    """Printed at the top of every run so post-mortems know the settings used."""
    log.info(
        "[CONFIG] repo=%s data_dir=%s shard_size=%d workers=%d",
        cfg.hf_repo_id, cfg.data_dir, cfg.shard_size, cfg.workers,
    )
    log.info(
        "[CONFIG] quota=clamp(%d, %.0f*tiles^%.2f, %d) tiles=z%d->z%d "
        "panos=%s quality_min=%s seq_cap=%d dry_run=%s",
        cfg.quota_min, cfg.quota_k, cfg.quota_alpha, cfg.quota_max,
        cfg.tile_base_zoom, cfg.tile_leaf_zoom, cfg.include_panoramas,
        cfg.min_quality_score, cfg.max_per_sequence, cfg.dry_run_uploads,
    )
