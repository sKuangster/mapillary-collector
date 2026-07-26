"""Command line interface. `mapillary --help` lists everything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import diagnostics
from .config import Config, load_dotenv
from .logging_setup import setup_logging
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapillary",
        description="Collect a street-level image dataset from Mapillary.",
    )
    parser.add_argument("--data-dir", type=Path,
                        help="where state, staging and shards live")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="file holding MAPILLARY_TOKEN and HF_TOKEN")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="collect images (resumable, safe to rerun)")
    run.add_argument("--workers", type=int, help="parallel fetches (default 6)")
    run.add_argument("--shard-size", type=int, help="samples per tar (default 1000)")
    run.add_argument("--repo", dest="hf_repo_id", help="hugging face dataset repo")
    run.add_argument("--country", action="append", dest="countries",
                     help="only this country (repeatable)")
    run.add_argument("--exclude", action="append", dest="excluded",
                     help="skip this country (repeatable)")
    run.add_argument("--quota-max", type=int, help="ceiling per country")
    run.add_argument("--quota-min", type=int, help="floor per country")
    run.add_argument("--quota-k", type=float, help="quota multiplier")
    run.add_argument("--quota-alpha", type=float, help="quota exponent (0.5 = sqrt)")
    run.add_argument("--include-panoramas", action="store_true", default=None)
    run.add_argument("--min-quality", type=float, dest="min_quality_score")
    run.add_argument("--retry-exhausted", action="store_true", default=None,
                     dest="retry_exhausted",
                     help="revisit countries previously found to have no coverage")
    run.add_argument("--dry-run", action="store_true", default=None,
                     dest="dry_run_uploads", help="collect but never upload")

    sub.add_parser("status", help="progress summary")
    sub.add_parser("repair", help="undo outage damage: requeue tiles, reopen countries")
    sub.add_parser("doctor", help="pre-flight check")
    sub.add_parser("quota", help="preview the quota curve")

    countries_cmd = sub.add_parser("countries", help="per-country results")
    countries_cmd.add_argument("--limit", type=int, default=40)

    verify = sub.add_parser("verify", help="check shard integrity")
    verify.add_argument("--remote", action="store_true",
                        help="also verify against the hub")

    finalize = sub.add_parser(
        "finalize", help="pack leftover staged images into a final short shard")
    finalize.add_argument("--repo", dest="hf_repo_id")

    reset = sub.add_parser("reset", help="back up and wipe local state")
    reset.add_argument("--confirm", default="", help='must be exactly: RESET')

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    load_dotenv(args.env_file)
    cfg = Config()
    overrides = {
        "data_dir": getattr(args, "data_dir", None),
        "log_level": getattr(args, "log_level", None),
        "workers": getattr(args, "workers", None),
        "shard_size": getattr(args, "shard_size", None),
        "hf_repo_id": getattr(args, "hf_repo_id", None),
        "quota_max": getattr(args, "quota_max", None),
        "quota_min": getattr(args, "quota_min", None),
        "quota_k": getattr(args, "quota_k", None),
        "quota_alpha": getattr(args, "quota_alpha", None),
        "include_panoramas": getattr(args, "include_panoramas", None),
        "min_quality_score": getattr(args, "min_quality_score", None),
        "dry_run_uploads": getattr(args, "dry_run_uploads", None),
        "retry_exhausted": getattr(args, "retry_exhausted", None),
    }
    countries = getattr(args, "countries", None)
    if countries:
        overrides["countries_include"] = tuple(countries)
    excluded = getattr(args, "excluded", None)
    if excluded:
        overrides["countries_exclude"] = tuple(excluded)
    return cfg.with_overrides(**overrides)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    if args.command == "run":
        log = setup_logging(cfg)
        Pipeline(cfg, log).run()
        return 0

    if args.command == "finalize":
        log = setup_logging(cfg)
        pipe = Pipeline(cfg, log)
        pipe.startup()
        try:
            pipe.finalize_partial_shard()
        finally:
            pipe._finalize("finalize")
        return 0

    if args.command == "status":
        diagnostics.status(cfg)
    elif args.command == "countries":
        diagnostics.countries(cfg, limit=args.limit)
    elif args.command == "doctor":
        diagnostics.doctor(cfg)
    elif args.command == "quota":
        diagnostics.show_quota(cfg)
    elif args.command == "verify":
        diagnostics.verify_local(cfg)
        if args.remote:
            diagnostics.verify_remote(cfg)
    elif args.command == "repair":
        diagnostics.repair(cfg)
    elif args.command == "reset":
        diagnostics.reset(cfg, args.confirm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
