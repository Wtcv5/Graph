from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_dataset_config
from .pipeline import preprocess_dataset
from .validation import validate_dataset


DEFAULT_CONFIG = Path("configs/datasets/right_tunnel_exit_vpvs.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tunnel velocity model pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Run CSV-to-NetCDF preprocessing.")
    add_shared_arguments(preprocess_parser)
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory from the dataset config.",
    )
    preprocess_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate a generated NetCDF file.")
    add_shared_arguments(validate_parser)
    validate_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory from the dataset config.",
    )
    validate_parser.add_argument(
        "--nc-path",
        type=Path,
        help="Validate an explicit NetCDF file instead of the configured output path.",
    )

    return parser


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve config and data paths.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a dataset config JSON file.",
    )


def resolve_config(args: argparse.Namespace):
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else (project_root / args.config).resolve()
    config = load_dataset_config(config_path, project_root=project_root)

    if getattr(args, "output_dir", None):
        output_dir = args.output_dir if args.output_dir.is_absolute() else (project_root / args.output_dir).resolve()
        config = config.with_output_dir(output_dir)

    return config


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(args)

    if args.command == "preprocess":
        preprocess_dataset(config, force=args.force)
        return 0

    if args.command == "validate":
        nc_path = args.nc_path
        if nc_path is None:
            nc_path = config.output_nc_path
        elif not nc_path.is_absolute():
            nc_path = (args.project_root.resolve() / nc_path).resolve()

        validate_dataset(nc_path)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
