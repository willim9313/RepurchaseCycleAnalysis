# repurchase_cycle/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import load_config
from .modules.data_loader import load_data
from .pipeline import run_all_categories


def _parse_cats_arg(cats_arg: str | None) -> List[str] | None:
    """Parse comma-separated categories argument into a list of strings."""
    if not cats_arg:
        return None
    # 允許: "3C,家電,美妝"
    return [c.strip() for c in cats_arg.split(",") if c.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repurchase_cycle",
        description="Repurchase cycle distribution analysis pipeline.",
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input file (CSV, Parquet, Excel, JSON, Feather, or Pickle).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store reports (JSON and figures).",
    )
    parser.add_argument(
        "--config-path",
        default="configs/default_config.yml",
        help="Path to YAML config file (default: configs/default_config.yml).",
    )
    parser.add_argument(
        "--cats",
        default=None,
        help="Comma-separated list of categories to process (e.g. '3C,家電'). "
             "If omitted, all cats in data will be processed.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config_path)
    cats = _parse_cats_arg(args.cats)

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg: Dict[str, Any] = load_config(config_path)

    df = load_data(input_path)
    run_all_categories(df, cfg, cats=cats)
