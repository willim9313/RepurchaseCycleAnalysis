# repurchase_cycle/config.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load pipeline configuration from a YAML file.

    Params
    ------
    path : str | os.PathLike
        Path to the YAML config file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration as a nested dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the config file is empty or cannot be parsed.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at top-level: {config_path}")

    return data
