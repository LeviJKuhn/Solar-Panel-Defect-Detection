"""Config loading with project-root-relative path resolution.

Every path in config.yaml is written relative to the project root. This module
resolves them to absolute paths so it doesn't matter whether you're running
from the project root (scripts/), from notebooks/, or anywhere else.
"""
from pathlib import Path

import yaml

# src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Config keys whose values are paths and should be made absolute.
_PATH_KEYS = [
    ("data", "data_dir"),
    ("output", "model_path"),
    ("output", "figures_dir"),
]


def load_config(path=None):
    """Load config.yaml and convert its relative paths to absolute ones."""
    config_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for section, key in _PATH_KEYS:
        if section in cfg and key in cfg[section]:
            cfg[section][key] = str(PROJECT_ROOT / cfg[section][key])

    return cfg
