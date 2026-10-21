"""Load configuration files from config/ and resolve data paths."""
import os
from pathlib import Path

import yaml

# Repository root = two levels above this file (src/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


def load_yaml(name):
    """Read config/<name> and return its contents as a dict."""
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths():
    """Return a dict of Path objects: 'root', 'ice', 'ev', 'processed'.

    The environment variable MTM_DATA_ROOT, if set, overrides data_root
    in paths.yaml (used in Colab, where the data are on Google Drive).
    A relative data_root refers to the repository root.
    """
    cfg = load_yaml("paths.yaml")
    root = Path(os.environ.get("MTM_DATA_ROOT", cfg["data_root"]))
    if not root.is_absolute():
        root = REPO_ROOT / root
    paths = {name: root / rel for name, rel in cfg["raw"].items()}
    paths["processed"] = root / cfg["processed"]
    paths["root"] = root
    return paths
