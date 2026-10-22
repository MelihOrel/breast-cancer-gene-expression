"""Load configs/config.yaml and resolve paths relative to the repository root."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None, quick: bool = False) -> dict[str, Any]:
    """Read the YAML config.

    ``quick`` shrinks the resampling (1 repeat, fewer trees) for a fast smoke run;
    the published numbers always come from the full configuration.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    root = path.resolve().parents[1] if path.parent.name == "configs" else REPO_ROOT
    cfg["paths"] = {k: str((root / v).resolve()) for k, v in cfg["paths"].items()}

    if quick:
        # Smoke-test settings, written to separate folders so published outputs are never overwritten.
        cfg = copy.deepcopy(cfg)
        cfg["cv"]["n_repeats"] = 1
        cfg["audit"]["n_repeats"] = 1
        cfg["models"]["rsf"]["n_estimators"] = 50
        cfg["paths"]["results"] += "_quick"
        cfg["paths"]["figures"] += "_quick"
    cfg["quick"] = quick
    cfg.setdefault("figures", {}).setdefault("titles", {})
    return cfg
