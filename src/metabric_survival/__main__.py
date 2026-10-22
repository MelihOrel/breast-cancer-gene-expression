"""Command line entry point.

    python -m metabric_survival                 # full pipeline (all stages)
    python -m metabric_survival --stage cv      # one stage
    python -m metabric_survival --quick         # fast smoke run into results_quick/
"""
from __future__ import annotations

import argparse
import warnings

from .config import load_config
from .pipeline import STAGES, run


def main() -> None:
    parser = argparse.ArgumentParser(prog="metabric_survival", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml (default: configs/config.yaml)")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all", nargs="+")
    parser.add_argument("--quick", action="store_true", help="1 CV repeat, small forest, separate output folders")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(args.config, quick=args.quick)
    stages = list(STAGES) if "all" in args.stage else args.stage
    run(cfg, stages)


if __name__ == "__main__":
    main()
