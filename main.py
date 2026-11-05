"""
RAG Bench Lab — main entry point.

Usage
-----
    uv run python main.py --recipe R0 --scenario nq --mode dev
    uv run python main.py --recipe R1 --scenario hotpotqa --mode eval
    uv run python main.py --recipe R4 --scenario nq --mode eval --description "Baseline R4" --comment "seed=42"

Available scenario IDs : nq | hotpotqa | watsonx | dabstep
Available recipe IDs   : R0 | R1 | R2 | R3 | R4  (Phase 1 baseline ladder)
Modes                  : dev (5 questions) | eval (10 questions)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a RAG Bench Lab experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recipe",
        required=True,
        metavar="RECIPE_ID",
        help="Recipe to run (e.g. R0, R1, R2, R3, R4)",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        metavar="SCENARIO_ID",
        help="Scenario dataset to use (nq | hotpotqa | watsonx | dabstep)",
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "eval"],
        default="dev",
        help="Subset mode: dev (5 questions) or eval (10 questions). Default: dev",
    )
    parser.add_argument(
        "--description",
        default="",
        metavar="TEXT",
        help="Optional free-text description stored in experiment_metadata JSON.",
    )
    parser.add_argument(
        "--comment",
        action="append",
        dest="comments",
        default=None,
        metavar="TEXT",
        help="Optional short comment (can be passed multiple times).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Root directory for experiment outputs. Default: output/",
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        metavar="DIR",
        help="Root directory for RagWorkbench cache. Default: cache/",
    )
    args = parser.parse_args()

    # Import here so argparse --help works without heavy imports
    from ragbenchlab.runner.experiment_runner import ExperimentRunner

    runner = ExperimentRunner(
        output_root=Path(args.output_dir),
        cache_root=Path(args.cache_dir),
    )

    try:
        out_dir = runner.run(
            recipe_id=args.recipe,
            scenario_id=args.scenario,
            mode=args.mode,
            description=args.description,
            comments=args.comments,
        )
        print(f"\n✓ Experiment complete. Results in: {out_dir}")
    except KeyError as exc:
        print(f"\n✗ Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as exc:
        print(f"\n✗ Not implemented: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
