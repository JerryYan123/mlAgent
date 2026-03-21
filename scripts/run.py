#!/usr/bin/env python3
"""Entry point: Planning + Coding agents for MLE-bench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mlagent.config import load_config
from mlagent.orchestrator import run_experiment


def main() -> None:
    p = argparse.ArgumentParser(description="mlAgent: Planning + Coding for MLE-bench")
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "default.yaml")
    p.add_argument("--competition", type=str, default=None, help="Override competition_id")
    p.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf overrides, e.g. max_rounds=3 planning_llm.model_name=o4-mini",
    )
    args = p.parse_args()
    overrides = list(args.overrides)
    if args.competition:
        overrides.append(f"competition_id={args.competition}")

    cfg = load_config(args.config, overrides if overrides else None)
    out = run_experiment(cfg)
    print("Done:", out)


if __name__ == "__main__":
    main()
