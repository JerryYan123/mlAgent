#!/usr/bin/env python3
"""Entry point: Planning + Coding agents for MLE-bench."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("mlAgent starting...", flush=True)
    try:
        from mlagent.config import load_config
        from mlagent.orchestrator_strategies import run_experiment

        main()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
