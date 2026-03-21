#!/usr/bin/env python3
"""Verify mlAgent: imports, config, mle-bench competition load.

Run:  python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    print("1) Imports...")
    from mlagent import __version__

    print(f"   mlagent {__version__} OK")

    print("2) Config load...")
    from mlagent.config import load_config

    cfg = load_config(_ROOT / "configs" / "default.yaml")
    print(f"   competition_id={cfg.competition_id!r} OK")

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    print("3) OPENAI_API_KEY:", "set" if key else "not set (export before full runs)")

    print("4) mle-bench competition...")
    try:
        from mlagent.competition_loader import load_competition

        comp = load_competition(cfg.competition_id)
        print(f"   OK data_dir={comp.data_dir}")
    except Exception as e:
        print(f"   FAILED: {e}")
        print("   Need: ../mle-bench (sibling repo) or installed mlebench, ~/.kaggle/kaggle.json")
        return 1

    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
