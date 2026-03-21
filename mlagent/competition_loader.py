"""Resolve myAgent Competition (mle-bench) from sibling checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def load_competition(competition_id: str):
    root = Path(__file__).resolve().parents[2]
    myagent_root = root / "myAgent"
    if myagent_root.exists() and str(myagent_root) not in sys.path:
        sys.path.insert(0, str(myagent_root))
    from myagent.competition import Competition  # noqa: WPS433

    return Competition.from_mlebench(competition_id)
