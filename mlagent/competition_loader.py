"""Load mle-bench competitions (implemented in `mlagent.competition`)."""

from __future__ import annotations

from mlagent.competition import Competition


def load_competition(competition_id: str) -> Competition:
    return Competition.from_mlebench(competition_id)
