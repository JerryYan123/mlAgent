"""mle-bench competition loading and grading (self-contained in mlagent)."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mlagent.data_preview import generate_preview

logger = logging.getLogger("mlagent.competition")

# Repo root = parent of package `mlagent/`; sibling `mle-bench` lives under ai4mle-research/
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MLEBENCH_ROOT = _REPO_ROOT.parent / "mle-bench"
if _MLEBENCH_ROOT.is_dir() and str(_MLEBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_MLEBENCH_ROOT))


@dataclass
class GradingReport:
    """Result of grading a submission."""

    score: float
    gold_medal: bool = False
    silver_medal: bool = False
    bronze_medal: bool = False
    above_median: bool = False
    gold_threshold: float = 0.0
    silver_threshold: float = 0.0
    bronze_threshold: float = 0.0
    median_threshold: float = 0.0
    is_lower_better: bool = False
    valid_submission: bool = True
    error: Optional[str] = None

    @property
    def medal(self) -> str:
        if self.gold_medal:
            return "gold"
        if self.silver_medal:
            return "silver"
        if self.bronze_medal:
            return "bronze"
        if self.above_median:
            return "above_median"
        return "none"


@dataclass
class Competition:
    """A competition loaded from mle-bench."""

    id: str
    name: str
    description: str
    data_dir: Path
    private_dir: Optional[Path] = None
    metric_name: str = ""
    is_lower_better: bool = False
    _mlebench_competition: Any = field(default=None, repr=False)

    @classmethod
    def from_mlebench(cls, competition_id: str) -> "Competition":
        """Load from mle-bench registry; download/prepare data if needed."""
        try:
            from mlebench.data import (
                download_and_prepare_dataset,
                get_leaderboard,
                is_dataset_prepared,
            )
            from mlebench.registry import registry
        except ImportError as e:
            raise ImportError(
                "mle-bench is not importable. Install it or clone "
                f"{_MLEBENCH_ROOT} next to this repo (ai4mle-research/mle-bench)."
            ) from e

        comp = registry.get_competition(competition_id)

        if not is_dataset_prepared(comp):
            logger.info("Preparing dataset for %s...", competition_id)
            download_and_prepare_dataset(comp)

        description = ""
        desc_path = comp.public_dir / "description.md"
        if desc_path.exists():
            description = desc_path.read_text(errors="ignore")

        is_lower = False
        try:
            is_lower_better = getattr(comp.grader, "is_lower_better", False)
            if callable(is_lower_better):
                leaderboard = get_leaderboard(comp)
                is_lower = bool(is_lower_better(leaderboard))
            else:
                is_lower = bool(is_lower_better)
        except Exception:
            pass

        return cls(
            id=competition_id,
            name=comp.name,
            description=description,
            data_dir=comp.public_dir,
            private_dir=comp.private_dir,
            metric_name=comp.grader.name if hasattr(comp.grader, "name") else "",
            is_lower_better=is_lower,
            _mlebench_competition=comp,
        )

    def grade(self, submission_path: str | Path) -> GradingReport:
        """Grade a submission CSV via mle-bench."""
        if self._mlebench_competition is None:
            raise RuntimeError("Cannot grade: competition not loaded from mle-bench")

        try:
            from mlebench.grade import grade_csv
        except ImportError as e:
            raise ImportError("mle-bench grading module not available") from e

        submission_path = Path(submission_path)
        if not submission_path.exists():
            return GradingReport(
                score=0.0, valid_submission=False, error="Submission file not found"
            )

        try:
            report = grade_csv(submission_path, self._mlebench_competition)
            return GradingReport(
                score=report.score,
                gold_medal=report.gold_medal,
                silver_medal=report.silver_medal,
                bronze_medal=report.bronze_medal,
                above_median=report.above_median,
                gold_threshold=report.gold_threshold,
                silver_threshold=report.silver_threshold,
                bronze_threshold=report.bronze_threshold,
                median_threshold=report.median_threshold,
                is_lower_better=self.is_lower_better,
                valid_submission=report.valid_submission,
            )
        except Exception as e:
            logger.error("Grading failed: %s", e)
            return GradingReport(score=0.0, valid_submission=False, error=str(e))

    def get_data_preview(self) -> str:
        """Short text preview of competition files for the planning agent."""
        return generate_preview(self.data_dir)
