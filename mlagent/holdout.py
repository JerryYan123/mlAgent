"""Utilities for finding data files and deterministic holdout validation (fallback).

Module-level functions (find_train_file, find_test_file, find_sample_submission,
get_target_info) are reused by eval_agent.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data discovery functions
# ---------------------------------------------------------------------------

def find_train_file(data_dir: Path) -> Optional[Path]:
    """Find the main training file (CSV or JSON)."""
    for name in ["train.csv", "train.json"]:
        p = data_dir / name
        if p.exists():
            return p
    return None


def find_test_file(data_dir: Path) -> Optional[Path]:
    """Find the main test file (CSV or JSON)."""
    for name in ["test.csv", "test.json"]:
        p = data_dir / name
        if p.exists():
            return p
    return None


def find_sample_submission(data_dir: Path) -> Optional[Path]:
    """Find sample submission to determine target columns."""
    for name in ["sample_submission.csv", "sampleSubmission.csv"]:
        p = data_dir / name
        if p.exists():
            return p
    return None


def get_target_info(data_dir: Path) -> Optional[dict]:
    """Discover ID column, target columns, and train label columns.

    Returns dict with keys:
        id_col, target_cols, train_label_cols, train_path, train_format
    or None if discovery fails.
    """
    train_path = find_train_file(data_dir)
    sample_sub = find_sample_submission(data_dir)
    if not train_path or not sample_sub:
        return None

    try:
        sub_df = pd.read_csv(sample_sub, nrows=5)
        id_col = sub_df.columns[0]
        target_cols = sub_df.columns[1:].tolist()

        # Load train header
        if train_path.suffix == ".csv":
            train_df = pd.read_csv(train_path, nrows=5)
            train_format = "csv"
        elif train_path.suffix == ".json":
            with open(train_path) as f:
                data = json.load(f)
            train_df = pd.DataFrame(data[:5]) if isinstance(data, list) else None
            train_format = "json"
        else:
            return None

        if train_df is None or id_col not in train_df.columns:
            return None

        # Check if submission target cols are in train
        targets_in_train = [c for c in target_cols if c in train_df.columns]
        if targets_in_train:
            train_label_cols = targets_in_train
        else:
            # Find label columns: in train but not in test
            test_path = find_test_file(data_dir)
            test_cols = set()
            if test_path:
                if test_path.suffix == ".csv":
                    test_cols = set(pd.read_csv(test_path, nrows=1).columns)
                elif test_path.suffix == ".json":
                    with open(test_path) as f:
                        td = json.load(f)
                    if isinstance(td, list) and td and isinstance(td[0], dict):
                        test_cols = set(td[0].keys())
            train_label_cols = [
                c for c in train_df.columns
                if c not in test_cols and c != id_col and not c.startswith("_")
            ]
            if not train_label_cols:
                return None

        return {
            "id_col": id_col,
            "target_cols": target_cols,
            "train_label_cols": train_label_cols,
            "train_path": train_path,
            "train_format": train_format,
        }
    except Exception as e:
        logger.warning("get_target_info failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# HoldoutValidator — deterministic fallback (no LLM needed)
# ---------------------------------------------------------------------------

class HoldoutValidator:
    """Manages a hidden holdout split. Used as fallback when EvalAgent fails."""

    def __init__(self, holdout_fraction: float = 0.15, seed: int = 42):
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self.holdout_labels: Optional[pd.DataFrame] = None
        self.target_cols: list[str] = []
        self.id_col: str = ""
        self._train_label_cols: list[str] = []
        self._active = False

    def setup(self, data_dir: Path, agent_input_dir: Path, competition: Any) -> bool:
        """Split training data and set up holdout."""
        try:
            info = get_target_info(data_dir)
            if info is None:
                return False

            self.id_col = info["id_col"]
            self.target_cols = info["target_cols"]
            self._train_label_cols = info["train_label_cols"]
            train_path = info["train_path"]

            if info["train_format"] == "csv":
                train_df = pd.read_csv(train_path)
            else:
                with open(train_path) as f:
                    train_df = pd.DataFrame(json.load(f))

            from sklearn.model_selection import train_test_split
            train_split, holdout = train_test_split(
                train_df, test_size=self.holdout_fraction, random_state=self.seed,
            )

            self.holdout_labels = holdout[[self.id_col] + self._train_label_cols].reset_index(drop=True)

            # Overwrite train with split version
            agent_train = agent_input_dir / train_path.name
            if info["train_format"] == "csv":
                train_split.to_csv(agent_train, index=False)
            else:
                train_split.to_json(agent_train, orient="records")

            # Save holdout features (no labels)
            holdout.drop(columns=self._train_label_cols).to_csv(
                agent_input_dir / "holdout.csv", index=False,
            )

            self._active = True
            logger.info("Holdout: %d → train %d + holdout %d",
                        len(train_df), len(train_split), len(holdout))
            return True
        except Exception as e:
            logger.warning("Holdout setup failed: %s", e)
            return False

    def score(self, agent_dir: Path) -> Optional[float]:
        """Score holdout predictions."""
        if not self._active or self.holdout_labels is None:
            return None
        pred_path = agent_dir / "holdout_predictions.csv"
        if not pred_path.exists():
            return None
        try:
            preds = pd.read_csv(pred_path)
            labels = self.holdout_labels
            merged = labels.merge(preds, on=self.id_col, suffixes=("_true", "_pred"))
            if len(merged) < len(labels) * 0.5:
                return None

            from sklearn.metrics import roc_auc_score, log_loss

            # Multi-target binary (e.g. jigsaw)
            pred_cols = [c for c in self.target_cols if c in preds.columns]
            if pred_cols and all(f"{c}_true" in merged.columns for c in pred_cols):
                scores = []
                for col in pred_cols:
                    try:
                        scores.append(roc_auc_score(merged[f"{col}_true"], merged[f"{col}_pred"]))
                    except Exception:
                        pass
                if scores:
                    return float(np.mean(scores))

            # Categorical label → probability columns (e.g. spooky)
            if len(self._train_label_cols) == 1 and len(self.target_cols) > 1:
                label_col = self._train_label_cols[0]
                if label_col in merged.columns:
                    available = [c for c in sorted(self.target_cols) if c in preds.columns]
                    if available:
                        try:
                            return log_loss(merged[label_col], merged[available].values, labels=available)
                        except Exception:
                            pass

            # Single binary target (e.g. pizza)
            if len(self._train_label_cols) == 1 and len(self.target_cols) == 1:
                label_col = self._train_label_cols[0]
                pred_col = self.target_cols[0]
                if label_col in merged.columns and pred_col in preds.columns:
                    try:
                        return roc_auc_score(merged[label_col], merged[pred_col])
                    except Exception:
                        pass

            return None
        except Exception as e:
            logger.warning("Holdout scoring failed: %s", e)
            return None

    @property
    def active(self) -> bool:
        return self._active