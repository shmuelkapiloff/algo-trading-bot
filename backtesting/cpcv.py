"""
Combinatorial Purged Cross-Validation (CPCV).

Replaces k-fold cross-validation for financial time series.

Problem with standard k-fold:
  Financial time series have autocorrelation and regime dependencies.
  k-fold leaks information across folds because it ignores time ordering
  and doesn't purge overlapping labels between train and test sets.

CPCV solution (Lopez de Prado, "Advances in Financial Machine Learning"):
  1. Split bars into N groups (not shuffled — preserving time order)
  2. For each combination of k test groups (C(N,k) combinations):
     a. Use remaining N-k groups as training set
     b. Purge: remove training samples whose labels overlap with the test period
     c. Embargo: remove training samples within H bars of the test boundary
  3. Report mean and std of performance across all combinations

This gives a nearly unbiased OOS performance estimate.

Usage::

    cpcv = CPCV(n_splits=6, n_test_splits=2, embargo_bars=10)
    for train_idx, test_idx in cpcv.split(bars):
        model.fit(bars.iloc[train_idx])
        score = evaluate(bars.iloc[test_idx])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CpcvFold:
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_group_ids: tuple[int, ...]


class CPCV:
    """
    Combinatorial Purged Cross-Validation iterator.

    Parameters
    ----------
    n_splits        : Total number of groups to split the time series into (N)
    n_test_splits   : Number of groups used as test per fold (k)
    embargo_bars    : Number of bars to embargo on each side of the test boundary
    purge           : If True, removes training samples with labels overlapping test period
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_bars: int = 5,
        purge: bool = True,
    ) -> None:
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_bars = embargo_bars
        self.purge = purge

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(
        self,
        X: np.ndarray,
        label_spans: Optional[np.ndarray] = None,
    ) -> Generator[CpcvFold, None, None]:
        """
        Yield CpcvFold objects with train/test index arrays.

        Parameters
        ----------
        X            : Feature matrix or array; only its length is used
        label_spans  : Optional (n_samples, 2) array of (label_start, label_end) indices
                       for purging overlapping labels. If None, no purging is done.
        """
        n = len(X)
        groups = np.array_split(np.arange(n), self.n_splits)

        fold_id = 0
        for test_group_ids in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([groups[i] for i in test_group_ids])
            train_groups = [g for i, g in enumerate(groups) if i not in test_group_ids]
            train_idx = np.concatenate(train_groups) if train_groups else np.array([], dtype=int)

            if self.embargo_bars > 0:
                test_min = int(test_idx.min())
                test_max = int(test_idx.max())
                embargo_mask = (
                    (train_idx >= test_min - self.embargo_bars) &
                    (train_idx <= test_max + self.embargo_bars)
                )
                train_idx = train_idx[~embargo_mask]

            if self.purge and label_spans is not None:
                test_min = int(test_idx.min())
                test_max = int(test_idx.max())
                purge_mask = (
                    (label_spans[train_idx, 1] >= test_min) &
                    (label_spans[train_idx, 0] <= test_max)
                )
                train_idx = train_idx[~purge_mask]

            if len(train_idx) == 0:
                logger.warning("cpcv.split fold=%d: train set empty after purge/embargo", fold_id)
                fold_id += 1
                continue

            yield CpcvFold(
                fold_id=fold_id,
                train_indices=train_idx,
                test_indices=test_idx,
                test_group_ids=test_group_ids,
            )
            fold_id += 1

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def n_folds(self) -> int:
        """Total number of folds = C(N, k)."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def __repr__(self) -> str:
        return (
            f"CPCV(n_splits={self.n_splits}, n_test_splits={self.n_test_splits}, "
            f"embargo_bars={self.embargo_bars}, n_folds={self.n_folds})"
        )
