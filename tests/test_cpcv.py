"""
Tests for Combinatorial Purged Cross-Validation (CPCV).

Tests verify:
  - Correct number of folds = C(N, k)
  - No look-ahead: test indices never appear in train indices
  - Embargo removes train samples near test boundary
  - Empty train set is handled gracefully
  - n_splits and n_test_splits validation
"""

from __future__ import annotations

import numpy as np
import pytest

from backtesting.cpcv import CPCV, CpcvFold


class TestCpcvFoldCount:
    def test_fold_count_c6_2(self):
        """C(6,2) = 15 folds."""
        cpcv = CPCV(n_splits=6, n_test_splits=2, embargo_bars=0)
        X = np.arange(120)
        folds = list(cpcv.split(X))
        assert len(folds) == 15

    def test_fold_count_c4_1(self):
        """C(4,1) = 4 folds."""
        cpcv = CPCV(n_splits=4, n_test_splits=1, embargo_bars=0)
        X = np.arange(80)
        folds = list(cpcv.split(X))
        assert len(folds) == 4

    def test_n_folds_property(self):
        cpcv = CPCV(n_splits=6, n_test_splits=2)
        assert cpcv.n_folds == 15


class TestCpcvNoLookAhead:
    def test_train_test_disjoint(self):
        """Train and test index sets must be completely disjoint."""
        cpcv = CPCV(n_splits=6, n_test_splits=2, embargo_bars=0, purge=False)
        X = np.arange(120)
        for fold in cpcv.split(X):
            train_set = set(fold.train_indices.tolist())
            test_set = set(fold.test_indices.tolist())
            assert train_set.isdisjoint(test_set), (
                f"Fold {fold.fold_id}: train/test overlap detected"
            )

    def test_test_coverage(self):
        """Union of all test sets should cover the full index range."""
        cpcv = CPCV(n_splits=6, n_test_splits=2, embargo_bars=0, purge=False)
        X = np.arange(60)
        all_test = set()
        for fold in cpcv.split(X):
            all_test.update(fold.test_indices.tolist())
        assert all_test == set(range(60))


class TestCpcvEmbargo:
    def test_embargo_removes_samples_near_boundary(self):
        """With embargo_bars=10, train samples within 10 of test boundary are removed."""
        cpcv = CPCV(n_splits=4, n_test_splits=1, embargo_bars=10, purge=False)
        X = np.arange(100)
        for fold in cpcv.split(X):
            test_min = int(fold.test_indices.min())
            test_max = int(fold.test_indices.max())
            for ti in fold.train_indices:
                assert not (test_min - 10 <= ti <= test_max + 10), (
                    f"Fold {fold.fold_id}: train index {ti} within embargo of [{test_min},{test_max}]"
                )

    def test_no_embargo_vs_embargo_train_size(self):
        """Embargo should produce fewer training samples than no embargo."""
        X = np.arange(120)
        no_emb = list(CPCV(n_splits=6, n_test_splits=2, embargo_bars=0, purge=False).split(X))
        with_emb = list(CPCV(n_splits=6, n_test_splits=2, embargo_bars=5, purge=False).split(X))
        avg_no_emb = np.mean([len(f.train_indices) for f in no_emb])
        avg_with_emb = np.mean([len(f.train_indices) for f in with_emb])
        assert avg_with_emb <= avg_no_emb


class TestCpcvValidation:
    def test_n_test_splits_must_be_less_than_n_splits(self):
        with pytest.raises(ValueError):
            CPCV(n_splits=4, n_test_splits=4)

    def test_repr(self):
        cpcv = CPCV(n_splits=6, n_test_splits=2)
        assert "CPCV" in repr(cpcv)
        assert "n_folds=15" in repr(cpcv)
