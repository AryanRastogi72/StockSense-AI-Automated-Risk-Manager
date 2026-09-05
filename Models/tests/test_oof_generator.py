import pytest
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import sys
from pathlib import Path

# Add Models directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oof_generator import generate_cv_indices

def test_strict_temporal_split():
    """
    Test that for every fold, all training indices are strictly before any validation indices.
    This prevents look-ahead bias in the meta-learner training set.
    """
    total_samples = 1000
    split_ratio = 0.8
    n_splits = 5
    lookback = 30
    
    folds = generate_cv_indices(total_samples, split_ratio, n_splits, lookback)
    
    assert len(folds) == n_splits, f"Expected {n_splits} folds, got {len(folds)}"
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        assert len(train_idx) > 0, f"Fold {fold_idx} has empty training set"
        assert len(val_idx) > 0, f"Fold {fold_idx} has empty validation set"
        assert np.max(train_idx) < np.min(val_idx), \
            f"Fold {fold_idx}: Train indices overlap or exceed validation indices temporally!"

def test_no_index_overlap():
    """
    Test that there is absolutely no intersection between train and validation indices in any fold.
    """
    total_samples = 1000
    split_ratio = 0.8
    n_splits = 5
    lookback = 30
    
    folds = generate_cv_indices(total_samples, split_ratio, n_splits, lookback)
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        intersection = set(train_idx).intersection(set(val_idx))
        assert len(intersection) == 0, f"Fold {fold_idx}: Overlapping indices found: {intersection}"

def test_meta_train_has_no_test_data():
    """
    Test that the validation indices across all folds (which become the meta-learner's training set)
    do not contain ANY data from the global 20% holdout test set.
    """
    total_samples = 1000
    split_ratio = 0.8
    n_splits = 5
    lookback = 30
    
    # 800 is the start of the 20% holdout
    holdout_start_idx = int(total_samples * split_ratio)
    
    folds = generate_cv_indices(total_samples, split_ratio, n_splits, lookback)
    
    all_val_indices = []
    for _, val_idx in folds:
        all_val_indices.extend(val_idx)
        
    all_val_indices = np.array(all_val_indices)
    
    # Check that no validation index is in the holdout set
    leakage = all_val_indices[all_val_indices >= holdout_start_idx]
    assert len(leakage) == 0, \
        f"Leakage detected! {len(leakage)} indices in the meta-train set are from the global test set."

def test_lookback_safety():
    """
    Test that there are enough samples in the training set to accommodate the LSTM lookback window.
    """
    total_samples = 1000
    split_ratio = 0.8
    n_splits = 5
    lookback = 30
    
    folds = generate_cv_indices(total_samples, split_ratio, n_splits, lookback)
    
    for fold_idx, (train_idx, _) in enumerate(folds):
        assert len(train_idx) >= lookback, \
            f"Fold {fold_idx}: Train set size ({len(train_idx)}) is smaller than lookback ({lookback})"
