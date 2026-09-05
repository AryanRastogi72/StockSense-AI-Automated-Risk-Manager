import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def generate_cv_indices(total_samples: int, split_ratio: float, n_splits: int, lookback: int):
    """
    Generates Out-Of-Fold (OOF) cross-validation indices ensuring strict chronological order
    and no leakage into the global test set.
    """
    split_index = int(total_samples * split_ratio)
    meta_train_indices = np.arange(split_index)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    
    for train_idx, val_idx in tscv.split(meta_train_indices):
        if len(train_idx) < lookback:
            raise ValueError(f"Training set size ({len(train_idx)}) is smaller than lookback ({lookback}). Decrease n_splits.")
        folds.append((train_idx, val_idx))
        
    return folds
