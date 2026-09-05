import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Application" / "Backend"))

from utils.risk_utils import compute_risk_for_ticker
from Models.risk_scorer import ROLLING_VOLATILITY_WINDOW


@patch("utils.risk_utils.load_risk_calibration")
@patch("utils.risk_utils._get_base_classification_probs")
@patch("utils.risk_utils._get_stack_classification_prediction")
@patch("utils.risk_utils._get_base_regression_predictions")
@patch("utils.risk_utils._score_regime")
@patch("utils.risk_utils.extract_regime_features")
def test_compute_risk_for_ticker_low_confidence(
    mock_extract_regime_features,
    mock_score_regime,
    mock_get_base_regression_predictions,
    mock_get_stack_classification_prediction,
    mock_get_base_classification_probs,
    mock_load_risk_calibration
):
    """
    Test that the low_confidence flag is triggered when the available price history
    has fewer data points than the ROLLING_VOLATILITY_WINDOW (which requires 
    ROLLING_VOLATILITY_WINDOW + 1 days of prices to compute the daily returns).
    """
    # Setup mocks
    mock_load_risk_calibration.return_value = {
        "training_p95_volatility": 0.02,
        "max_dispersion": 0.01,
        "iso_forest": MagicMock(),
        "regime_scaler": MagicMock(),
        "d_p1": -0.1
    }
    mock_score_regime.return_value = {
        "decision_score": 0.1,
        "regime_multiplier": 1.0,
        "outlier_gate": False
    }
    
    mock_extract_regime_features.return_value = np.zeros((10, 7))
    mock_get_base_classification_probs.return_value = {
        "rf": [0.3, 0.4, 0.3],
        "xgb": [0.3, 0.4, 0.3],
        "lstm": [0.3, 0.4, 0.3]
    }
    mock_get_stack_classification_prediction.return_value = (
        np.array([0.3, 0.4, 0.3]),
        "Flat",
        ["Down", "Flat", "Up"]
    )
    mock_get_base_regression_predictions.return_value = [0.01, 0.015, 0.01]

    # Create dummy data missing sufficient history
    # ROLLING_VOLATILITY_WINDOW is 20. We need < 20 returns to trigger low_confidence,
    # meaning <= 20 days of price history.
    short_price_history = [100.0 * (1 + 0.01)**i for i in range(10)]
    
    live_data_short = {
        "raw_df": pd.DataFrame({"Close": short_price_history}),
        "latest_row": pd.DataFrame({"Close": [short_price_history[-1]]}),
        "sequence": pd.DataFrame()
    }
    
    result_short = compute_risk_for_ticker("LT", live_data_short)
    assert result_short["regression"]["low_confidence"] is True, \
        "low_confidence should be True when history length is less than ROLLING_VOLATILITY_WINDOW + 1"

    # Test the 0-1 observation guard explicitly (1 price point -> 0 returns)
    zero_return_history = [100.0]
    live_data_zero = {
        "raw_df": pd.DataFrame({"Close": zero_return_history}),
        "latest_row": pd.DataFrame({"Close": [zero_return_history[-1]]}),
        "sequence": pd.DataFrame()
    }
    result_zero = compute_risk_for_ticker("LT", live_data_zero)
    assert result_zero["regression"]["low_confidence"] is True
    assert result_zero["regression"]["current_volatility"] == 0.0

    # Now test with sufficient history
    long_price_history = [100.0 * (1 + 0.01)**i for i in range(ROLLING_VOLATILITY_WINDOW + 5)]
    live_data_long = {
        "raw_df": pd.DataFrame({"Close": long_price_history}),
        "latest_row": pd.DataFrame({"Close": [long_price_history[-1]]}),
        "sequence": pd.DataFrame()
    }
    
    result_long = compute_risk_for_ticker("LT", live_data_long)
    assert result_long["regression"]["low_confidence"] is False, \
        "low_confidence should be False when history is sufficient"
