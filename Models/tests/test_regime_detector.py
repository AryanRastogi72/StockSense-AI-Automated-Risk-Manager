"""Integration tests for the Isolation Forest regime detector on real LT.NS data.

These tests validate that the regime detector fires on known volatile periods
(COVID-19, March 2020) and stays quiet on known calm periods (mid-2024).
They also validate the calibration pipeline end-to-end.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regime_detector import (
    train_regime_detector,
    extract_regime_features,
    compute_training_calibration,
    score_regime,
)


TICKER = "LT"
DATA_DIR = "Supporting Data"
SPLIT_RATIO = 0.8


@pytest.fixture(scope="module")
def lt_data():
    """Loads the full LT cleaned dataset and splits into train/test."""
    df = pd.read_csv(
        f"{DATA_DIR}/{TICKER}_cleaned_data.csv", index_col="Date", parse_dates=True
    )
    split_index = int(len(df) * SPLIT_RATIO)
    return {
        "df": df,
        "split_index": split_index,
    }


@pytest.fixture(scope="module")
def trained_detector(lt_data):
    X_regime_train = extract_regime_features(lt_data["df"].iloc[:lt_data["split_index"]])
    iso_forest, regime_scaler, d_p1 = train_regime_detector(X_regime_train)
    return iso_forest, regime_scaler, d_p1


class TestRegimeDetectorHistorical:
    def test_covid_crash_fires_anomaly(self, lt_data, trained_detector):
        """March 2020 should be flagged as anomalous (regime multiplier > 1.0)."""
        iso_forest, regime_scaler, d_p1 = trained_detector
        df = lt_data["df"]

        covid_window = df.loc["2020-03-01":"2020-03-31"]
        if len(covid_window) == 0:
            pytest.skip("No data for March 2020 in the dataset")

        covid_features = extract_regime_features(covid_window)
        covid_scaled = regime_scaler.transform(covid_features)
        scores = iso_forest.decision_function(covid_scaled)

        anomalous_days = np.sum(scores < 0)
        total_days = len(scores)

        assert anomalous_days / total_days > 0.5, (
            f"Expected majority of March 2020 days to be anomalous, "
            f"but only {anomalous_days}/{total_days} were flagged."
        )

    def test_mid_2024_stays_quiet(self, lt_data, trained_detector):
        """May-July 2024 should be normal (regime multiplier ~1.0)."""
        iso_forest, regime_scaler, d_p1 = trained_detector
        df = lt_data["df"]

        calm_window = df.loc["2024-05-01":"2024-07-31"]
        if len(calm_window) == 0:
            pytest.skip("No data for mid-2024 in the dataset")

        calm_features = extract_regime_features(calm_window)
        calm_scaled = regime_scaler.transform(calm_features)
        scores = iso_forest.decision_function(calm_scaled)

        normal_days = np.sum(scores >= 0)
        total_days = len(scores)

        assert normal_days / total_days > 0.7, (
            f"Expected >70% of mid-2024 days to be normal, "
            f"but only {normal_days}/{total_days} were flagged as normal."
        )

    def test_covid_has_higher_anomaly_than_calm(self, lt_data, trained_detector):
        """The average anomaly score for COVID must be strictly worse
        (more negative) than for the calm mid-2024 period."""
        iso_forest, regime_scaler, d_p1 = trained_detector
        df = lt_data["df"]

        covid_window = df.loc["2020-03-01":"2020-03-31"]
        calm_window = df.loc["2024-05-01":"2024-07-31"]

        if len(covid_window) == 0 or len(calm_window) == 0:
            pytest.skip("Missing data for one of the test windows")

        covid_scaled = regime_scaler.transform(extract_regime_features(covid_window))
        calm_scaled = regime_scaler.transform(extract_regime_features(calm_window))

        covid_scores = iso_forest.decision_function(covid_scaled)
        calm_scores = iso_forest.decision_function(calm_scaled)

        assert np.mean(covid_scores) < np.mean(calm_scores), (
            f"Expected COVID mean score ({np.mean(covid_scores):.4f}) to be lower "
            f"than calm mean score ({np.mean(calm_scores):.4f})"
        )


class TestCalibrationPipeline:
    def test_calibration_produces_valid_constants(self):
        calibration = compute_training_calibration(TICKER, DATA_DIR)

        assert calibration["d_p1"] < 0, "D_p1 should be negative"
        assert calibration["training_p95_volatility"] > 0, "P95 vol should be positive"
        assert calibration["max_dispersion"] > 0, "MaxDispersion should be positive"
        assert calibration["iso_forest"] is not None
        assert calibration["regime_scaler"] is not None

    def test_d_p1_is_below_median_decision_score(self):
        """D_p1 (1st percentile) must be well below the median training score."""
        calibration = compute_training_calibration(TICKER, DATA_DIR)

        df = pd.read_csv(
            f"{DATA_DIR}/{TICKER}_cleaned_data.csv", index_col="Date", parse_dates=True
        )
        split_index = int(len(df) * SPLIT_RATIO)
        X_regime_train = extract_regime_features(df.iloc[:split_index])
        X_scaled = calibration["regime_scaler"].transform(X_regime_train)

        scores = calibration["iso_forest"].decision_function(X_scaled)
        median_score = float(np.median(scores))

        assert calibration["d_p1"] < median_score, (
            f"D_p1 ({calibration['d_p1']:.4f}) should be below "
            f"median ({median_score:.4f})"
        )
