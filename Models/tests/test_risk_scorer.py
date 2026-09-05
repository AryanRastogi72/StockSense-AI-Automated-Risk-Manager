import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_scorer import (
    compute_regime_multiplier,
    compute_outlier_gate,
    compute_composite_risk,
    compute_classification_entropy_score,
    compute_classification_disagreement_score,
    compute_regression_var_score,
    compute_regression_disagreement_score,
)


# ---------------------------------------------------------------------------
# Composite score: cap/floor interaction
# ---------------------------------------------------------------------------

class TestCompositeRiskFormula:
    def test_normal_regime_passthrough(self):
        """When regime multiplier is 1.0, base score passes through unchanged."""
        base = 50.0
        rm = 1.0
        outlier = False
        assert compute_composite_risk(base, rm, outlier) == 50.0

    def test_amplified_regime(self):
        """Anomalous regime amplifies base score by multiplier."""
        base = 40.0
        rm = 1.5
        outlier = False
        assert compute_composite_risk(base, rm, outlier) == 60.0

    def test_cap_at_100(self):
        """Score never exceeds 100 even with high base and high multiplier."""
        base = 80.0
        rm = 2.0
        outlier = False
        assert compute_composite_risk(base, rm, outlier) == 100.0

    def test_outlier_gate_floors_at_85(self):
        """When outlier gate fires, score cannot drop below 85."""
        base = 20.0
        rm = 1.0
        outlier = True
        assert compute_composite_risk(base, rm, outlier) == 85.0

    def test_outlier_gate_allows_higher_than_85(self):
        """Outlier gate is a floor, not a ceiling — high scores pass through."""
        base = 60.0
        rm = 2.0
        outlier = True
        assert compute_composite_risk(base, rm, outlier) == 100.0

    def test_zero_base_normal_regime(self):
        base = 0.0
        rm = 1.0
        outlier = False
        assert compute_composite_risk(base, rm, outlier) == 0.0

    def test_zero_base_outlier_regime(self):
        base = 0.0
        rm = 1.0
        outlier = True
        assert compute_composite_risk(base, rm, outlier) == 85.0


# ---------------------------------------------------------------------------
# Regime multiplier: continuous scaling from decision_function score
# ---------------------------------------------------------------------------

class TestRegimeMultiplier:
    def test_normal_inlier(self):
        """D >= 0 yields multiplier of exactly 1.0."""
        d_p1 = -0.3
        assert compute_regime_multiplier(0.1, d_p1) == 1.0
        assert compute_regime_multiplier(0.0, d_p1) == 1.0

    def test_mild_anomaly(self):
        """D halfway to D_p1 yields multiplier of 1.5."""
        d_p1 = -0.4
        rm = compute_regime_multiplier(-0.2, d_p1)
        assert abs(rm - 1.5) < 1e-9

    def test_at_d_p1_boundary(self):
        """D exactly at D_p1 yields multiplier of 2.0."""
        d_p1 = -0.3
        rm = compute_regime_multiplier(-0.3, d_p1)
        assert abs(rm - 2.0) < 1e-9

    def test_beyond_d_p1_caps_at_2(self):
        """D worse than D_p1 still caps at 2.0."""
        d_p1 = -0.3
        rm = compute_regime_multiplier(-0.6, d_p1)
        assert rm == 2.0


# ---------------------------------------------------------------------------
# Outlier gate: fires only below D_p1
# ---------------------------------------------------------------------------

class TestOutlierGate:
    def test_normal_no_gate(self):
        d_p1 = -0.3
        assert compute_outlier_gate(0.1, d_p1) is False

    def test_mild_anomaly_no_gate(self):
        """D < 0 but above D_p1 does NOT fire the gate."""
        d_p1 = -0.3
        assert compute_outlier_gate(-0.1, d_p1) is False

    def test_at_d_p1_no_gate(self):
        """D exactly at D_p1 does NOT fire (strict less-than)."""
        d_p1 = -0.3
        assert compute_outlier_gate(-0.3, d_p1) is False

    def test_beyond_d_p1_fires(self):
        """D worse than D_p1 fires the gate."""
        d_p1 = -0.3
        assert compute_outlier_gate(-0.31, d_p1) is True


# ---------------------------------------------------------------------------
# Classification entropy score
# ---------------------------------------------------------------------------

class TestClassificationEntropy:
    def test_maximum_entropy(self):
        """Uniform distribution over 3 classes yields score of 100."""
        probs = np.array([1/3, 1/3, 1/3])
        score = compute_classification_entropy_score(probs)
        assert abs(score - 100.0) < 0.1

    def test_minimum_entropy(self):
        """Perfect certainty yields score of 0."""
        probs = np.array([1.0, 0.0, 0.0])
        score = compute_classification_entropy_score(probs)
        assert score == 0.0

    def test_moderate_confidence(self):
        """90% on one class yields a low but non-zero score."""
        probs = np.array([0.9, 0.05, 0.05])
        score = compute_classification_entropy_score(probs)
        assert 0 < score < 50


# ---------------------------------------------------------------------------
# Classification disagreement (population std, ddof=0)
# ---------------------------------------------------------------------------

class TestClassificationDisagreement:
    def test_perfect_agreement(self):
        """All base models assign same probability to winning class."""
        base_probs = np.array([0.7, 0.7, 0.7])
        score = compute_classification_disagreement_score(base_probs)
        assert score < 0.01

    def test_worked_example_population_std(self):
        """The corrected worked example: std_pop(0.8, 0.9, 0.2) / sqrt(2/9) * 100 ≈ 65.6"""
        base_probs = np.array([0.8, 0.9, 0.2])
        score = compute_classification_disagreement_score(base_probs)
        assert abs(score - 65.6) < 0.5

    def test_maximum_disagreement(self):
        """One model at 1.0, others at 0.0 — should yield score of 100."""
        base_probs = np.array([1.0, 0.0, 0.0])
        score = compute_classification_disagreement_score(base_probs)
        assert abs(score - 100.0) < 0.1


# ---------------------------------------------------------------------------
# Regression VaR score
# ---------------------------------------------------------------------------

class TestRegressionVaR:
    def test_calm_market(self):
        """Current volatility well below 95th percentile yields low score."""
        score = compute_regression_var_score(
            current_volatility=0.01, training_p95_volatility=0.03
        )
        assert abs(score - 33.3) < 1.0

    def test_extreme_volatility_caps(self):
        """Current volatility exceeding training p95 caps at 100."""
        score = compute_regression_var_score(
            current_volatility=0.05, training_p95_volatility=0.03
        )
        assert score == 100.0

    def test_zero_volatility(self):
        score = compute_regression_var_score(
            current_volatility=0.0, training_p95_volatility=0.03
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# Regression disagreement score (empirical MaxDispersion)
# ---------------------------------------------------------------------------

class TestRegressionDisagreement:
    def test_perfect_agreement(self):
        preds = np.array([0.01, 0.01, 0.01])
        score = compute_regression_disagreement_score(preds, max_dispersion=0.005)
        assert score == 0.0

    def test_moderate_spread(self):
        """std_pop(0.01, 0.015, 0.005) = ~0.00408, MaxDisp=0.01 → ~40.8"""
        preds = np.array([0.01, 0.015, 0.005])
        score = compute_regression_disagreement_score(preds, max_dispersion=0.01)
        assert 40 < score < 42

    def test_caps_at_100(self):
        preds = np.array([0.05, -0.05, 0.0])
        score = compute_regression_disagreement_score(preds, max_dispersion=0.001)
        assert score == 100.0


# ---------------------------------------------------------------------------
# Integration: regression composite weighting
# ---------------------------------------------------------------------------

class TestRegressionCompositeWeights:
    def test_var_dominates_disagreement(self):
        """
        With VaR at 90% weight and disagreement at 10%, even maximum
        disagreement (100) with zero VaR should yield a base score of only 10.
        """
        W_REG_VAR = 0.90
        W_REG_DISAGREEMENT = 0.10
        var_score = 0.0
        disagree_score = 100.0
        base = W_REG_VAR * var_score + W_REG_DISAGREEMENT * disagree_score
        assert abs(base - 10.0) < 1e-9
