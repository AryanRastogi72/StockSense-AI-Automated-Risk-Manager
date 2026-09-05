import numpy as np


# ---------------------------------------------------------------------------
# Named constants for composite score weighting
# ---------------------------------------------------------------------------

W_CLASS_ENTROPY = 0.60
W_CLASS_DISAGREEMENT = 0.40

W_REG_VAR = 0.90
W_REG_DISAGREEMENT = 0.10

OUTLIER_FLOOR = 85.0
REGIME_MULTIPLIER_CAP = 2.0

MAX_ENTROPY_3_CLASSES = np.log(3)
SIGMA_MAX_3_MODELS = np.sqrt(2 / 9)

ROLLING_VOLATILITY_WINDOW = 20


# ---------------------------------------------------------------------------
# Composite score formula
# S_final = max(85 * I_outlier, min(100, S_base * R_m))
# ---------------------------------------------------------------------------

def compute_composite_risk(base_score: float, regime_multiplier: float, outlier_gate: bool) -> float:
    amplified = min(100.0, base_score * regime_multiplier)
    floor = OUTLIER_FLOOR if outlier_gate else 0.0
    return max(floor, amplified)


# ---------------------------------------------------------------------------
# Regime multiplier: continuous linear interpolation anchored to D_p1
# ---------------------------------------------------------------------------

def compute_regime_multiplier(decision_score: float, d_p1: float) -> float:
    if decision_score >= 0:
        return 1.0
    return min(REGIME_MULTIPLIER_CAP, 1.0 + abs(decision_score) / abs(d_p1))


# ---------------------------------------------------------------------------
# Outlier gate: fires only when D < D_p1 (strict less-than)
# ---------------------------------------------------------------------------

def compute_outlier_gate(decision_score: float, d_p1: float) -> bool:
    return decision_score < d_p1


# ---------------------------------------------------------------------------
# Classification entropy score (Shannon entropy, normalized to 0-100)
# ---------------------------------------------------------------------------

def compute_classification_entropy_score(class_probabilities: np.ndarray) -> float:
    probs = class_probabilities[class_probabilities > 0]
    if len(probs) == 0:
        return 100.0
    entropy = -np.sum(probs * np.log(probs))
    return min(100.0, (entropy / MAX_ENTROPY_3_CLASSES) * 100.0)


# ---------------------------------------------------------------------------
# Classification disagreement (population std of winning-class probs, ddof=0)
# ---------------------------------------------------------------------------

def compute_classification_disagreement_score(base_model_winning_class_probs: np.ndarray) -> float:
    sigma = np.std(base_model_winning_class_probs, ddof=0)
    return min(100.0, (sigma / SIGMA_MAX_3_MODELS) * 100.0)


# ---------------------------------------------------------------------------
# Regression VaR score (current rolling volatility vs training 95th pctile)
# ---------------------------------------------------------------------------

def compute_regression_var_score(current_volatility: float, training_p95_volatility: float) -> float:
    if training_p95_volatility <= 0:
        return 100.0
    return min(100.0, (current_volatility / training_p95_volatility) * 100.0)


# ---------------------------------------------------------------------------
# Regression disagreement (population std of return predictions, ddof=0)
# ---------------------------------------------------------------------------

def compute_regression_disagreement_score(
    base_model_predictions: np.ndarray, max_dispersion: float
) -> float:
    if max_dispersion <= 0:
        return 100.0
    sigma = np.std(base_model_predictions, ddof=0)
    return min(100.0, (sigma / max_dispersion) * 100.0)


# ---------------------------------------------------------------------------
# Track-level base score aggregation
# ---------------------------------------------------------------------------

def compute_classification_base_risk(
    entropy_score: float, disagreement_score: float
) -> float:
    return W_CLASS_ENTROPY * entropy_score + W_CLASS_DISAGREEMENT * disagreement_score


def compute_regression_base_risk(
    var_score: float, disagreement_score: float
) -> float:
    return W_REG_VAR * var_score + W_REG_DISAGREEMENT * disagreement_score
