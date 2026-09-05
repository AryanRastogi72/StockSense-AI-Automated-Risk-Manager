import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from risk_scorer import (
    W_CLASS_ENTROPY,
    W_CLASS_DISAGREEMENT,
    W_REG_VAR,
    W_REG_DISAGREEMENT,
    ROLLING_VOLATILITY_WINDOW,
    compute_regime_multiplier,
    compute_outlier_gate,
    compute_composite_risk,
    compute_classification_entropy_score,
    compute_classification_disagreement_score,
    compute_classification_base_risk,
    compute_regression_var_score,
    compute_regression_disagreement_score,
    compute_regression_base_risk,
)

SPLIT_RATIO = 0.8
SEED = 42
ISOLATION_FOREST_CONTAMINATION = 0.05

# Scale-invariant features for regime detection.
# Absolute-price features (Close, Open, SMA_20, BB bands, lags, OBV, Volatility_20)
# cause the detector to flag normal price-level drift as anomalous.
# Volatility_20 is Close.rolling(20).std() — absolute-price-denominated — so it's
# replaced by Return_Vol_20 (Daily_Return.rolling(20).std()), computed on the fly.
# Day_of_Week is dropped: cyclical encoding made no measurable difference,
# and day-of-week carries no regime-shift signal.
REGIME_FEATURES = [
    "Daily_Return",
    "Return_Vol_20",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "ROC_10",
    "Log_Volume",
]

RETURN_VOL_WINDOW = 20


def train_regime_detector(X_train: np.ndarray) -> tuple:
    """Trains an Isolation Forest on standardized scale-invariant features
    and returns the model, scaler, and 1st-percentile decision threshold (D_p1).

    Only scale-invariant features (returns, volatility, RSI, MACD, etc.) are
    used so the detector identifies distributional pattern shifts rather than
    secular price-level drift."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    iso_forest = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=SEED,
        n_estimators=200,
    )
    iso_forest.fit(X_scaled)

    train_scores = iso_forest.decision_function(X_scaled)
    d_p1 = float(np.percentile(train_scores, 1))

    return iso_forest, scaler, d_p1


def extract_regime_features(df: pd.DataFrame) -> np.ndarray:
    """Extracts the scale-invariant feature subset used by the regime detector.
    Computes Return_Vol_20 on the fly if it is not already in the DataFrame.
    Drops any rows with NaN values resulting from the rolling window computation.
    """
    df_copy = df.copy()
    if "Return_Vol_20" not in df_copy.columns and "Daily_Return" in df_copy.columns:
        df_copy["Return_Vol_20"] = df_copy["Daily_Return"].rolling(RETURN_VOL_WINDOW).std()
    
    available = [f for f in REGIME_FEATURES if f in df_copy.columns]
    # Drop rows where rolling std resulted in NaNs
    df_clean = df_copy[available].dropna()
    return df_clean.values


def compute_training_calibration(ticker: str, data_dir: str = "Supporting Data") -> dict:
    """Computes and returns training-time calibration constants needed for
    risk score normalization at inference time."""
    data_path = Path(__file__).resolve().parent.parent / "Supporting Data" / f"{ticker}_cleaned_data.csv"
    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    feature_cols = [
        c for c in df.columns
        if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]
    ]

    split_index = int(len(df) * SPLIT_RATIO)
    X_regime_train = extract_regime_features(df.iloc[:split_index])

    iso_forest, regime_scaler, d_p1 = train_regime_detector(X_regime_train)

    daily_returns = df["Close"].pct_change().dropna().iloc[:split_index]
    rolling_vol = daily_returns.rolling(window=ROLLING_VOLATILITY_WINDOW).std().dropna()
    training_p95_volatility = float(np.percentile(rolling_vol.values, 95))

    oof_reg_path = Path(f"oof_predictions_reg_{ticker}.csv")
    if oof_reg_path.exists():
        oof_df = pd.read_csv(oof_reg_path, index_col="Date", parse_dates=True)
        per_row_std = oof_df[["RF_Pred", "XGB_Pred", "LSTM_Pred"]].apply(
            lambda row: np.std(row.values, ddof=0), axis=1
        )
        max_dispersion = float(np.percentile(per_row_std.values, 95))
    else:
        max_dispersion = 0.015

    return {
        "iso_forest": iso_forest,
        "regime_scaler": regime_scaler,
        "d_p1": d_p1,
        "training_p95_volatility": training_p95_volatility,
        "max_dispersion": max_dispersion,
        "feature_cols": feature_cols,
        "regime_features": [f for f in REGIME_FEATURES if f in df.columns],
    }


def save_calibration(calibration: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibration["iso_forest"], save_dir / "iso_forest.joblib")
    joblib.dump(calibration["regime_scaler"], save_dir / "regime_scaler.joblib")
    constants = {
        "d_p1": calibration["d_p1"],
        "training_p95_volatility": calibration["training_p95_volatility"],
        "max_dispersion": calibration["max_dispersion"],
        "regime_features": calibration.get("regime_features", REGIME_FEATURES),
    }
    with open(save_dir / "calibration.json", "w") as f:
        json.dump(constants, f, indent=2)


def load_calibration(save_dir: Path) -> dict:
    iso_forest = joblib.load(save_dir / "iso_forest.joblib")
    regime_scaler = joblib.load(save_dir / "regime_scaler.joblib")
    with open(save_dir / "calibration.json", "r") as f:
        constants = json.load(f)
    return {
        "iso_forest": iso_forest,
        "regime_scaler": regime_scaler,
        "d_p1": constants["d_p1"],
        "training_p95_volatility": constants["training_p95_volatility"],
        "max_dispersion": constants["max_dispersion"],
        "regime_features": constants.get("regime_features", REGIME_FEATURES),
    }


def score_regime(
    regime_features_row: np.ndarray, iso_forest, regime_scaler, d_p1: float
) -> dict:
    """Evaluates the regime detector on a single row of scale-invariant features
    and returns the multiplier, gate status, and raw decision score."""
    if regime_features_row.ndim == 1:
        regime_features_row = regime_features_row.reshape(1, -1)
    features_scaled = regime_scaler.transform(regime_features_row)
    decision_score = float(iso_forest.decision_function(features_scaled)[0])
    rm = compute_regime_multiplier(decision_score, d_p1)
    outlier = compute_outlier_gate(decision_score, d_p1)
    return {
        "decision_score": decision_score,
        "regime_multiplier": rm,
        "outlier_gate": outlier,
    }


def compute_single_ticker_risk(
    ticker: str,
    calibration: dict,
    live_regime_features: np.ndarray,
    price_history: pd.Series,
    classification_stack_probs: np.ndarray = None,
    base_class_probs: dict = None,
    winning_class_label: str = None,
    regression_base_preds: np.ndarray = None,
) -> dict:
    """Computes the full risk breakdown for a single ticker."""
    regime = score_regime(
        live_regime_features,
        calibration["iso_forest"],
        calibration["regime_scaler"],
        calibration["d_p1"],
    )

    result = {
        "ticker": ticker,
        "regime": regime,
    }

    if classification_stack_probs is not None and base_class_probs is not None:
        entropy_score = compute_classification_entropy_score(classification_stack_probs)

        winning_idx = list(base_class_probs.keys()).index(winning_class_label)
        winning_probs = np.array([
            base_class_probs[model][winning_idx]
            for model in base_class_probs
        ])
        disagree_score = compute_classification_disagreement_score(winning_probs)

        class_base = compute_classification_base_risk(entropy_score, disagree_score)
        class_final = compute_composite_risk(
            class_base, regime["regime_multiplier"], regime["outlier_gate"]
        )

        result["classification"] = {
            "entropy_score": round(entropy_score, 2),
            "disagreement_score": round(disagree_score, 2),
            "base_risk": round(class_base, 2),
            "final_risk": round(class_final, 2),
        }

    if regression_base_preds is not None:
        daily_returns = price_history.pct_change().dropna()
        if len(daily_returns) >= ROLLING_VOLATILITY_WINDOW:
            current_vol = float(daily_returns.iloc[-ROLLING_VOLATILITY_WINDOW:].std())
        else:
            current_vol = float(daily_returns.std()) if len(daily_returns) > 1 else 0.0

        var_score = compute_regression_var_score(
            current_vol, calibration["training_p95_volatility"]
        )
        disagree_score = compute_regression_disagreement_score(
            regression_base_preds, calibration["max_dispersion"]
        )

        reg_base = compute_regression_base_risk(var_score, disagree_score)
        reg_final = compute_composite_risk(
            reg_base, regime["regime_multiplier"], regime["outlier_gate"]
        )

        result["regression"] = {
            "var_score": round(var_score, 2),
            "disagreement_score": round(disagree_score, 2),
            "base_risk": round(reg_base, 2),
            "final_risk": round(reg_final, 2),
            "current_volatility": round(current_vol, 6),
            "low_confidence": len(daily_returns) < ROLLING_VOLATILITY_WINDOW,
        }

    return result
