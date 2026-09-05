"""Risk scoring utilities for the FastAPI backend.

Loads the persisted calibration artifacts and provides functions to compute
single-ticker and portfolio-level risk scores alongside predictions.
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path

from utils.paths import PROJECT_ROOT, SAVED_MODELS_DIR, MODELS_DIR

sys.path.insert(0, str(MODELS_DIR))

from risk_scorer import (
    compute_classification_entropy_score,
    compute_classification_disagreement_score,
    compute_classification_base_risk,
    compute_regression_var_score,
    compute_regression_disagreement_score,
    compute_regression_base_risk,
    compute_composite_risk,
    compute_regime_multiplier,
    compute_outlier_gate,
    ROLLING_VOLATILITY_WINDOW,
)
from regime_detector import REGIME_FEATURES, extract_regime_features


from utils.model_loader import StockLSTM, StockLSTMClassifier


def load_risk_calibration(ticker: str) -> dict:
    """Loads the persisted Isolation Forest and calibration constants."""
    from regime_detector import load_calibration
    
    cal_dir = SAVED_MODELS_DIR / ticker / "risk_calibration"
    if not cal_dir.exists():
        raise FileNotFoundError(f"Risk calibration artifacts not found at {cal_dir}")
        
    return load_calibration(cal_dir)


def _score_regime(regime_features_row: np.ndarray, calibration: dict) -> dict:
    from regime_detector import score_regime
    
    raw = score_regime(
        regime_features_row,
        calibration["iso_forest"],
        calibration["regime_scaler"],
        calibration["d_p1"]
    )
    
    return {
        "decision_score": round(raw["decision_score"], 6),
        "regime_multiplier": round(raw["regime_multiplier"], 4),
        "outlier_gate": raw["outlier_gate"],
    }


def _get_base_regression_predictions(ticker: str, features_row, sequence_df) -> np.ndarray:
    """Gets regression predictions from all 3 tuned base models."""
    preds = []
    for algo in ["rf", "xgb"]:
        variant_dir = SAVED_MODELS_DIR / ticker / f"{algo}_reg_tuned"
        model = joblib.load(variant_dir / "model.joblib")
        saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
        pred = model.predict(features_row[saved_cols])[0]
        preds.append(float(pred))

    variant_dir = SAVED_MODELS_DIR / ticker / "lstm_reg_tuned"
    with open(variant_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
    scaler_x = joblib.load(variant_dir / "scaler_x.joblib")
    scaler_y = joblib.load(variant_dir / "scaler_y.joblib")

    seq = sequence_df[saved_cols]
    seq_scaled = scaler_x.transform(seq.values)
    X_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

    lstm = StockLSTM(meta["input_size"], meta["hidden_size"], meta["num_layers"], meta["dropout"])
    lstm.load_state_dict(torch.load(variant_dir / "model.pt", weights_only=True))
    lstm.eval()
    with torch.no_grad():
        pred_scaled = lstm(X_tensor).numpy().flatten()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    preds.append(float(pred))

    return np.array(preds)


def _get_base_classification_probs(ticker: str, features_row, sequence_df) -> dict:
    """Gets class probabilities from all 3 tuned classification base models."""
    result = {}
    for algo in ["rf", "xgb"]:
        variant_dir = SAVED_MODELS_DIR / ticker / f"{algo}_class_tuned"
        model = joblib.load(variant_dir / "model.joblib")
        saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
        probs = model.predict_proba(features_row[saved_cols])[0]
        result[algo] = probs

    variant_dir = SAVED_MODELS_DIR / ticker / "lstm_class_tuned"
    with open(variant_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
    scaler_x = joblib.load(variant_dir / "scaler_x.joblib")

    seq = sequence_df[saved_cols]
    seq_scaled = scaler_x.transform(seq.values)
    X_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

    lstm = StockLSTMClassifier(
        meta["input_size"], meta["hidden_size"], meta["num_layers"],
        meta["dropout"], meta["num_classes"]
    )
    lstm.load_state_dict(torch.load(variant_dir / "model.pt", weights_only=True))
    lstm.eval()
    with torch.no_grad():
        logits = lstm(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    result["lstm"] = probs

    return result


def _get_stack_classification_prediction(ticker: str, base_probs: dict) -> tuple:
    """Runs the stacked classification meta-learner on base model probabilities."""
    stack_dir = SAVED_MODELS_DIR / ticker / "stack_class"
    meta_model = joblib.load(stack_dir / "model.joblib")
    encoder = joblib.load(stack_dir / "encoder.joblib")
    classes = encoder.classes_

    feature_vec = []
    for class_name in classes:
        for algo in ["rf", "xgb", "lstm"]:
            idx = list(classes).index(class_name)
            feature_vec.append(base_probs[algo][idx])

    feature_array = np.array(feature_vec).reshape(1, -1)
    stack_probs = meta_model.predict_proba(feature_array)[0]
    predicted_class_idx = np.argmax(stack_probs)
    predicted_class = encoder.inverse_transform([meta_model.classes_[predicted_class_idx]])[0]

    return stack_probs, predicted_class, classes


def compute_risk_for_ticker(ticker: str, live_data: dict) -> dict:
    """Computes the full risk breakdown for a single ticker using live data."""
    calibration = load_risk_calibration(ticker)
    
    with open(SAVED_MODELS_DIR / ticker / "model_selection.json", "r") as f:
        selection_config = json.load(f)

    raw_df = live_data["raw_df"]
    regime_features_array = extract_regime_features(raw_df)
    regime_row = regime_features_array[-1]

    regime = _score_regime(regime_row, calibration)

    price_history = raw_df["Close"]
    features_row = live_data["latest_row"]
    sequence_df = live_data["sequence"]

    result = {
        "ticker": ticker,
        "regime": regime,
    }

    # --- Classification Risk ---
    base_class_probs = _get_base_classification_probs(ticker, features_row, sequence_df)
    stack_probs, stack_winning_class, classes = _get_stack_classification_prediction(
        ticker, base_class_probs
    )

    class_model_id = selection_config["classification"]["model"]
    if class_model_id == "stack":
        served_class_model = "Stacked Model"
        winning_class = stack_winning_class
        final_probs = stack_probs
    else:
        algo = class_model_id.split("_")[0]
        served_class_model = f"{algo.upper()} Tuned"
        final_probs = base_class_probs[algo]
        winning_class = classes[np.argmax(final_probs)]

    entropy_score = compute_classification_entropy_score(final_probs)

    winning_class_idx = list(classes).index(winning_class)
    winning_probs_across_models = np.array([
        base_class_probs[model][winning_class_idx]
        for model in ["rf", "xgb", "lstm"]
    ])
    disagree_score = compute_classification_disagreement_score(winning_probs_across_models)

    # Use globally fixed weighting logic (0.6 / 0.4) handled by compute_classification_base_risk
    class_base = compute_classification_base_risk(entropy_score, disagree_score)
    class_final = compute_composite_risk(
        class_base, regime["regime_multiplier"], regime["outlier_gate"]
    )

    result["classification"] = {
        "served_model": served_class_model,
        "predicted_direction": winning_class,
        "stack_probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, final_probs)},
        "entropy_score": round(entropy_score, 2),
        "disagreement_score": round(disagree_score, 2),
        "base_risk": round(class_base, 2),
        "final_risk": round(class_final, 2),
    }

    # --- Regression Risk ---
    reg_preds = _get_base_regression_predictions(ticker, features_row, sequence_df)
    # reg_preds is [rf_pred, xgb_pred, lstm_pred]
    reg_model_id = selection_config["regression"]["model"]
    
    if reg_model_id == "stack":
        served_reg_model = "Stacked Model"
        # Predict using stack (Ridge)
        stack_dir = SAVED_MODELS_DIR / ticker / "stack_reg"
        meta_model = joblib.load(stack_dir / "model.joblib")
        # reg_preds is an array of 3. But stack expects features [RF, XGB, LSTM]
        stack_pred = meta_model.predict(reg_preds.reshape(1, -1))[0]
        predicted_pct = stack_pred
    else:
        algo = reg_model_id.split("_")[0]
        served_reg_model = f"{algo.upper()} Tuned"
        algo_idx = {"rf": 0, "xgb": 1, "lstm": 2}[algo]
        predicted_pct = reg_preds[algo_idx]
        
    latest_close = float(features_row["Close"].values[0])

    daily_returns = price_history.pct_change().dropna()
    if len(daily_returns) >= ROLLING_VOLATILITY_WINDOW:
        current_vol = float(daily_returns.iloc[-ROLLING_VOLATILITY_WINDOW:].std())
    elif len(daily_returns) > 1:
        current_vol = float(daily_returns.std())
    else:
        # Guard: std is mathematically undefined for 0 or 1 observations.
        current_vol = 0.0

    var_score = compute_regression_var_score(
        current_vol, calibration["training_p95_volatility"]
    )
    disagree_score_reg = compute_regression_disagreement_score(
        reg_preds, calibration["max_dispersion"]
    )

    # Use globally fixed weighting logic (0.9 / 0.1) handled by compute_regression_base_risk
    reg_base = compute_regression_base_risk(var_score, disagree_score_reg)
    reg_final = compute_composite_risk(
        reg_base, regime["regime_multiplier"], regime["outlier_gate"]
    )

    result["regression"] = {
        "served_model": served_reg_model,
        "predicted_close": round(latest_close * (1 + predicted_pct), 2),
        "predicted_close_xgb": round(latest_close * (1 + reg_preds[1]), 2), # Keep for backwards compat in tests
        "base_model_predictions": {
            "rf": round(latest_close * (1 + reg_preds[0]), 2),
            "xgb": round(latest_close * (1 + reg_preds[1]), 2),
            "lstm": round(latest_close * (1 + reg_preds[2]), 2),
        },
        "var_score": round(var_score, 2),
        "disagreement_score": round(disagree_score_reg, 2),
        "base_risk": round(reg_base, 2),
        "final_risk": round(reg_final, 2),
        "current_volatility": round(current_vol, 6),
        "low_confidence": len(daily_returns) < ROLLING_VOLATILITY_WINDOW,
    }

    return result


def compute_portfolio_risk(tickers: list, live_data_map: dict) -> dict:
    """Computes portfolio-level risk including correlation-adjusted VaR."""
    ticker_results = {}
    return_series = {}

    for ticker in tickers:
        if ticker not in live_data_map:
            continue
        ticker_results[ticker] = compute_risk_for_ticker(ticker, live_data_map[ticker])
        raw_df = live_data_map[ticker]["raw_df"]
        returns = raw_df["Close"].pct_change().dropna()
        return_series[ticker] = returns

    available_tickers = list(return_series.keys())
    if len(available_tickers) < 2:
        return {
            "tickers": available_tickers,
            "individual_risks": ticker_results,
            "portfolio_var_95": None,
            "portfolio_volatility": None,
            "correlation_matrix": None,
            "weights": None,
            "note": "Need at least 2 tickers for portfolio VaR",
        }

    min_len = min(len(s) for s in return_series.values())
    aligned_returns = pd.DataFrame({
        t: return_series[t].iloc[-min_len:].values for t in available_tickers
    })

    corr_matrix = aligned_returns.corr()
    cov_matrix = aligned_returns.cov()

    n = len(available_tickers)
    weights = np.array([1.0 / n] * n)

    portfolio_vol = float(np.sqrt(weights @ cov_matrix.values @ weights))
    z_95 = 1.645
    portfolio_var_95 = round(portfolio_vol * z_95, 6)

    return {
        "tickers": available_tickers,
        "individual_risks": ticker_results,
        "portfolio_var_95": portfolio_var_95,
        "portfolio_volatility": round(portfolio_vol, 6),
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "weights": {t: round(w, 4) for t, w in zip(available_tickers, weights)},
    }
