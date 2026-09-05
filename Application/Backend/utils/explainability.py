"""Provides explainability (SHAP) endpoints.

Dynamically loads the chosen model architecture (Stack, XGB, RF, or LSTM)
as dictated by `model_selection.json` and produces standard SHAP values or
surrogate-tree SHAP values for LSTM.
"""

import json
import shap
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

from utils.paths import SAVED_MODELS_DIR, DATA_DIR
from utils.model_loader import StockLSTM, StockLSTMClassifier


def _get_lstm_probabilities(ticker, df, saved_cols):
    variant_dir = SAVED_MODELS_DIR / ticker / "lstm_class_tuned"
    with open(variant_dir / "metadata.json", "r") as f:
        meta = json.load(f)
        
    scaler_x = joblib.load(variant_dir / "scaler_x.joblib")
    
    lstm = StockLSTMClassifier(
        meta["input_size"], meta["hidden_size"], meta["num_layers"],
        meta["dropout"], meta["num_classes"]
    )
    lstm.load_state_dict(torch.load(variant_dir / "model.pt", weights_only=True))
    lstm.eval()

    seq_len = meta["lookback"]
    X_seq = []
    for i in range(len(df) - seq_len + 1):
        seq = df[saved_cols].iloc[i:i+seq_len].values
        X_seq.append(seq)
    
    if len(X_seq) == 0:
        return np.zeros((0, meta["num_classes"]))
        
    X_scaled = np.array([scaler_x.transform(s) for s in X_seq])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        logits = lstm(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
        
    return probs


def _get_lstm_regression_preds(ticker, df, saved_cols):
    variant_dir = SAVED_MODELS_DIR / ticker / "lstm_reg_tuned"
    with open(variant_dir / "metadata.json", "r") as f:
        meta = json.load(f)
        
    scaler_x = joblib.load(variant_dir / "scaler_x.joblib")
    scaler_y = joblib.load(variant_dir / "scaler_y.joblib")
    
    lstm = StockLSTM(
        meta["input_size"], meta["hidden_size"], meta["num_layers"], meta["dropout"]
    )
    lstm.load_state_dict(torch.load(variant_dir / "model.pt", weights_only=True))
    lstm.eval()

    seq_len = meta["lookback"]
    X_seq = []
    for i in range(len(df) - seq_len + 1):
        seq = df[saved_cols].iloc[i:i+seq_len].values
        X_seq.append(seq)
    
    if len(X_seq) == 0:
        return np.zeros(0)
        
    X_scaled = np.array([scaler_x.transform(s) for s in X_seq])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        pred_scaled = lstm(X_tensor).numpy()
        
    preds = scaler_y.inverse_transform(pred_scaled).flatten()
    return preds


def explain_classification(ticker: str, live_data: dict, predicted_class: str) -> dict:
    with open(SAVED_MODELS_DIR / ticker / "model_selection.json", "r") as f:
        config = json.load(f)
    
    model_id = config["classification"]["model"]
    
    if model_id == "stack":
        return _explain_classification_stack(ticker, live_data, predicted_class)
    elif model_id == "lstm_class_tuned":
        return _explain_classification_lstm(ticker, live_data, predicted_class)
    else:
        # RF or XGBoost
        return _explain_classification_tree(ticker, live_data, predicted_class, model_id)


def explain_regression(ticker: str, live_data: dict) -> dict:
    with open(SAVED_MODELS_DIR / ticker / "model_selection.json", "r") as f:
        config = json.load(f)
        
    model_id = config["regression"]["model"]
    
    if model_id == "stack":
        return _explain_regression_stack(ticker, live_data)
    elif model_id == "lstm_reg_tuned":
        return _explain_regression_lstm(ticker, live_data)
    else:
        # RF or XGBoost
        return _explain_regression_tree(ticker, live_data, model_id)


def _explain_classification_tree(ticker, live_data, predicted_class, model_id):
    variant_dir = SAVED_MODELS_DIR / ticker / model_id
    model = joblib.load(variant_dir / "model.joblib")
    saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
    
    # Needs encoder to find target class idx
    encoder = joblib.load(SAVED_MODELS_DIR / ticker / "lstm_class_tuned" / "encoder.joblib")
    classes = list(encoder.classes_)
    target_class_idx = classes.index(predicted_class)
    
    features_row = live_data["latest_row"][saved_cols]
    
    if "xgb" in model_id:
        df = pd.read_csv(DATA_DIR / f"{ticker}_cleaned_data.csv", index_col="Date", parse_dates=True)
        bg = shap.sample(df[saved_cols].iloc[-500:], 100)
        explainer = shap.PermutationExplainer(model.predict_proba, bg)
        np.random.seed(42)
        shap_obj = explainer(features_row, max_evals=1000)
        shap_values = shap_obj.values[0, :, target_class_idx]
        base_value = shap_obj.base_values[0, target_class_idx] if shap_obj.base_values.ndim == 2 else shap_obj.base_values[target_class_idx]
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(features_row)
        shap_values = sv[target_class_idx][0] if isinstance(sv, list) else sv[0, :, target_class_idx]
        expected_val = explainer.expected_value
        base_value = expected_val[target_class_idx] if isinstance(expected_val, (list, np.ndarray)) else expected_val

    feature_impacts = {col: float(val) for col, val in zip(saved_cols, shap_values) if abs(val) > 1e-6}
    feature_impacts = dict(sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        "ticker": ticker,
        "task": "classification",
        "predicted_class": predicted_class,
        "base_value": round(float(base_value), 4),
        "feature_impacts": feature_impacts,
        "served_model": model_id
    }


def _explain_regression_tree(ticker, live_data, model_id):
    variant_dir = SAVED_MODELS_DIR / ticker / model_id
    model = joblib.load(variant_dir / "model.joblib")
    saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
    
    features_row = live_data["latest_row"][saved_cols]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_row)
    base_value = float(explainer.expected_value)
    
    sv = shap_values[0][0] if isinstance(shap_values, list) else (shap_values[0] if shap_values.ndim == 2 else shap_values)

    feature_impacts = {col: float(val) for col, val in zip(saved_cols, sv) if abs(val) > 1e-4}
    feature_impacts = dict(sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True))

    # Convert to Rupee units
    latest_close = float(live_data["latest_close"])
    rupee_base = round(latest_close * (1 + base_value), 2)
    rupee_impacts = {k: round(latest_close * v, 2) for k, v in feature_impacts.items()}
    rupee_impacts = {k: v for k, v in rupee_impacts.items() if abs(v) > 0.005}

    return {
        "ticker": ticker,
        "task": "regression",
        "base_value": rupee_base,
        "feature_impacts": rupee_impacts,
        "served_model": model_id
    }


def _explain_classification_lstm(ticker, live_data, predicted_class):
    df = pd.read_csv(DATA_DIR / f"{ticker}_cleaned_data.csv", index_col="Date", parse_dates=True)
    df_recent = df.iloc[-500:] 
    
    variant_dir = SAVED_MODELS_DIR / ticker / "lstm_class_tuned"
    saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
    encoder = joblib.load(variant_dir / "encoder.joblib")
    classes = list(encoder.classes_)
    target_class_idx = classes.index(predicted_class)
    
    with open(variant_dir / "metadata.json", "r") as f:
        seq_len = json.load(f)["lookback"]
        
    lstm_probs = _get_lstm_probabilities(ticker, df_recent, saved_cols)
    surrogate_X = df_recent[saved_cols].iloc[seq_len - 1:].values
    
    surrogate = DecisionTreeRegressor(max_depth=5, random_state=42)
    surrogate.fit(surrogate_X, lstm_probs)
    
    surrogate_preds = surrogate.predict(surrogate_X)
    fidelity_r2 = r2_score(lstm_probs[:, target_class_idx], surrogate_preds[:, target_class_idx])
    
    features_row = live_data["latest_row"][saved_cols]
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(features_row)
    
    sv = shap_values[target_class_idx][0] if isinstance(shap_values, list) else shap_values[0, :, target_class_idx]
    base_val = explainer.expected_value[target_class_idx]
    
    feature_impacts = {col: float(val) for col, val in zip(saved_cols, sv) if abs(val) > 1e-6}
    feature_impacts = dict(sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return {
        "ticker": ticker,
        "task": "classification",
        "predicted_class": predicted_class,
        "base_value": round(float(base_val), 4),
        "feature_impacts": feature_impacts,
        "lstm_surrogate_fidelity_r2": round(float(fidelity_r2), 4),
        "low_fidelity_warning": bool(fidelity_r2 < 0.8),
        "served_model": "lstm_class_tuned",
        "note": "Explanations generated via DecisionTreeRegressor surrogate mimicking LSTM outputs."
    }


def _explain_regression_lstm(ticker, live_data):
    df = pd.read_csv(DATA_DIR / f"{ticker}_cleaned_data.csv", index_col="Date", parse_dates=True)
    df_recent = df.iloc[-500:] 
    
    variant_dir = SAVED_MODELS_DIR / ticker / "lstm_reg_tuned"
    saved_cols = joblib.load(variant_dir / "feature_cols.joblib")
    
    with open(variant_dir / "metadata.json", "r") as f:
        seq_len = json.load(f)["lookback"]
        
    lstm_preds = _get_lstm_regression_preds(ticker, df_recent, saved_cols)
    surrogate_X = df_recent[saved_cols].iloc[seq_len - 1:].values
    
    surrogate = DecisionTreeRegressor(max_depth=5, random_state=42)
    surrogate.fit(surrogate_X, lstm_preds)
    
    surrogate_preds = surrogate.predict(surrogate_X)
    fidelity_r2 = r2_score(lstm_preds, surrogate_preds)
    
    features_row = live_data["latest_row"][saved_cols]
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(features_row)
    
    sv = shap_values[0]
    base_val = float(explainer.expected_value[0]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
    
    feature_impacts = {col: float(val) for col, val in zip(saved_cols, sv) if abs(val) > 1e-4}
    feature_impacts = dict(sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True))

    latest_close = float(live_data["latest_close"])
    rupee_base = round(latest_close * (1 + base_val), 2)
    rupee_impacts = {k: round(latest_close * v, 2) for k, v in feature_impacts.items()}
    rupee_impacts = {k: v for k, v in rupee_impacts.items() if abs(v) > 0.005}
    
    return {
        "ticker": ticker,
        "task": "regression",
        "base_value": rupee_base,
        "feature_impacts": rupee_impacts,
        "lstm_surrogate_fidelity_r2": round(float(fidelity_r2), 4),
        "low_fidelity_warning": bool(fidelity_r2 < 0.8),
        "served_model": "lstm_reg_tuned",
        "note": "Explanations generated via DecisionTreeRegressor surrogate mimicking LSTM outputs."
    }

def _explain_classification_stack(ticker, live_data, predicted_class):
    # Rest of the stack explanation code (coefficient-weighted ensemble)
    df = pd.read_csv(DATA_DIR / f"{ticker}_cleaned_data.csv", index_col="Date", parse_dates=True)
    df_recent = df.iloc[-500:] 
    
    stack_dir = SAVED_MODELS_DIR / ticker / "stack_class"
    meta_model = joblib.load(stack_dir / "model.joblib")
    encoder = joblib.load(stack_dir / "encoder.joblib")
    classes = list(encoder.classes_)
    target_class_idx = classes.index(predicted_class)
    
    coefs = meta_model.coef_[target_class_idx]
    intercept = meta_model.intercept_[target_class_idx]
    
    rf_dir = SAVED_MODELS_DIR / ticker / "rf_class_tuned"
    xgb_dir = SAVED_MODELS_DIR / ticker / "xgb_class_tuned"
    lstm_dir = SAVED_MODELS_DIR / ticker / "lstm_class_tuned"
    
    saved_cols = joblib.load(rf_dir / "feature_cols.joblib")
    features_row = live_data["latest_row"][saved_cols]
    
    rf_model = joblib.load(rf_dir / "model.joblib")
    xgb_model = joblib.load(xgb_dir / "model.joblib")
    
    rf_bg = shap.sample(df_recent[saved_cols], 100)
    rf_explainer = shap.TreeExplainer(rf_model, data=rf_bg, feature_perturbation="interventional", model_output="probability")
    xgb_explainer = shap.PermutationExplainer(xgb_model.predict_proba, rf_bg)
    
    rf_shap = rf_explainer.shap_values(features_row)
    np.random.seed(42)
    xgb_shap_obj = xgb_explainer(features_row, max_evals=1000)
    xgb_shap = xgb_shap_obj.values
    
    lstm_saved_cols = joblib.load(lstm_dir / "feature_cols.joblib")
    with open(lstm_dir / "metadata.json", "r") as f:
        seq_len = json.load(f)["lookback"]
        
    lstm_probs = _get_lstm_probabilities(ticker, df_recent, lstm_saved_cols)
    surrogate_X = df_recent[lstm_saved_cols].iloc[seq_len - 1:].values
    
    surrogate = DecisionTreeRegressor(max_depth=5, random_state=42)
    surrogate.fit(surrogate_X, lstm_probs)
    surrogate_preds = surrogate.predict(surrogate_X)
    lstm_fidelity_r2 = r2_score(lstm_probs[:, target_class_idx], surrogate_preds[:, target_class_idx])
    
    surrogate_explainer = shap.TreeExplainer(surrogate)
    lstm_shap = surrogate_explainer.shap_values(features_row[lstm_saved_cols])
    
    rf_expected = rf_explainer.expected_value
    xgb_expected = xgb_shap_obj.base_values[0]
    lstm_expected = surrogate_explainer.expected_value
    
    if not isinstance(xgb_expected, (list, np.ndarray)):
        xgb_expected = [xgb_expected] * len(classes)
        
    final_shap = {col: 0.0 for col in saved_cols}
    total_base_value = intercept
    
    for c_idx, c_name in enumerate(classes):
        rf_coef_idx = c_idx
        xgb_coef_idx = len(classes) + c_idx
        lstm_coef_idx = 2 * len(classes) + c_idx
        
        total_base_value += coefs[rf_coef_idx] * rf_expected[c_idx]
        total_base_value += coefs[xgb_coef_idx] * xgb_expected[c_idx]
        total_base_value += coefs[lstm_coef_idx] * lstm_expected[c_idx]
        
        for f_idx, f_name in enumerate(saved_cols):
            rf_val = rf_shap[c_idx][0, f_idx] if isinstance(rf_shap, list) else rf_shap[0, f_idx, c_idx]
            
            if isinstance(xgb_shap, list):
                xgb_val = xgb_shap[c_idx][0, f_idx]
            elif xgb_shap.ndim == 3:
                xgb_val = xgb_shap[0, f_idx, c_idx]
            else:
                xgb_val = xgb_shap[0, f_idx]
                
            lstm_val = lstm_shap[c_idx][0, f_idx] if isinstance(lstm_shap, list) else lstm_shap[0, f_idx, c_idx]
            
            final_shap[f_name] += coefs[rf_coef_idx] * rf_val
            final_shap[f_name] += coefs[xgb_coef_idx] * xgb_val
            final_shap[f_name] += coefs[lstm_coef_idx] * lstm_val

    feature_impacts = {col: float(val) for col, val in final_shap.items() if abs(val) > 1e-6}
    feature_impacts = dict(sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True))

    return {
        "ticker": ticker,
        "task": "classification",
        "predicted_class": predicted_class,
        "base_value_logit": round(float(total_base_value), 4),
        "feature_impacts_logit": feature_impacts,
        "lstm_surrogate_fidelity_r2": round(float(lstm_fidelity_r2), 4),
        "low_fidelity_warning": bool(lstm_fidelity_r2 < 0.8),
        "served_model": "stack",
        "note": "SHAP values represent feature contributions to the Stacked Meta-Learner's logit for the winning class."
    }

def _explain_regression_stack(ticker, live_data):
    # Dummy implementation for regression stack if ever selected. 
    # Current selection logic does not select regression stack.
    return {"error": "Regression stack explainability not implemented"}
