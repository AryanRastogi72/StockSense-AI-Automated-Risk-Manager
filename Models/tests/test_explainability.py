import pytest
import numpy as np
import pandas as pd
import joblib

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Application" / "Backend"))

from utils.explainability import explain_classification, explain_regression
from utils.live_features import get_live_features
from utils.model_loader import predict_tree

@pytest.fixture(scope="module")
def setup_data():
    ticker = "LT"
    live_data = get_live_features(ticker)
    
    stack_dir = PROJECT_ROOT / "saved_models" / ticker / "stack_class"
    meta_model = joblib.load(stack_dir / "model.joblib")
    encoder = joblib.load(stack_dir / "encoder.joblib")
    classes = list(encoder.classes_)
    
    return ticker, live_data, meta_model, encoder, classes

def test_explain_regression_sum(setup_data):
    """Verify that regression SHAP impacts sum to the predicted price difference."""
    ticker, live_data, _, _, _ = setup_data
    
    pred_res = predict_tree(ticker, "xgb", "reg", "tuned", live_data["latest_row"])
    real_predicted_close = pred_res["predicted_close"]
    
    expl = explain_regression(ticker, live_data)
    base_val = expl["base_value"]
    impacts = expl["feature_impacts"]
    
    sum_impacts = sum(impacts.values())
    reconstructed = base_val + sum_impacts
    
    # We tolerate a tiny difference due to round(base_value, 2) in the UI code
    # and dropping impacts < 1e-4.
    difference = abs(real_predicted_close - reconstructed)
    assert difference < 0.01, f"Regression SHAP sum reconstructed {reconstructed} but predict_tree returned {real_predicted_close}"

def test_explain_classification_logit_sum(setup_data):
    """Verify that the logit reconstruction works for normal classification SHAP."""
    ticker, live_data, meta_model, encoder, classes = setup_data
    
    target_class = "Up"
    target_idx = classes.index(target_class)
    
    explanation = explain_classification(ticker, live_data, target_class)
    base_val = explanation["base_value_logit"]
    shap_sum = sum(explanation["feature_impacts_logit"].values())
    reconstructed_logit = base_val + shap_sum
    
    from utils.risk_utils import _get_base_classification_probs
    real_base_probs = _get_base_classification_probs(ticker, live_data["latest_row"], live_data["sequence"])
    
    feature_vec = []
    for c_name in classes:
        for algo in ["rf", "xgb", "lstm"]:
            idx = classes.index(c_name)
            feature_vec.append(real_base_probs[algo][idx])
            
    real_logit = meta_model.decision_function(np.array(feature_vec).reshape(1, -1))[0]
    if real_logit.ndim == 1:
        real_target_logit = real_logit[target_idx]
    else:
        real_target_logit = real_logit
        
    difference = abs(reconstructed_logit - real_target_logit)
    assert difference < 1.0, f"SHAP logit sum ({reconstructed_logit}) diverges too far from actual logit ({real_target_logit})"

def test_explain_low_fidelity_adversarial(setup_data):
    """Verify surrogate error expands significantly on an adversarial low-fidelity day, proving the real LSTM drives the stack."""
    ticker, live_data_dummy, meta_model, encoder, classes = setup_data
    
    # Use May 27, 2026, which we identified earlier as having the worst surrogate discrepancy (error ~0.198)
    DATA_DIR = PROJECT_ROOT / "Supporting Data"
    df = pd.read_csv(DATA_DIR / f"{ticker}_cleaned_data.csv", index_col="Date", parse_dates=True)
    
    worst_date = pd.to_datetime("2026-05-27")
    end_loc = df.index.get_loc(worst_date)
    history = df.iloc[end_loc - 60 : end_loc + 1]
    
    seq_len = 30 # from lstm metadata
    latest_row = history.iloc[[-1]]
    seq_df = history.tail(seq_len)
    
    live_data = {
        "latest_row": latest_row,
        "sequence": seq_df,
        "last_date": str(worst_date.date()),
        "raw_df": history
    }
    
    target_class = "Up"
    target_idx = classes.index(target_class)
    
    expl = explain_classification(ticker, live_data, target_class)
    reconstructed_logit = expl["base_value_logit"] + sum(expl["feature_impacts_logit"].values())
    
    from utils.risk_utils import _get_base_classification_probs
    real_base_probs = _get_base_classification_probs(ticker, latest_row, seq_df)
    
    feature_vec = []
    for c_name in classes:
        for algo in ["rf", "xgb", "lstm"]:
            idx = classes.index(c_name)
            feature_vec.append(real_base_probs[algo][idx])
            
    real_logit = meta_model.decision_function(np.array(feature_vec).reshape(1, -1))[0]
    if real_logit.ndim == 1:
        real_target_logit = real_logit[target_idx]
    else:
        real_target_logit = real_logit
        
    diff = abs(reconstructed_logit - real_target_logit)
    # Baseline difference is ~0.004. On the adversarial day, it should be > 0.05
    assert diff > 0.05, f"Expected large surrogate residual on low-fidelity day, but got {diff}"