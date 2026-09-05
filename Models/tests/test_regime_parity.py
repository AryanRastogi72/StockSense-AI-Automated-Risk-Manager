import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Application" / "Backend"))
sys.path.insert(0, str(PROJECT_ROOT / "Models"))

import numpy as np
from utils.risk_utils import _score_regime as old_score_regime
from regime_detector import score_regime
from utils.risk_utils import load_risk_calibration

def test_regime_scoring_parity():
    """Verify that score_regime and old _score_regime produce identical results modulo rounding."""
    ticker = "LT"
    calibration = load_risk_calibration(ticker)
    
    regime_features_row = np.array([0.01, 0.02, 50, 1.0, 0.5, 0.05, 15.0])
    
    old_res = old_score_regime(regime_features_row, calibration)
    
    new_raw = score_regime(
        regime_features_row, 
        calibration["iso_forest"], 
        calibration["regime_scaler"], 
        calibration["d_p1"]
    )
    
    new_res = {
        "decision_score": round(new_raw["decision_score"], 6),
        "regime_multiplier": round(new_raw["regime_multiplier"], 4),
        "outlier_gate": new_raw["outlier_gate"]
    }
    
    assert old_res["decision_score"] == new_res["decision_score"]
    assert old_res["regime_multiplier"] == new_res["regime_multiplier"]
    assert old_res["outlier_gate"] == new_res["outlier_gate"]
