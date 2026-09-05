"""Trains and persists all risk-layer calibration artifacts for a ticker.

Artifacts saved to saved_models/{TICKER}/risk_calibration/:
- iso_forest.joblib    : Trained Isolation Forest model
- calibration.json     : D_p1, training_p95_volatility, MaxDispersion constants
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from regime_detector import compute_training_calibration, save_calibration

import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"


def main():
    calibration = compute_training_calibration(TICKER, "Supporting Data")

    save_dir = Path("saved_models") / TICKER / "risk_calibration"
    save_calibration(calibration, save_dir)

    print(f"Risk calibration artifacts saved to {save_dir}/")
    print(f"  D_p1:                   {calibration['d_p1']:.6f}")
    print(f"  Training P95 Volatility:{calibration['training_p95_volatility']:.6f}")
    print(f"  Max Dispersion (OOF):   {calibration['max_dispersion']:.6f}")


if __name__ == "__main__":
    main()
