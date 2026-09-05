# StockSense AI: Automated Risk Manager

## Overview
Point predictions from machine learning models are too often treated as certainties by downstream systems. The reality is that market forecasting models have varying levels of confidence, model disagreement, and contextual accuracy depending on current market regimes. StockSense AI acts as an **AI Risk Manager**, wrapping predictive forecasting pipelines in a rigorous risk-scoring and explainability layer. It answers not just "what is the prediction?", but "how much should we trust this prediction right now?".

Originally built during an LTTS internship, this project has been significantly extended for the **Razorpay AI Builder Internship 2026 (Track 2: AI Risk Manager)**.

## Architecture
The system supports three assets (`LT.NS`, `TCS.NS`, `RELIANCE.NS`) through a unified pipeline:
1. **Forecasting Engine**: 12 base models per ticker (Random Forest, XGBoost, PyTorch LSTM across regression and classification tracks).
2. **Meta-Learner Stack**: A Logistic Regression / Ridge Regression stacking layer trained purely on Out-Of-Fold (OOF) predictions to eliminate data leakage.
3. **Dynamic Model Routing**: API dynamically serves the Stacked model or the best Base model depending on which objectively clears a minimum generalization margin (0.5%).
4. **Risk-Scoring Layer**:
    * **Regime Detection**: IsolationForest gates extreme anomalous volatility.
    * **Value-at-Risk (VaR)**: Volatility-scaled downside risk metrics.
    * **Disagreement Scoring**: Penalizes predictions where base models heavily diverge.
5. **Explainability Layer**: Dynamic SHAP extraction. Uses native `TreeExplainer` for tree models, a coefficient-weighted ensemble reconstructor for stacked models, and a validated `DecisionTreeRegressor` surrogate proxy for deep learning (LSTM) models.
6. **Portfolio Aggregation**: Computes true cross-asset correlation matrices to produce realistic portfolio-level VaR.

## Tech Stack
* **Core ML**: `scikit-learn`, `xgboost`, `torch`
* **Explainability**: `shap`
* **Backend**: `fastapi`, `uvicorn`
* **Frontend**: `streamlit`, `plotly`
* **Data**: `yfinance`, `pandas`, `pandas-ta-classic`

## Key Results & Model Selection
The pipeline utilizes a strict `0.5%` margin threshold to justify the complexity of serving a stacked meta-learner. Otherwise, it falls back to the single best-performing base model.

### 1. LT.NS
* **Classification**: **Stack** served. Stack Accuracy (37.84%) outperformed XGBoost (36.75%) by +1.09%, clearing the margin.
* **Regression**: **XGBoost Tuned** served. Stack (57.19 RMSE) failed to beat XGBoost (57.16 RMSE).

### 2. TCS.NS
* **Classification**: **XGBoost Tuned** served. XGBoost (34.06%) beat the Stack (30.93%). We verified that OOF probability column ordering was perfectly consistent; this was a genuine failure of the meta-learner to generalize for this specific asset.
* **Regression**: **XGBoost Tuned** served. Stack (46.27 RMSE) beat XGBoost (46.39 RMSE) by only +0.25%, failing to clear the strict 0.5% relative margin required to justify ensemble serving costs.

### 3. RELIANCE.NS
* **Classification**: **LSTM Tuned** served. LSTM (37.06%) beat the Stack (36.65%).
* **Regression**: **LSTM Tuned** served. LSTM (18.12 RMSE) beat the Stack (18.15 RMSE).

## Risk-Scoring Methodology
* **Regime Multiplier**: An IsolationForest detects anomalies based on 20-day return volatility and price dispersion. Normal regimes map to 1.0; anomalous regimes scale risk upwards.
* **Regression Risk (90/10 Split)**: Heavily weighted toward empirical VaR (Z-score 1.645) since base models historically cluster in their predictions, making model disagreement a secondary signal.
* **Classification Risk (60/40 Split)**: Blends prediction Entropy (how uncertain the winning probabilities are) with base-model Disagreement (how wildly the base models disagreed on the winning class).

## Explainability & SHAP
The API exposes `/explain/{ticker}`. Explanations dynamically adapt to the `served_model` configuration:
* Single Tree Models use SHAP `TreeExplainer` / `PermutationExplainer`.
* Stacked Models use a custom reconstructor that computes feature impact by weighting base-model SHAP values by the meta-learner's learned coefficients.
* **LSTM Surrogate Proxy**: Deep learning predictions are explained by training a high-fidelity `DecisionTreeRegressor` surrogate on the most recent 500 trading days. The API performs live fidelity checks. 
  * *Current RELIANCE Surrogate Fidelity*: **0.8246** $R^2$ (Classification) / **0.9637** $R^2$ (Regression).

## Portfolio Risk Aggregation
The `/portfolio/risk` endpoint goes beyond simple additive VaR. It extracts the real 15-year return series for the requested tickers, calculates a live correlation matrix ($\Sigma$), and computes the portfolio variance using $w^T \Sigma w$. This ensures diversification benefits (or correlated liabilities) are mathematically reflected in the final Portfolio VaR.

## Setup & Run Instructions
1. **Install Dependencies**: 
   ```bash
   pip install -r Requirements.txt
   ```
2. **Train Models (Optional - pre-trained artifacts included)**:
   ```bash
   python train_pipeline.py LT
   python train_pipeline.py TCS
   python train_pipeline.py RELIANCE
   ```
3. **Start the Backend API**:
   ```bash
   cd Application/Backend
   uvicorn main:app --reload --port 8000
   ```
4. **Start the Dashboard**:
   ```bash
   cd Application/Frontend
   streamlit run streamlit_app.py
   ```

## API Endpoint Reference
* `GET /risk/{ticker}`: Returns live predictions, `served_model` metadata, and full risk breakdowns (Base Risk, Final Risk, VaR, Disagreement, Regime).
* `GET /explain/{ticker}?task=class|reg`: Returns `base_value`, dictionary of `feature_impacts`, and `low_fidelity_warning` (for LSTM proxies).
* `POST /portfolio/risk`: Accepts a JSON payload of tickers, returning individual risks, the cross-asset `correlation_matrix`, and the global `portfolio_var_95`.

## Testing
The `Models/tests/` directory contains strict validation suites preventing schema desyncs between backend logic and frontend consumption, and verifying mathematical parity in regime scoring and explainability. Run via:
```bash
pytest Models/tests/
```

## Known Limitations & Future Work
* **Fixed Risk Weightings**: The 90/10 (Regression) and 60/40 (Classification) VaR/Disagreement weightings are currently fixed globally due to time constraints, though they should ideally be calibrated per-ticker based on the historical variance of that specific asset's base models.
* **LSTM Surrogate Limitations**: The DecisionTree surrogate is an approximation of the LSTM's non-linear decision boundary. While current fidelity is high (>0.80), sudden regime shifts could decouple the surrogate.
* **TCS Classification Meta-Learner**: The stacking layer consistently underperformed baseline models for TCS classification. While bugs were ruled out, the underlying data distribution cause remains an open research item.
* **Asset Universe**: Hardcoded to 3 specific Indian equities.

## Credits
Project originally authored during an **LTTS Internship**, actively extended and refactored for the **Razorpay AI Builder Internship 2026 (Track 2)**.
