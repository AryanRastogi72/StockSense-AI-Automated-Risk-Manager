import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import joblib
from pathlib import Path

SPLIT_RATIO = 0.8
SEED = 42
import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"

PARAM_GRID = {
    "n_estimators": [50, 100, 150, 200, 250],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [3, 5, 7, 10],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_alpha": [0, 0.3, 0.6, 1.0],
    "reg_lambda": [1, 2, 3, 5],
}

if __name__ == '__main__':
    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)

    feature_cols = [c for c in df.columns if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]]
    X = df[feature_cols]
    y = df["Target_Pct_Change"]

    split_index = int(len(df) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    today_close = X_test["Close"].values
    y_true_price = df["Target_Close_Next"].iloc[split_index:].values
    y_naive_price = today_close

    baseline_model = XGBRegressor(n_estimators=100, random_state=SEED)
    baseline_model.fit(X_train, y_train)
    baseline_pred_price = today_close * (1 + baseline_model.predict(X_test))
    rmse_baseline = np.sqrt(mean_squared_error(y_true_price, baseline_pred_price))
    r2_baseline = r2_score(y_true_price, baseline_pred_price)

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        XGBRegressor(random_state=SEED),
        PARAM_GRID,
        n_iter=60,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=SEED,
    )
    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    tuned_pred_price = today_close * (1 + tuned_model.predict(X_test))
    rmse_tuned = np.sqrt(mean_squared_error(y_true_price, tuned_pred_price))
    r2_tuned = r2_score(y_true_price, tuned_pred_price)

    rmse_naive = np.sqrt(mean_squared_error(y_true_price, y_naive_price))
    r2_naive = r2_score(y_true_price, y_naive_price)

    # Baseline metrics saved to JSON
    baseline_metrics = {
        "rmse_naive": float(rmse_naive),
        "rmse_baseline": float(rmse_baseline),
        "r2_naive": float(r2_naive),
        "r2_baseline": float(r2_baseline)
    }
    from pathlib import Path
    save_path = Path("saved_models") / TICKER / "xg_reg_tuned"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"{TICKER}_XG_baseline_comparison.json", "w") as f:
        json.dump(baseline_metrics, f, indent=4)

    save_dir = Path("saved_models") / TICKER / "xgb_reg_tuned"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, save_dir / "model.joblib")
    joblib.dump(list(feature_cols), save_dir / "feature_cols.joblib")

    plot_window = 126
    plot_test = X_test.iloc[-plot_window:]
    plot_pred = tuned_pred_price[-plot_window:]
    plot_naive = y_naive_price[-plot_window:]

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=plot_test.index, open=plot_test["Open"], high=plot_test["High"],
        low=plot_test["Low"], close=plot_test["Close"], name="Actual OHLC",
        increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_test.index, y=plot_pred, mode="lines", name="Tuned XGBoost Predicted",
        line=dict(color="#FF9800", width=2, dash="dash"),
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_test.index, y=plot_naive, mode="lines", name="Naive Baseline",
        line=dict(color="#4CAF50", width=2, dash="dot"),
    ))
    price_fig.update_layout(
        title="Tuned XGBoost: Predictions vs Actual Price Action (Last 6 Months)",
        xaxis_title="Date", yaxis_title="Price (Rs)",
        template="plotly_white", xaxis_rangeslider_visible=False,
    )
    price_fig.write_html(f"plot_xgb_regression_tuned_candlestick_{TICKER}.html")

    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=y_true_price, y=tuned_pred_price, mode="markers", name="Predictions",
        marker=dict(color="#FF9800", size=5, opacity=0.6),
    ))
    line_min = min(y_true_price.min(), tuned_pred_price.min())
    line_max = max(y_true_price.max(), tuned_pred_price.max())
    scatter_fig.add_trace(go.Scatter(
        x=[line_min, line_max], y=[line_min, line_max], mode="lines",
        name="Perfect Prediction", line=dict(color="gray", dash="dash"),
    ))
    scatter_fig.update_layout(
        title="Tuned XGBoost: Predicted vs Actual",
        xaxis_title="Actual Price (Rs)", yaxis_title="Predicted Price (Rs)",
        template="plotly_white",
    )
    scatter_fig.write_html(f"plot_xgb_regression_tuned_scatter_{TICKER}.html")

