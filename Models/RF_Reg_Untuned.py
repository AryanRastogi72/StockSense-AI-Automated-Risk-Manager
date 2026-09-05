import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

SPLIT_RATIO = 0.8
SEED = 42
import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"

if __name__ == '__main__':
    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)

    feature_cols = [c for c in df.columns if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]]
    X = df[feature_cols]
    y = df["Target_Pct_Change"]

    split_index = int(len(df) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    pred_pct_change = model.predict(X_test)

    today_close = X_test["Close"].values
    y_pred_price = today_close * (1 + pred_pct_change)
    y_true_price = df["Target_Close_Next"].iloc[split_index:].values

    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    r2 = r2_score(y_true_price, y_pred_price)

    y_naive_price = today_close
    rmse_naive = np.sqrt(mean_squared_error(y_true_price, y_naive_price))
    r2_naive = r2_score(y_true_price, y_naive_price)

    save_dir = Path("saved_models") / TICKER / "rf_reg_baseline"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_dir / "model.joblib")
    joblib.dump(list(feature_cols), save_dir / "feature_cols.joblib")

    plot_window = 126
    plot_test = X_test.iloc[-plot_window:]
    plot_pred = y_pred_price[-plot_window:]
    plot_naive = y_naive_price[-plot_window:]

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=plot_test.index, open=plot_test["Open"], high=plot_test["High"],
        low=plot_test["Low"], close=plot_test["Close"], name="Actual OHLC",
        increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_test.index, y=plot_pred, mode="lines", name="RF Predicted",
        line=dict(color="#FF5722", width=2, dash="dash"),
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_test.index, y=plot_naive, mode="lines", name="Naive Baseline",
        line=dict(color="#4CAF50", width=2, dash="dot"),
    ))
    price_fig.update_layout(
        title="Random Forest: Predictions vs Actual Price Action (Last 6 Months)",
        xaxis_title="Date", yaxis_title="Price (Rs)",
        template="plotly_white", xaxis_rangeslider_visible=False,
    )
    price_fig.write_html(f"plot_regression_candlestick_{TICKER}.html")

    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=y_true_price, y=y_pred_price, mode="markers", name="Predictions",
        marker=dict(color="#FF5722", size=5, opacity=0.6),
    ))
    line_min = min(y_true_price.min(), y_pred_price.min())
    line_max = max(y_true_price.max(), y_pred_price.max())
    scatter_fig.add_trace(go.Scatter(
        x=[line_min, line_max], y=[line_min, line_max], mode="lines",
        name="Perfect Prediction", line=dict(color="gray", dash="dash"),
    ))
    scatter_fig.update_layout(
        title="Random Forest: Predicted vs Actual",
        xaxis_title="Actual Price (Rs)", yaxis_title="Predicted Price (Rs)",
        template="plotly_white",
    )
    scatter_fig.write_html(f"plot_regression_scatter_{TICKER}.html")

