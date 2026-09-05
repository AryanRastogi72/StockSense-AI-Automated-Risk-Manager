import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import joblib

from utils.paths import PROJECT_ROOT, SAVED_MODELS_DIR

SUPPORTED_TICKERS = ["LT", "TCS", "RELIANCE"]

VALID_ALGORITHMS = ["rf", "xgb", "lstm"]
VALID_TASKS = ["reg", "class"]
VALID_TUNINGS = ["baseline", "tuned"]

ALL_VARIANTS = [
    {"algorithm": a, "task": t, "tuning": u}
    for a in VALID_ALGORITHMS
    for t in VALID_TASKS
    for u in VALID_TUNINGS
]




class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class StockLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def get_variant_dir(ticker, algorithm, task, tuning):
    variant_name = f"{algorithm}_{task}_{tuning}"
    return SAVED_MODELS_DIR / ticker / variant_name


def validate_variant(ticker, algorithm, task, tuning):
    if ticker not in SUPPORTED_TICKERS:
        return False, f"Ticker '{ticker}' is not supported. Available: {SUPPORTED_TICKERS}"
    if algorithm not in VALID_ALGORITHMS:
        return False, f"Algorithm '{algorithm}' is not valid. Available: {VALID_ALGORITHMS}"
    if task not in VALID_TASKS:
        return False, f"Task '{task}' is not valid. Available: {VALID_TASKS}"
    if tuning not in VALID_TUNINGS:
        return False, f"Tuning '{tuning}' is not valid. Available: {VALID_TUNINGS}"

    variant_dir = get_variant_dir(ticker, algorithm, task, tuning)
    if not variant_dir.exists():
        return False, f"Model variant {algorithm}/{task}/{tuning} not found for ticker {ticker}. Run the training script first."
    return True, ""


def generate_plotly_graph(ticker, algorithm, task, tuning, live_data, prediction_result):
    df = live_data["raw_df"]
    # Plot last 126 trading days (~6 months)
    plot_df = df.tail(126)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="Actual OHLC",
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",
    ))

    # Determine next trading date for prediction point
    last_date = plot_df.index[-1]
    # Simple addition of 1 business day for plotting purposes
    next_date = last_date + pd.offsets.BDay(1)
    
    model_name = f"{algorithm.upper()} {tuning.capitalize()}"
    
    if task == "reg":
        pred_val = prediction_result["predicted_close"]
        
        # Draw a line from the last close to the predicted close
        fig.add_trace(go.Scatter(
            x=[last_date, next_date],
            y=[plot_df["Close"].iloc[-1], pred_val],
            mode="lines+markers",
            name=f"{model_name} Prediction",
            line=dict(color="#FF5722", width=2, dash="dash"),
            marker=dict(size=8, symbol="star")
        ))
        title = f"{model_name} Regression: Next Session Prediction for {ticker}"
    else:
        pred_dir = prediction_result["predicted_direction"]
        
        # Annotate the next day with the direction
        color = "#26A69A" if pred_dir == "Up" else ("#EF5350" if pred_dir == "Down" else "gray")
        ay_offset = -40 if pred_dir == "Up" else (40 if pred_dir == "Down" else 0)
        
        fig.add_annotation(
            x=next_date,
            y=plot_df["Close"].iloc[-1],
            text=f"Prediction: {pred_dir}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=ay_offset,
            font=dict(size=14, color="white"),
            bgcolor=color
        )
        title = f"{model_name} Classification: Next Session Prediction for {ticker}"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (Rs)",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
    )
    
    return fig


def predict_tree(ticker, algorithm, task, tuning, features_row):
    variant_dir = get_variant_dir(ticker, algorithm, task, tuning)
    model = joblib.load(variant_dir / "model.joblib")
    saved_feature_cols = joblib.load(variant_dir / "feature_cols.joblib")

    features_row = features_row[saved_feature_cols]

    if task == "reg":
        pred_pct_change = model.predict(features_row)[0]
        latest_close = features_row["Close"].values[0]
        predicted_close = float(latest_close * (1 + pred_pct_change))
        return {"predicted_close": round(predicted_close, 2)}

    encoder = joblib.load(variant_dir / "encoder.joblib") if algorithm == "xgb" else None
    raw_pred = model.predict(features_row)[0]

    if encoder is not None:
        predicted_direction = encoder.inverse_transform([raw_pred])[0]
    else:
        predicted_direction = raw_pred

    return {"predicted_direction": str(predicted_direction)}


def predict_lstm(ticker, task, tuning, sequence_df):
    variant_dir = get_variant_dir(ticker, "lstm", task, tuning)

    with open(variant_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    saved_feature_cols = joblib.load(variant_dir / "feature_cols.joblib")
    scaler_x = joblib.load(variant_dir / "scaler_x.joblib")

    sequence_df = sequence_df[saved_feature_cols]
    sequence_scaled = scaler_x.transform(sequence_df.values)
    X_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)

    if task == "reg":
        model = StockLSTM(
            metadata["input_size"], metadata["hidden_size"],
            metadata["num_layers"], metadata["dropout"],
        )
        model.load_state_dict(torch.load(variant_dir / "model.pt", weights_only=True))
        model.eval()

        scaler_y = joblib.load(variant_dir / "scaler_y.joblib")

        with torch.no_grad():
            pred_scaled = model(X_tensor).numpy().flatten()
        pred_pct_change = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

        latest_close = sequence_df["Close"].iloc[-1]
        predicted_close = float(latest_close * (1 + pred_pct_change))
        return {"predicted_close": round(predicted_close, 2)}

    model = StockLSTMClassifier(
        metadata["input_size"], metadata["hidden_size"],
        metadata["num_layers"], metadata["dropout"],
        metadata["num_classes"],
    )
    model.load_state_dict(torch.load(variant_dir / "model.pt", weights_only=True))
    model.eval()

    encoder = joblib.load(variant_dir / "encoder.joblib")

    with torch.no_grad():
        logits = model(X_tensor)
        pred_idx = torch.argmax(logits, dim=1).numpy()[0]

    predicted_direction = encoder.inverse_transform([pred_idx])[0]
    return {"predicted_direction": str(predicted_direction)}
