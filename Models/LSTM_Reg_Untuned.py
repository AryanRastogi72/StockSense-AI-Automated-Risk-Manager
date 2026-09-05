import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

SPLIT_RATIO = 0.8
SEED = 42
LOOKBACK = 30
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"

if __name__ == '__main__':
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)

    feature_cols = [c for c in df.columns if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]]
    X_raw = df[feature_cols].values
    y_raw = df["Target_Pct_Change"].values

    split_index = int(len(df) * SPLIT_RATIO)

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_raw[:split_index])
    X_test_scaled = scaler_x.transform(X_raw[split_index - LOOKBACK:])

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_raw[:split_index].reshape(-1, 1)).flatten()


    def create_sequences(X_data, y_data, lookback):
        Xs, ys = [], []
        for i in range(len(X_data) - lookback):
            Xs.append(X_data[i:i + lookback])
            ys.append(y_data[i + lookback])
        return np.array(Xs), np.array(ys)


    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, LOOKBACK)
    X_test_seq, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)), LOOKBACK)


    class StockLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])


    model = StockLSTM(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=False)

    model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(X_test_t).numpy().flatten()
    pred_pct_change = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    today_close = df["Close"].iloc[split_index:].values
    y_pred_price = today_close * (1 + pred_pct_change)
    y_true_price = df["Target_Close_Next"].iloc[split_index:].values
    y_naive_price = today_close

    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    r2 = r2_score(y_true_price, y_pred_price)
    rmse_naive = np.sqrt(mean_squared_error(y_true_price, y_naive_price))
    r2_naive = r2_score(y_true_price, y_naive_price)

    save_dir = Path("saved_models") / TICKER / "lstm_reg_baseline"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    joblib.dump(scaler_x, save_dir / "scaler_x.joblib")
    joblib.dump(scaler_y, save_dir / "scaler_y.joblib")
    joblib.dump(list(feature_cols), save_dir / "feature_cols.joblib")
    metadata = {
        "input_size": len(feature_cols),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": 0.2,
        "lookback": LOOKBACK,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    test_dates = df.index[split_index:]
    plot_window = 126
    plot_dates = test_dates[-plot_window:]
    plot_open = df["Open"].iloc[split_index:].values[-plot_window:]
    plot_high = df["High"].iloc[split_index:].values[-plot_window:]
    plot_low = df["Low"].iloc[split_index:].values[-plot_window:]
    plot_close = df["Close"].iloc[split_index:].values[-plot_window:]
    plot_pred = y_pred_price[-plot_window:]
    plot_naive = y_naive_price[-plot_window:]

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=plot_dates, open=plot_open, high=plot_high, low=plot_low, close=plot_close,
        name="Actual OHLC", increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_dates, y=plot_pred, mode="lines", name="LSTM Predicted",
        line=dict(color="#E91E63", width=2, dash="dash"),
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_dates, y=plot_naive, mode="lines", name="Naive Baseline",
        line=dict(color="#4CAF50", width=2, dash="dot"),
    ))
    price_fig.update_layout(
        title="LSTM: Predictions vs Actual Price Action (Last 6 Months)",
        xaxis_title="Date", yaxis_title="Price (Rs)",
        template="plotly_white", xaxis_rangeslider_visible=False,
    )
    price_fig.write_html(f"plot_lstm_regression_candlestick_{TICKER}.html")

    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=y_true_price, y=y_pred_price, mode="markers", name="Predictions",
        marker=dict(color="#E91E63", size=5, opacity=0.6),
    ))
    line_min = min(y_true_price.min(), y_pred_price.min())
    line_max = max(y_true_price.max(), y_pred_price.max())
    scatter_fig.add_trace(go.Scatter(
        x=[line_min, line_max], y=[line_min, line_max], mode="lines",
        name="Perfect Prediction", line=dict(color="gray", dash="dash"),
    ))
    scatter_fig.update_layout(
        title="LSTM: Predicted vs Actual",
        xaxis_title="Actual Price (Rs)", yaxis_title="Predicted Price (Rs)",
        template="plotly_white",
    )
    scatter_fig.write_html(f"plot_lstm_regression_scatter_{TICKER}.html")

