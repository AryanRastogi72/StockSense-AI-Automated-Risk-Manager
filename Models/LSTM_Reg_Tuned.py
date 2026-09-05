import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

SPLIT_RATIO = 0.8
SEED = 42
LOOKBACK = 30
SEARCH_EPOCHS = 30
FINAL_EPOCHS = 100
BATCH_SIZE = 32
NUM_LAYERS = 2
import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"

PARAM_GRID = {
    "hidden_size": [32, 64],
    "learning_rate": [0.01, 0.001],
    "dropout": [0.1, 0.3],
}

if __name__ == '__main__':
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)

    feature_cols = [c for c in df.columns if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]]
    X_raw = df[feature_cols].values
    y_raw = df["Target_Pct_Change"].values

    split_index = int(len(df) * SPLIT_RATIO)
    X_train_raw = X_raw[:split_index]
    y_train_raw = y_raw[:split_index]


    def create_sequences(X_data, y_data, lookback):
        Xs, ys = [], []
        for i in range(len(X_data) - lookback):
            Xs.append(X_data[i:i + lookback])
            ys.append(y_data[i + lookback])
        return np.array(Xs), np.array(ys)


    class StockLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])


    def train_model(X_seq, y_seq, hidden_size, learning_rate, dropout, epochs):
        model = StockLSTM(len(feature_cols), hidden_size, NUM_LAYERS, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False)
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()
        return model


    keys = list(PARAM_GRID.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*PARAM_GRID.values())]
    tscv = TimeSeriesSplit(n_splits=2)

    best_score = float("inf")
    best_params = None

    for params in combos:
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_train_raw):
            if len(train_idx) < LOOKBACK:
                continue
            scaler_x_fold = StandardScaler()
            X_fold_train = scaler_x_fold.fit_transform(X_train_raw[train_idx])
            X_fold_val = scaler_x_fold.transform(X_train_raw[val_idx])

            scaler_y_fold = StandardScaler()
            y_fold_train = scaler_y_fold.fit_transform(y_train_raw[train_idx].reshape(-1, 1)).flatten()

            X_seq_train, y_seq_train = create_sequences(X_fold_train, y_fold_train, LOOKBACK)
            X_val_with_lb = np.vstack([X_fold_train[-LOOKBACK:], X_fold_val])
            X_seq_val, _ = create_sequences(X_val_with_lb, np.zeros(len(X_val_with_lb)), LOOKBACK)
            y_seq_val_true = y_train_raw[val_idx]

            model = train_model(X_seq_train, y_seq_train, params["hidden_size"],
                                 params["learning_rate"], params["dropout"], SEARCH_EPOCHS)
            model.eval()
            with torch.no_grad():
                preds_scaled = model(torch.tensor(X_seq_val, dtype=torch.float32)).numpy().flatten()
            preds_pct = scaler_y_fold.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            fold_scores.append(np.sqrt(mean_squared_error(y_seq_val_true, preds_pct)))

        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train_raw)
    X_test_scaled = scaler_x.transform(X_raw[split_index - LOOKBACK:])

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, LOOKBACK)
    X_test_seq, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)), LOOKBACK)

    final_model = train_model(X_train_seq, y_train_seq, best_params["hidden_size"],
                               best_params["learning_rate"], best_params["dropout"], FINAL_EPOCHS)
    final_model.eval()
    with torch.no_grad():
        pred_scaled = final_model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy().flatten()
    pred_pct_change = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    today_close = df["Close"].iloc[split_index:].values
    y_pred_price = today_close * (1 + pred_pct_change)
    y_true_price = df["Target_Close_Next"].iloc[split_index:].values
    y_naive_price = today_close

    rmse_tuned = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    r2_tuned = r2_score(y_true_price, y_pred_price)
    rmse_naive = np.sqrt(mean_squared_error(y_true_price, y_naive_price))
    r2_naive = r2_score(y_true_price, y_naive_price)

    save_dir = Path("saved_models") / TICKER / "lstm_reg_tuned"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), save_dir / "model.pt")
    joblib.dump(scaler_x, save_dir / "scaler_x.joblib")
    joblib.dump(scaler_y, save_dir / "scaler_y.joblib")
    joblib.dump(list(feature_cols), save_dir / "feature_cols.joblib")
    metadata = {
        "input_size": len(feature_cols),
        "hidden_size": best_params["hidden_size"],
        "num_layers": NUM_LAYERS,
        "dropout": best_params["dropout"],
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
        x=plot_dates, y=plot_pred, mode="lines", name="Tuned LSTM Predicted",
        line=dict(color="#E91E63", width=2, dash="dash"),
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_dates, y=plot_naive, mode="lines", name="Naive Baseline",
        line=dict(color="#4CAF50", width=2, dash="dot"),
    ))
    price_fig.update_layout(
        title="Tuned LSTM: Predictions vs Actual Price Action (Last 6 Months)",
        xaxis_title="Date", yaxis_title="Price (Rs)",
        template="plotly_white", xaxis_rangeslider_visible=False,
    )
    price_fig.write_html(f"plot_lstm_regression_tuned_candlestick_{TICKER}.html")

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
        title="Tuned LSTM: Predicted vs Actual",
        xaxis_title="Actual Price (Rs)", yaxis_title="Predicted Price (Rs)",
        template="plotly_white",
    )
    scatter_fig.write_html(f"plot_lstm_regression_tuned_scatter_{TICKER}.html")

