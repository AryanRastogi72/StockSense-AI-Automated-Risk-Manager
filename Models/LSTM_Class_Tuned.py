import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import plotly.graph_objects as go
import plotly.express as px
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
CLASSES = ["Up", "Down", "Flat"]
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
    y_strings = df["Target_Class_Next"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_strings)
    num_classes = len(encoder.classes_)

    split_index = int(len(df) * SPLIT_RATIO)
    X_train_raw = X_raw[:split_index]
    y_train_raw = y_encoded[:split_index]
    y_test_strings = y_strings[split_index:]


    def create_sequences(X_data, y_data, lookback):
        Xs, ys = [], []
        for i in range(len(X_data) - lookback):
            Xs.append(X_data[i:i + lookback])
            ys.append(y_data[i + lookback])
        return np.array(Xs), np.array(ys)


    class StockLSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])


    def train_model(X_seq, y_seq, hidden_size, learning_rate, dropout, epochs, class_weights):
        model = StockLSTMClassifier(len(feature_cols), hidden_size, NUM_LAYERS, dropout, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.long)
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

    best_score = -1.0
    best_params = None

    for params in combos:
        fold_scores = []
        for train_idx, val_idx in tscv.split(X_train_raw):
            if len(train_idx) < LOOKBACK:
                continue
            scaler_fold = StandardScaler()
            X_fold_train = scaler_fold.fit_transform(X_train_raw[train_idx])
            X_fold_val = scaler_fold.transform(X_train_raw[val_idx])

            y_fold_train = y_train_raw[train_idx]
            weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_fold_train)
            weights_t = torch.tensor(weights, dtype=torch.float32)

            X_seq_train, y_seq_train = create_sequences(X_fold_train, y_fold_train, LOOKBACK)
            X_val_with_lb = np.vstack([X_fold_train[-LOOKBACK:], X_fold_val])
            X_seq_val, _ = create_sequences(X_val_with_lb, np.zeros(len(X_val_with_lb)), LOOKBACK)
            y_seq_val_true = y_train_raw[val_idx]

            model = train_model(X_seq_train, y_seq_train, params["hidden_size"],
                                 params["learning_rate"], params["dropout"], SEARCH_EPOCHS, weights_t)
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(model(torch.tensor(X_seq_val, dtype=torch.float32)), dim=1).numpy()
            fold_scores.append(f1_score(y_seq_val_true, preds, average="macro"))

        avg_score = np.mean(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_raw[split_index - LOOKBACK:])

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, LOOKBACK)
    X_test_seq, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)), LOOKBACK)

    full_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train_raw)
    full_weights_t = torch.tensor(full_weights, dtype=torch.float32)

    final_model = train_model(X_train_seq, y_train_seq, best_params["hidden_size"],
                               best_params["learning_rate"], best_params["dropout"],
                               FINAL_EPOCHS, full_weights_t)
    final_model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(final_model(torch.tensor(X_test_seq, dtype=torch.float32)), dim=1).numpy()
    y_pred_strings = encoder.inverse_transform(y_pred)

    y_test_seq = y_encoded[split_index:]
    accuracy_tuned = accuracy_score(y_test_seq, y_pred)
    precision_tuned = precision_score(y_test_seq, y_pred, average="macro")
    recall_tuned = recall_score(y_test_seq, y_pred, average="macro")
    f1_tuned = f1_score(y_test_seq, y_pred, average="macro")

    majority_class = pd.Series(y_strings[:split_index]).mode()[0]
    y_naive_strings = [majority_class] * len(y_test_strings)
    accuracy_naive = accuracy_score(y_test_strings, y_naive_strings)

    save_dir = Path("saved_models") / TICKER / "lstm_class_tuned"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), save_dir / "model.pt")
    joblib.dump(scaler, save_dir / "scaler_x.joblib")
    joblib.dump(encoder, save_dir / "encoder.joblib")
    joblib.dump(list(feature_cols), save_dir / "feature_cols.joblib")
    metadata = {
        "input_size": len(feature_cols),
        "hidden_size": best_params["hidden_size"],
        "num_layers": NUM_LAYERS,
        "dropout": best_params["dropout"],
        "lookback": LOOKBACK,
        "num_classes": num_classes,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    cm = confusion_matrix(y_test_strings, y_pred_strings, labels=CLASSES)
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=CLASSES, y=CLASSES, title="Tuned LSTM Classifier: Confusion Matrix",
    )
    cm_fig.write_html(f"plot_lstm_classification_tuned_confusion_matrix_{TICKER}.html")

    test_dates = df.index[split_index:]
    plot_window = 126
    plot_dates = test_dates[-plot_window:]
    plot_close = df["Close"].iloc[split_index:].values[-plot_window:]
    plot_open = df["Open"].iloc[split_index:].values[-plot_window:]
    plot_high = df["High"].iloc[split_index:].values[-plot_window:]
    plot_low = df["Low"].iloc[split_index:].values[-plot_window:]
    plot_correct = (y_test_strings == y_pred_strings)[-plot_window:]

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=plot_dates, open=plot_open, high=plot_high, low=plot_low, close=plot_close,
        name="Actual OHLC", increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_dates[plot_correct], y=plot_close[plot_correct], mode="markers", name="Correct Prediction",
        marker=dict(color="#4CAF50", size=8, symbol="circle"),
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_dates[~plot_correct], y=plot_close[~plot_correct], mode="markers", name="Incorrect Prediction",
        marker=dict(color="#F44336", size=8, symbol="x"),
    ))
    price_fig.update_layout(
        title="Tuned LSTM Classifier: Predictions on Price Action (Last 6 Months)",
        xaxis_title="Date", yaxis_title="Price (Rs)",
        template="plotly_white", xaxis_rangeslider_visible=False,
    )
    price_fig.write_html(f"plot_lstm_classification_tuned_candlestick_{TICKER}.html")

