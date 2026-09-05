import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"
SPLIT_RATIO = 0.8
SEED = 42

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def create_sequences(X_data, lookback):
    Xs = []
    for i in range(len(X_data) - lookback + 1):
        Xs.append(X_data[i:i + lookback])
    return np.array(Xs)

def evaluate_base_models_on_test():
    """Generates base model predictions on the 20% holdout test set."""
    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)
    split_index = int(len(df) * SPLIT_RATIO)
    
    base_dir = Path("saved_models") / TICKER
    
    feature_cols = joblib.load(base_dir / "rf_reg_tuned" / "feature_cols.joblib")
    X_raw = df[feature_cols].values
    y_true = df["Target_Pct_Change"].iloc[split_index:].values
    today_close = df["Close"].iloc[split_index:].values
    y_true_price = df["Target_Close_Next"].iloc[split_index:].values
    test_dates = df.index[split_index:]
    
    X_test_raw = X_raw[split_index:]
    
    # RF
    rf = joblib.load(base_dir / "rf_reg_tuned" / "model.joblib")
    rf_pred = rf.predict(X_test_raw)
    
    # XGB
    xgb = joblib.load(base_dir / "xgb_reg_tuned" / "model.joblib")
    xgb_pred = xgb.predict(X_test_raw)
    
    # LSTM
    with open(base_dir / "lstm_reg_tuned" / "metadata.json", "r") as f:
        lstm_params = json.load(f)
    lookback = lstm_params["lookback"]
    scaler_x = joblib.load(base_dir / "lstm_reg_tuned" / "scaler_x.joblib")
    scaler_y = joblib.load(base_dir / "lstm_reg_tuned" / "scaler_y.joblib")
    
    lstm = StockLSTM(
        input_size=lstm_params["input_size"],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"]
    )
    lstm.load_state_dict(torch.load(base_dir / "lstm_reg_tuned" / "model.pt", weights_only=True))
    lstm.eval()
    
    X_test_with_lb = X_raw[split_index - lookback + 1:]
    X_test_scaled = scaler_x.transform(X_test_with_lb)
    X_seq_test = create_sequences(X_test_scaled, lookback)
    
    with torch.no_grad():
        preds_scaled = lstm(torch.tensor(X_seq_test, dtype=torch.float32)).numpy().flatten()
    lstm_pred = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    
    return pd.DataFrame({
        "RF_Pred": rf_pred,
        "XGB_Pred": xgb_pred,
        "LSTM_Pred": lstm_pred,
        "True_Target": y_true
    }, index=test_dates), today_close, y_true_price

def train_and_evaluate_stack():
    oof_df = pd.read_csv(f"oof_predictions_reg_{TICKER}.csv", index_col="Date", parse_dates=True)
    X_meta_train = oof_df[["RF_Pred", "XGB_Pred", "LSTM_Pred"]]
    y_meta_train = oof_df["True_Target"]
    
    meta_model = Ridge(alpha=1.0, random_state=SEED)
    meta_model.fit(X_meta_train, y_meta_train)
    
    weights = {
        "Intercept": float(meta_model.intercept_),
        "RF_Weight": float(meta_model.coef_[0]),
        "XGB_Weight": float(meta_model.coef_[1]),
        "LSTM_Weight": float(meta_model.coef_[2])
    }
    
    save_dir = Path("saved_models") / TICKER / "stack_reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, save_dir / "model.joblib")
    with open(save_dir / "meta_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
        
    print("Meta-Learner Weights (Ridge Regression):")
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")
        
    test_df, today_close, y_true_price = evaluate_base_models_on_test()
    X_meta_test = test_df[["RF_Pred", "XGB_Pred", "LSTM_Pred"]]
    
    stack_pred_pct = meta_model.predict(X_meta_test)
    test_df["Stack_Pred"] = stack_pred_pct
    
    print("\n--- Regression Stack Evaluation (Holdout 20%) ---")
    models_to_eval = [
        ("RF Tuned", test_df["RF_Pred"]),
        ("XGB Tuned", test_df["XGB_Pred"]),
        ("LSTM Tuned", test_df["LSTM_Pred"]),
        ("Stacked Model", test_df["Stack_Pred"])
    ]
    
    for name, pred_pct in models_to_eval:
        pred_price = today_close * (1 + pred_pct)
        rmse = np.sqrt(mean_squared_error(y_true_price, pred_price))
        r2 = r2_score(y_true_price, pred_price)
        print(f"{name:<15} RMSE: {rmse:.2f} | R2: {r2:.4f}")

if __name__ == "__main__":
    train_and_evaluate_stack()
