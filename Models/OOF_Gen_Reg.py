import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import json
from pathlib import Path
from oof_generator import generate_cv_indices

SPLIT_RATIO = 0.8
SEED = 42
import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"
N_SPLITS = 5

# LSTM specific
LSTM_EPOCHS = 100
BATCH_SIZE = 32
device = torch.device("cpu")

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def create_sequences(X_data, y_data, lookback):
    Xs, ys = [], []
    for i in range(len(X_data) - lookback):
        Xs.append(X_data[i:i + lookback])
        ys.append(y_data[i + lookback])
    return np.array(Xs), np.array(ys)

def get_base_model_params(ticker: str):
    """Loads tuned hyperparameters for the 3 base models."""
    base_dir = Path("saved_models") / ticker
    
    # RF
    rf_model = joblib.load(base_dir / "rf_reg_tuned" / "model.joblib")
    rf_params = rf_model.get_params()
    
    # XGB
    xgb_model = joblib.load(base_dir / "xgb_reg_tuned" / "model.joblib")
    xgb_params = xgb_model.get_params()
    
    # LSTM
    with open(base_dir / "lstm_reg_tuned" / "metadata.json", "r") as f:
        lstm_params = json.load(f)
        
    return rf_params, xgb_params, lstm_params

def train_lstm_fold(X_train, y_train, lstm_params):
    """Trains an LSTM model on a single fold."""
    lookback = lstm_params["lookback"]
    
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_seq, y_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    
    model = StockLSTM(
        input_size=X_train.shape[1],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"]
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_params.get("learning_rate", 0.001)) # Assuming LR from tuning
    
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False)
    
    model.train()
    for _ in range(LSTM_EPOCHS):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    return model, scaler_x, scaler_y

def generate_oof_regression():
    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)
    feature_cols = [c for c in df.columns if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]]
    
    X_raw = df[feature_cols].values
    y_raw = df["Target_Pct_Change"].values
    
    rf_params, xgb_params, lstm_params = get_base_model_params(TICKER)
    lookback = lstm_params["lookback"]
    
    folds = generate_cv_indices(len(df), SPLIT_RATIO, N_SPLITS, lookback)
    
    oof_predictions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"Processing Fold {fold_idx + 1}/{len(folds)}...")
        
        X_train, y_train = X_raw[train_idx], y_raw[train_idx]
        X_val, y_val = X_raw[val_idx], y_raw[val_idx]
        
        # --- Random Forest ---
        rf = RandomForestRegressor(**{k: v for k, v in rf_params.items() if k != "random_state"})
        rf.set_params(random_state=SEED)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_val)
        
        # --- XGBoost ---
        xgb = XGBRegressor(**{k: v for k, v in xgb_params.items() if k != "random_state"})
        xgb.set_params(random_state=SEED)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_val)
        
        # --- LSTM ---
        # Note: In the tuned script, learning_rate wasn't saved in metadata.json! 
        # I need to ensure it uses the correct LR or a default if missing.
        # But wait, looking at metadata.json generated, it only saved:
        # input_size, hidden_size, num_layers, dropout, lookback.
        # It missed learning_rate! We will assume learning_rate=0.001 which is the default in PyTorch Adam if not specified.
        # For validation, LSTM needs the sequence to include the last `lookback` samples from training set
        lstm, scaler_x, scaler_y = train_lstm_fold(X_train, y_train, lstm_params)
        
        X_val_with_lb = np.vstack([X_train[-lookback:], X_val])
        X_val_scaled = scaler_x.transform(X_val_with_lb)
        X_seq_val, _ = create_sequences(X_val_scaled, np.zeros(len(X_val_scaled)), lookback)
        
        lstm.eval()
        with torch.no_grad():
            preds_scaled = lstm(torch.tensor(X_seq_val, dtype=torch.float32)).numpy().flatten()
        lstm_pred = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        
        # --- Store OOF Data ---
        fold_dates = df.index[val_idx]
        fold_df = pd.DataFrame({
            "Date": fold_dates,
            "RF_Pred": rf_pred,
            "XGB_Pred": xgb_pred,
            "LSTM_Pred": lstm_pred,
            "True_Target": y_val
        })
        oof_predictions.append(fold_df)
        
    oof_df = pd.concat(oof_predictions).set_index("Date")
    oof_df.to_csv(f"oof_predictions_reg_{TICKER}.csv")
    print("Regression OOF generation complete. Saved to oof_predictions_reg_LT.csv")

if __name__ == "__main__":
    generate_oof_regression()
