import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import joblib
import json
from pathlib import Path
from oof_generator import generate_cv_indices

SPLIT_RATIO = 0.8
SEED = 42
import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"
N_SPLITS = 5
CLASSES = ["Up", "Down", "Flat"]

# LSTM specific
LSTM_EPOCHS = 100
BATCH_SIZE = 32
device = torch.device("cpu")

class StockLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

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
    base_dir = Path("saved_models") / ticker
    
    rf_model = joblib.load(base_dir / "rf_class_tuned" / "model.joblib")
    rf_params = rf_model.get_params()
    
    xgb_model = joblib.load(base_dir / "xgb_class_tuned" / "model.joblib")
    xgb_params = xgb_model.get_params()
    
    with open(base_dir / "lstm_class_tuned" / "metadata.json", "r") as f:
        lstm_params = json.load(f)
        
    return rf_params, xgb_params, lstm_params
    
def train_lstm_fold_class(X_train, y_train, lstm_params, num_classes):
    lookback = lstm_params["lookback"]
    
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    
    X_seq, y_seq = create_sequences(X_train_scaled, y_train, lookback)
    
    model = StockLSTMClassifier(
        input_size=X_train.shape[1],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"],
        num_classes=num_classes
    ).to(device)
    
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_params.get("learning_rate", 0.001))
    
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y_seq, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=False)
    
    model.train()
    for _ in range(LSTM_EPOCHS):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    return model, scaler_x
    
def generate_oof_classification():
    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)
    feature_cols = [c for c in df.columns if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]]
    
    X_raw = df[feature_cols].values
    y_strings = df["Target_Class_Next"].values
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_strings)
    num_classes = len(encoder.classes_)
    
    rf_params, xgb_params, lstm_params = get_base_model_params(TICKER)
    lookback = lstm_params["lookback"]
    
    folds = generate_cv_indices(len(df), SPLIT_RATIO, N_SPLITS, lookback)
    
    oof_predictions = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"Processing Fold {fold_idx + 1}/{len(folds)}...")
        
        X_train, y_train = X_raw[train_idx], y_encoded[train_idx]
        X_val, y_val = X_raw[val_idx], y_encoded[val_idx]
        
        # --- Random Forest ---
        rf = RandomForestClassifier(**{k: v for k, v in rf_params.items() if k != "random_state" and k != "class_weight"})
        rf.set_params(random_state=SEED, class_weight="balanced")
        rf.fit(X_train, y_train)
        rf_proba = rf.predict_proba(X_val)
        
        # --- XGBoost ---
        xgb = XGBClassifier(**{k: v for k, v in xgb_params.items() if k != "random_state"})
        xgb.set_params(random_state=SEED)
        
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        xgb.fit(X_train, y_train, sample_weight=sample_weights)
        xgb_proba = xgb.predict_proba(X_val)
        
        # --- LSTM ---
        lstm, scaler_x = train_lstm_fold_class(X_train, y_train, lstm_params, num_classes)
        
        X_val_with_lb = np.vstack([X_train[-lookback:], X_val])
        X_val_scaled = scaler_x.transform(X_val_with_lb)
        X_seq_val, _ = create_sequences(X_val_scaled, np.zeros(len(X_val_scaled)), lookback)
        
        lstm.eval()
        with torch.no_grad():
            logits = lstm(torch.tensor(X_seq_val, dtype=torch.float32))
            lstm_proba = torch.softmax(logits, dim=1).numpy()
            
        fold_dates = df.index[val_idx]
        
        fold_dict = {"Date": fold_dates}
        for i, class_name in enumerate(encoder.classes_):
            fold_dict[f"RF_Prob_{class_name}"] = rf_proba[:, i]
            fold_dict[f"XGB_Prob_{class_name}"] = xgb_proba[:, i]
            fold_dict[f"LSTM_Prob_{class_name}"] = lstm_proba[:, i]
            
        fold_dict["True_Target"] = y_val
        
        fold_df = pd.DataFrame(fold_dict)
        oof_predictions.append(fold_df)
        
    oof_df = pd.concat(oof_predictions).set_index("Date")
    oof_df.to_csv(f"oof_predictions_class_{TICKER}.csv")
    joblib.dump(encoder, f"oof_class_encoder_{TICKER}.joblib")
    print("Classification OOF generation complete.")

if __name__ == "__main__":
    generate_oof_classification()
