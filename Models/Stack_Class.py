import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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

class StockLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def create_sequences(X_data, lookback):
    Xs = []
    for i in range(len(X_data) - lookback + 1):
        Xs.append(X_data[i:i + lookback])
    return np.array(Xs)

def evaluate_base_models_on_test():
    """Generates base model predictions (probabilities) on the 20% holdout test set."""
    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)
    split_index = int(len(df) * SPLIT_RATIO)
    
    base_dir = Path("saved_models") / TICKER
    
    feature_cols = joblib.load(base_dir / "rf_class_tuned" / "feature_cols.joblib")
    encoder = joblib.load(base_dir / "lstm_class_tuned" / "encoder.joblib")
    classes = encoder.classes_
    
    X_raw = df[feature_cols].values
    y_true_strings = df["Target_Class_Next"].iloc[split_index:].values
    y_true_encoded = encoder.transform(y_true_strings)
    test_dates = df.index[split_index:]
    
    X_test_raw = X_raw[split_index:]
    
    # RF
    rf = joblib.load(base_dir / "rf_class_tuned" / "model.joblib")
    rf_proba = rf.predict_proba(X_test_raw)
    
    # XGB
    xgb = joblib.load(base_dir / "xgb_class_tuned" / "model.joblib")
    xgb_proba = xgb.predict_proba(X_test_raw)
    
    # LSTM
    with open(base_dir / "lstm_class_tuned" / "metadata.json", "r") as f:
        lstm_params = json.load(f)
    lookback = lstm_params["lookback"]
    scaler_x = joblib.load(base_dir / "lstm_class_tuned" / "scaler_x.joblib")
    
    lstm = StockLSTMClassifier(
        input_size=lstm_params["input_size"],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        dropout=lstm_params["dropout"],
        num_classes=lstm_params["num_classes"]
    )
    lstm.load_state_dict(torch.load(base_dir / "lstm_class_tuned" / "model.pt", weights_only=True))
    lstm.eval()
    
    X_test_with_lb = X_raw[split_index - lookback + 1:]
    X_test_scaled = scaler_x.transform(X_test_with_lb)
    X_seq_test = create_sequences(X_test_scaled, lookback)
    
    with torch.no_grad():
        logits = lstm(torch.tensor(X_seq_test, dtype=torch.float32))
        lstm_proba = torch.softmax(logits, dim=1).numpy()
        
    res_dict = {"True_Target": y_true_encoded}
    for i, class_name in enumerate(classes):
        res_dict[f"RF_Prob_{class_name}"] = rf_proba[:, i]
        res_dict[f"XGB_Prob_{class_name}"] = xgb_proba[:, i]
        res_dict[f"LSTM_Prob_{class_name}"] = lstm_proba[:, i]
        
    return pd.DataFrame(res_dict, index=test_dates), y_true_strings, classes

def train_and_evaluate_stack():
    oof_df = pd.read_csv(f"oof_predictions_class_{TICKER}.csv", index_col="Date", parse_dates=True)
    encoder = joblib.load(f"oof_class_encoder_{TICKER}.joblib")
    classes = encoder.classes_
    
    feature_cols = []
    for class_name in classes:
        feature_cols.extend([f"RF_Prob_{class_name}", f"XGB_Prob_{class_name}", f"LSTM_Prob_{class_name}"])
        
    X_meta_train = oof_df[feature_cols]
    y_meta_train = oof_df["True_Target"]
    
    meta_model = LogisticRegression(random_state=SEED, class_weight="balanced", max_iter=500)
    meta_model.fit(X_meta_train, y_meta_train)
    
    weights = {}
    for i, class_name in enumerate(meta_model.classes_):
        true_class_name = encoder.inverse_transform([class_name])[0]
        weights[f"Class_{true_class_name}"] = {
            "Intercept": float(meta_model.intercept_[i]),
            "Coefficients": {feat: float(coef) for feat, coef in zip(feature_cols, meta_model.coef_[i])}
        }
    
    save_dir = Path("saved_models") / TICKER / "stack_class"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, save_dir / "model.joblib")
    joblib.dump(encoder, save_dir / "encoder.joblib")
    with open(save_dir / "meta_weights.json", "w") as f:
        json.dump(weights, f, indent=2)
        
    print("Meta-Learner Weights saved to meta_weights.json.")
        
    test_df, y_true_strings, _ = evaluate_base_models_on_test()
    X_meta_test = test_df[feature_cols]
    
    stack_pred_encoded = meta_model.predict(X_meta_test)
    stack_pred_strings = encoder.inverse_transform(stack_pred_encoded)
    
    def get_discrete_preds(prefix):
        cols = [f"{prefix}_Prob_{c}" for c in classes]
        return encoder.inverse_transform(test_df[cols].values.argmax(axis=1))
        
    rf_pred_strings = get_discrete_preds("RF")
    xgb_pred_strings = get_discrete_preds("XGB")
    lstm_pred_strings = get_discrete_preds("LSTM")
    
    # Note: On certain tickers (e.g., TCS.NS), this stack may underperform the best base model.
    # A diagnostic check confirmed this is a genuine generalization failure of the meta-learner on that data,
    # and NOT a bug in class-probability column ordering between RF, XGB, and LSTM (all align on [Down, Flat, Up]).
    print("\n--- Classification Stack Evaluation (Holdout 20%) ---")
    models_to_eval = [
        ("RF Tuned", rf_pred_strings),
        ("XGB Tuned", xgb_pred_strings),
        ("LSTM Tuned", lstm_pred_strings),
        ("Stacked Model", stack_pred_strings)
    ]
    
    for name, pred_strings in models_to_eval:
        acc = accuracy_score(y_true_strings, pred_strings)
        f1 = f1_score(y_true_strings, pred_strings, average="macro")
        print(f"{name:<15} Accuracy: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    train_and_evaluate_stack()
