import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

SPLIT_RATIO = 0.8
SEED = 42
CLASSES = ["Up", "Down", "Flat"]
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
    y_strings = df["Target_Class_Next"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_strings)

    split_index = int(len(df) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    y_test_strings = y_strings.iloc[split_index:]

    majority_class = y_strings.iloc[:split_index].mode()[0]
    y_naive_strings = [majority_class] * len(y_test)
    accuracy_naive = accuracy_score(y_test_strings, y_naive_strings)

    baseline_model = XGBClassifier(n_estimators=100, random_state=SEED)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
    precision_baseline = precision_score(y_test, y_pred_baseline, average="macro")
    recall_baseline = recall_score(y_test, y_pred_baseline, average="macro")
    f1_baseline = f1_score(y_test, y_pred_baseline, average="macro")

    # Baseline metrics saved to JSON
    baseline_metrics = {
        "accuracy_naive": float(accuracy_naive),
        "accuracy_baseline": float(accuracy_baseline),
        "precision_baseline": float(precision_baseline),
        "recall_baseline": float(recall_baseline),
        "f1_baseline": float(f1_baseline)
    }
    from pathlib import Path
    save_path = Path("saved_models") / TICKER / "xg_class_tuned"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"{TICKER}_XG_baseline_comparison.json", "w") as f:
        json.dump(baseline_metrics, f, indent=4)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        XGBClassifier(random_state=SEED),
        PARAM_GRID,
        n_iter=60,
        cv=tscv,
        scoring="f1_macro",
        random_state=SEED,
    )
    search.fit(X_train, y_train, sample_weight=sample_weights)

    tuned_model = search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    y_pred_tuned_strings = encoder.inverse_transform(y_pred_tuned)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned, average="macro")
    recall_tuned = recall_score(y_test, y_pred_tuned, average="macro")
    f1_tuned = f1_score(y_test, y_pred_tuned, average="macro")

    save_dir = Path("saved_models") / TICKER / "xgb_class_tuned"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, save_dir / "model.joblib")
    joblib.dump(encoder, save_dir / "encoder.joblib")
    joblib.dump(list(feature_cols), save_dir / "feature_cols.joblib")

    cm = confusion_matrix(y_test_strings, y_pred_tuned_strings, labels=CLASSES)
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Oranges",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=CLASSES, y=CLASSES, title="Tuned XGBoost Classifier: Confusion Matrix",
    )
    cm_fig.write_html(f"plot_xgb_classification_tuned_confusion_matrix_{TICKER}.html")

    plot_window = 126
    plot_test = X_test.iloc[-plot_window:]
    plot_correct = (y_test_strings.values == y_pred_tuned_strings)[-plot_window:]

    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=plot_test.index, open=plot_test["Open"], high=plot_test["High"],
        low=plot_test["Low"], close=plot_test["Close"], name="Actual OHLC",
        increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_test.index[plot_correct], y=plot_test["Close"][plot_correct], mode="markers", name="Correct Prediction",
        marker=dict(color="#4CAF50", size=8, symbol="circle"),
    ))
    price_fig.add_trace(go.Scatter(
        x=plot_test.index[~plot_correct], y=plot_test["Close"][~plot_correct], mode="markers", name="Incorrect Prediction",
        marker=dict(color="#F44336", size=8, symbol="x"),
    ))
    price_fig.update_layout(
        title="Tuned XGBoost Classifier: Predictions on Price Action (Last 6 Months)",
        xaxis_title="Date", yaxis_title="Price (Rs)",
        template="plotly_white", xaxis_rangeslider_visible=False,
    )
    price_fig.write_html(f"plot_xgb_classification_tuned_candlestick_{TICKER}.html")

