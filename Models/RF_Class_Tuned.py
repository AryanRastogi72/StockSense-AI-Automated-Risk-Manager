import json
import joblib
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

SPLIT_RATIO = 0.8
SEED = 42
CLASSES = ["Up", "Down", "Flat"]
import sys
TICKER = sys.argv[1] if len(sys.argv) > 1 else "LT"

PARAM_GRID = {
    "n_estimators": [100, 150, 200, 250, 300],
    "max_depth": [10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.5, 0.7, 1.0],
}

if __name__ == '__main__':
    df = pd.read_csv(str(Path(__file__).resolve().parent.parent / "Supporting Data" / f"{TICKER}_cleaned_data.csv"), index_col="Date", parse_dates=True)

    feature_cols = [c for c in df.columns if c not in ["Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"]]
    X = df[feature_cols]
    y = df["Target_Class_Next"]

    split_index = int(len(df) * SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    majority_class = y_train.mode()[0]
    y_naive = [majority_class] * len(y_test)
    accuracy_naive = accuracy_score(y_test, y_naive)

    baseline_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
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
    save_path = Path("saved_models") / TICKER / "rf_class_tuned"
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / f"{TICKER}_RF_baseline_comparison.json", "w") as f:
        json.dump(baseline_metrics, f, indent=4)

    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=SEED),
        PARAM_GRID,
        n_iter=20,
        cv=tscv,
        scoring="accuracy",
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned, average="macro")
    recall_tuned = recall_score(y_test, y_pred_tuned, average="macro")
    f1_tuned = f1_score(y_test, y_pred_tuned, average="macro")

    save_dir = Path("saved_models") / TICKER / "rf_class_tuned"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, save_dir / "model.joblib")
    joblib.dump(list(feature_cols), save_dir / "feature_cols.joblib")

    cm = confusion_matrix(y_test, y_pred_tuned, labels=CLASSES)
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Greens",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=CLASSES, y=CLASSES, title="Tuned Random Forest Classifier: Confusion Matrix",
    )
    cm_fig.write_html(f"plot_classification_tuned_confusion_matrix_{TICKER}.html")

    correct = (y_test.values == y_pred_tuned)

    plot_window = 126
    plot_test = X_test.iloc[-plot_window:]
    plot_correct = correct[-plot_window:]

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
        title="Tuned Random Forest Classifier: Predictions on Price Action (Last 6 Months)",
        xaxis_title="Date", yaxis_title="Price (Rs)",
        template="plotly_white", xaxis_rangeslider_visible=False,
    )
    price_fig.write_html(f"plot_classification_tuned_candlestick_{TICKER}.html")

