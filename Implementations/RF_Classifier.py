import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

# Load dataset

df = pd.read_csv('cleaned_data.csv', index_col='Date', parse_dates=True)

print(f"Loaded cleaned_data.csv: {df.shape[0]} rows, {df.shape[1]} columns")

# Define target and features

target_col = 'Target_Class_Next'

exclude_cols = [target_col, 'Target_Close_Next', 'Target_Pct_Change']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

print(f"Target: ")
print(f"Features ({len(feature_cols)} columns): ")

# Chronological train/test split

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print(f"\nTrain set: {len(X_train)} rows  ({X_train.index.min().date()} to {X_train.index.max().date()})")
print(f"Test  set: {len(X_test)} rows  ({X_test.index.min().date()} to {X_test.index.max().date()})")

# Train classifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("\nRandom Forest Classifier trained successfully.")

# Predictions and evaluation

y_pred_rf = rf_model.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(
    y_test,
    y_pred_rf,
    labels=['Up', 'Down', 'Flat']
)

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    y_pred_rf,
    labels=['Up', 'Down', 'Flat']
))

# Majority-class baseline

majority_class = y_train.mode()[0]

y_pred_baseline = [majority_class] * len(y_test)

acc_baseline = accuracy_score(y_test, y_pred_baseline)

print("=" * 60)
print("  MODEL COMPARISON — Accuracy")
print("=" * 60)
print(f"Random Forest Classifier: {acc_rf:.2%}")
print(f"Majority Baseline (''): {acc_baseline:.2%}")
print("=" * 60)

if acc_rf > acc_baseline:
    print("Random Forest BEATS the majority baseline.")
else:
    print("Random Forest does NOT beat the majority baseline.")

# Confusion matrix heatmap

fig_cm = px.imshow(
    conf_matrix_rf,
    text_auto=True,
    color_continuous_scale='Blues',
    labels=dict(
        x="Predicted Class",
        y="Actual Class",
        color="Count"
    ),
    x=['Up', 'Down', 'Flat'],
    y=['Up', 'Down', 'Flat'],
    title='Random Forest Classifier: Confusion Matrix'
)

fig_cm.write_html('plot_class_a_confusion_matrix.html')

print("\nSaved: plot_class_a_confusion_matrix.html")

# Prediction time series

test_dates = X_test.index

correct_mask = (y_test == y_pred_rf)
incorrect_mask = ~correct_mask

fig_ts = go.Figure()

fig_ts.add_trace(go.Scatter(
    x=test_dates,
    y=X_test['Close'],
    mode='lines',
    name='Closing Price',
    line=dict(color='lightgray', width=2)
))

fig_ts.add_trace(go.Scatter(
    x=test_dates[correct_mask],
    y=X_test['Close'][correct_mask],
    mode='markers',
    name='Correct Prediction',
    marker=dict(
        color='#4CAF50',
        size=8,
        symbol='circle'
    ),
    text=[
        f"Actual: {a} | Pred: {p}"
        for a, p in zip(
            y_test[correct_mask],
            y_pred_rf[correct_mask]
        )
    ],
    hoverinfo='text+x+y'
))

fig_ts.add_trace(go.Scatter(
    x=test_dates[incorrect_mask],
    y=X_test['Close'][incorrect_mask],
    mode='markers',
    name='Incorrect Prediction',
    marker=dict(
        color='#F44336',
        size=8,
        symbol='x'
    ),
    text=[
        f"Actual: {a} | Pred: {p}"
        for a, p in zip(
            y_test[incorrect_mask],
            y_pred_rf[incorrect_mask]
        )
    ],
    hoverinfo='text+x+y'
))

fig_ts.update_layout(
    title='Random Forest Classifier: Test Set Predictions',
    xaxis_title='Date',
    yaxis_title='Closing Price (₹)',
    template='plotly_white',
    hovermode='closest'
)

fig_ts.write_html('plot_class_b_time_series.html')

print("Saved: plot_class_b_time_series.html")

print("\nDone. Classifier charts saved as HTML files.")