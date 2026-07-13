import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px

# Load dataset

df = pd.read_csv('cleaned_data.csv', index_col='Date', parse_dates=True)

# Define target and features

target_col = 'Target_Class_Next'

exclude_cols = [target_col, 'Target_Close_Next', 'Target_Pct_Change']
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

# Chronological train/test split

split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training set: {len(X_train)} rows")
print(f"Testing set:  {len(X_test)} rows\n")

# Hyperparameter grid

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.5, 1.0]
}

# Time-series cross-validation

tscv = TimeSeriesSplit(n_splits=5)

base_clf = RandomForestClassifier(random_state=42, class_weight='balanced')

grid_search = GridSearchCV(
    estimator=base_clf,
    param_grid=param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Run grid search

print("Starting GridSearchCV for Random Forest Classifier...")

start_time = time.time()

grid_search.fit(X_train, y_train)

elapsed = time.time() - start_time

print(f"\nGrid search completed in  seconds.")
print("\nBest Hyperparameters Found:")

for param, value in grid_search.best_params_.items():
    print(f"  : ")

# Evaluate tuned model

best_clf = grid_search.best_estimator_

y_pred_tuned = best_clf.predict(X_test)

acc_tuned = accuracy_score(y_test, y_pred_tuned)

conf_matrix_tuned = confusion_matrix(
    y_test,
    y_pred_tuned,
    labels=['Up', 'Down', 'Flat']
)

print("\nTuned Random Forest Classification Report:")
print(classification_report(
    y_test,
    y_pred_tuned,
    labels=['Up', 'Down', 'Flat']
))

# Majority-class baseline

majority_class = y_train.mode()[0]

y_pred_naive = [majority_class] * len(y_test)

acc_naive = accuracy_score(y_test, y_pred_naive)

# Baseline Random Forest

baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_rf.fit(X_train, y_train)

y_pred_base_rf = baseline_rf.predict(X_test)

acc_base_rf = accuracy_score(y_test, y_pred_base_rf)

print("=" * 60)
print("  FINAL CLASSIFIER COMPARISON — Accuracy")
print("=" * 60)
print(f"Tuned Random Forest:      {acc_tuned:.2%}")
print(f"Baseline Random Forest:   {acc_base_rf:.2%}")
print(f"Majority Baseline (''): {acc_naive:.2%}")
print("=" * 60)

# Confusion matrix heatmap

fig_cm = px.imshow(
    conf_matrix_tuned,
    text_auto=True,
    color_continuous_scale='Greens',
    labels=dict(
        x="Predicted Class",
        y="Actual Class",
        color="Count"
    ),
    x=['Up', 'Down', 'Flat'],
    y=['Up', 'Down', 'Flat'],
    title='Tuned RF Classifier: Confusion Matrix'
)

fig_cm.write_html('plot_class_tuned_a_confusion_matrix.html')

print("\nSaved: plot_class_tuned_a_confusion_matrix.html")

# Prediction time series

test_dates = X_test.index

correct_mask = (y_test == y_pred_tuned)
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
            y_pred_tuned[correct_mask]
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
            y_pred_tuned[incorrect_mask]
        )
    ],
    hoverinfo='text+x+y'
))

fig_ts.update_layout(
    title='Tuned RF Classifier: Test Set Predictions',
    xaxis_title='Date',
    yaxis_title='Closing Price (₹)',
    template='plotly_white',
    hovermode='closest'
)

fig_ts.write_html('plot_class_tuned_b_time_series.html')

print("Saved: plot_class_tuned_b_time_series.html")

print("\nDone. Tuned classifier charts saved as HTML files.")