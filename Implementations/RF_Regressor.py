import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

# Load dataset

df = pd.read_csv('cleaned_data.csv', index_col='Date', parse_dates=True)
print(f"Loaded cleaned_data.csv: {df.shape[0]} rows, {df.shape[1]} columns")

# Define target and features

target_col = 'Target_Close_Next'

feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols]
y = df[target_col]

print(f"\nFeatures ({len(feature_cols)} columns): ")
print(f"Target: ")

# Chronological train/test split

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print(f"\nTrain set: {len(X_train)} rows  ({X_train.index.min().date()} to {X_train.index.max().date()})")
print(f"Test  set: {len(X_test)} rows  ({X_test.index.min().date()} to {X_test.index.max().date()})")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
print("\nRandom Forest trained successfully.")

y_pred_rf = rf_model.predict(X_test)

# Evaluation metrics

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)

# Naive baseline

y_pred_naive = X_test['Close'].values

rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
r2_naive   = r2_score(y_test, y_pred_naive)

print("\n" + "=" * 55)
print("  MODEL EVALUATION — Test Set Results")
print("=" * 55)
print(f"{'Metric':<12} {'Random Forest':>15} {'Naive Baseline':>16}")
print("-" * 55)
print(f"{'RMSE':<12} {rmse_rf:>15.2f} {rmse_naive:>16.2f}")
print(f"{'R²':<12} {r2_rf:>15.4f} {r2_naive:>16.4f}")
print("=" * 55)

if rmse_rf < rmse_naive:
    print("Random Forest BEATS the naive baseline (lower RMSE is better).")
else:
    print("Random Forest does NOT beat the naive baseline.")

test_dates = X_test.index

fig_a = go.Figure()

fig_a.add_trace(go.Scatter(
    x=test_dates, y=y_test,
    mode='lines', name='Actual Close',
    line=dict(color='#2196F3', width=2)
))

fig_a.add_trace(go.Scatter(
    x=test_dates, y=y_pred_rf,
    mode='lines', name='RF Predicted',
    line=dict(color='#FF5722', width=2, dash='dash')
))

fig_a.update_layout(
    title='Random Forest: Predicted vs Actual Closing Price',
    xaxis_title='Date',
    yaxis_title='Closing Price (₹)',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

fig_a.write_html('plot_a_rf_vs_actual.html')
print("\nSaved: plot_a_rf_vs_actual.html")

fig_b = go.Figure()

fig_b.add_trace(go.Scatter(
    x=test_dates, y=y_test,
    mode='lines', name='Actual Close',
    line=dict(color='#2196F3', width=2)
))

fig_b.add_trace(go.Scatter(
    x=test_dates, y=y_pred_rf,
    mode='lines', name='RF Predicted',
    line=dict(color='#FF5722', width=2, dash='dash')
))

fig_b.add_trace(go.Scatter(
    x=test_dates, y=y_pred_naive,
    mode='lines', name='Naive Baseline',
    line=dict(color='#4CAF50', width=2, dash='dot')
))

fig_b.update_layout(
    title='Random Forest vs Naive Baseline vs Actual',
    xaxis_title='Date',
    yaxis_title='Closing Price (₹)',
    template='plotly_white',
    hovermode='x unified',
    legend=dict(x=0.01, y=0.99)
)

fig_b.write_html('plot_b_rf_vs_naive.html')
print("Saved: plot_b_rf_vs_naive.html")

fig_c = go.Figure()

fig_c.add_trace(go.Scatter(
    x=y_test, y=y_pred_rf,
    mode='markers', name='RF Predictions',
    marker=dict(color='#FF5722', size=6, opacity=0.7)
))

# Reference line

min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())

fig_c.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode='lines', name='Perfect Prediction',
    line=dict(color='gray', width=1, dash='dash')
))

fig_c.update_layout(
    title='Random Forest: Predicted vs Actual (Scatter)',
    xaxis_title='Actual Closing Price (₹)',
    yaxis_title='Predicted Closing Price (₹)',
    template='plotly_white',
    legend=dict(x=0.01, y=0.99)
)

fig_c.write_html('plot_c_rf_scatter.html')
print("Saved: plot_c_rf_scatter.html")

print("\nDone. All three charts saved as HTML files.")