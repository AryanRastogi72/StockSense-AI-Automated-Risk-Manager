import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "Application" / "Backend"
MODELS_DIR = PROJECT_ROOT / "Models"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(MODELS_DIR))

import streamlit as st
import pandas as pd

from utils.live_features import get_live_features
from utils.model_loader import (
    SUPPORTED_TICKERS,
    VALID_ALGORITHMS,
    VALID_TASKS,
    VALID_TUNINGS,
    validate_variant,
    predict_tree,
    predict_lstm,
    generate_plotly_graph,
)
from utils.risk_utils import compute_risk_for_ticker, compute_portfolio_risk
from utils.explainability import explain_regression, explain_classification
from dashboard_ui import render_risk_dashboard, render_portfolio_dashboard

ALGO_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM (Deep Learning)"}
TASK_LABELS = {"reg": "Regression (Price)", "class": "Classification (Direction)"}
TUNING_LABELS = {"baseline": "Baseline (Default)", "tuned": "Tuned (Optimized)"}

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-card.highlight {
        border-color: rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
    }
    .metric-card.highlight .value {
        background: linear-gradient(135deg, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header"><h1>📈 StockSense AI</h1>'
    "<p>Real-time predictions powered by 12 trained ML & DL models</p></div>",
    unsafe_allow_html=True,
)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None
if "risk_result" not in st.session_state:
    st.session_state.risk_result = None
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None

with st.sidebar:
    st.header("⚙️ Model Configuration")
    st.caption("Select the model parameters below and hit Predict.")

    ticker = st.selectbox("Asset", SUPPORTED_TICKERS, format_func=lambda t: f"Larsen & Toubro ({t}.NS)")

    algorithm = st.selectbox("Algorithm", VALID_ALGORITHMS, format_func=lambda a: ALGO_LABELS[a])

    task = st.selectbox("Prediction Task", VALID_TASKS, format_func=lambda t: TASK_LABELS[t])

    tuning = st.selectbox("Model Tuning", list(reversed(VALID_TUNINGS)), format_func=lambda t: TUNING_LABELS[t])

    predict_clicked = st.button("🚀 Generate Prediction", use_container_width=True, type="primary")

    st.markdown("---")
    compare_clicked = st.button("⚔️ Compare All 12 Models", use_container_width=True)
    
    st.markdown("---")
    risk_clicked = st.button("🛡️ Risk & Explainability", use_container_width=True, type="primary")

    st.markdown("---")
    portfolio_clicked = st.button("🌐 Portfolio Risk View", use_container_width=True)

if predict_clicked:
    st.session_state.comparison_result = None
    st.session_state.risk_result = None
    st.session_state.portfolio_result = None
    valid, error_msg = validate_variant(ticker, algorithm, task, tuning)
    if not valid:
        st.error(f"❌ {error_msg}")
    else:
        with st.spinner("Fetching live data from Yahoo Finance and running prediction..."):
            try:
                live_data = get_live_features(ticker)
            except ValueError as exc:
                st.error(f"❌ Data Error: {exc}")
                st.stop()

            try:
                if algorithm in ("rf", "xgb"):
                    result = predict_tree(ticker, algorithm, task, tuning, live_data["latest_row"])
                else:
                    result = predict_lstm(ticker, task, tuning, live_data["sequence"])
            except FileNotFoundError as exc:
                st.error(f"❌ Model not found. Run the training script first. ({exc})")
                st.stop()

        st.session_state.prediction_result = {
            "algorithm": algorithm,
            "task": task,
            "tuning": tuning,
            "ticker": ticker,
            "result": result,
            "live_data": live_data,
        }

if compare_clicked:
    st.session_state.prediction_result = None
    st.session_state.risk_result = None
    st.session_state.portfolio_result = None
    with st.spinner("Running all 12 models on live data... this may take a moment."):
        try:
            live_data = get_live_features(ticker)
        except ValueError as exc:
            st.error(f"❌ Data Error: {exc}")
            st.stop()

        rows = []
        for algo in VALID_ALGORITHMS:
            for t in VALID_TASKS:
                for tune in VALID_TUNINGS:
                    try:
                        if algo in ("rf", "xgb"):
                            res = predict_tree(ticker, algo, t, tune, live_data["latest_row"])
                        else:
                            res = predict_lstm(ticker, t, tune, live_data["sequence"])

                        if t == "reg":
                            prediction = f"₹ {res['predicted_close']:.2f}"
                        else:
                            prediction = res["predicted_direction"]

                        rows.append({
                            "Algorithm": ALGO_LABELS[algo],
                            "Task": TASK_LABELS[t],
                            "Tuning": TUNING_LABELS[tune],
                            "Prediction": prediction,
                            "Status": "✅",
                        })
                    except Exception as exc:
                        rows.append({
                            "Algorithm": ALGO_LABELS[algo],
                            "Task": TASK_LABELS[t],
                            "Tuning": TUNING_LABELS[tune],
                            "Prediction": "—",
                            "Status": f"❌ {exc}",
                        })

        st.session_state.comparison_result = {
            "rows": rows,
            "last_date": live_data["last_date"],
            "ticker": ticker,
        }

if risk_clicked:
    st.session_state.prediction_result = None
    st.session_state.comparison_result = None
    st.session_state.portfolio_result = None
    with st.spinner("Computing Risk & Explainability..."):
        try:
            live_data = get_live_features(ticker)
            risk = compute_risk_for_ticker(ticker, live_data)
            
            # For explanation, we need the predicted class from the stack (if class)
            pred_class = risk["classification"]["predicted_direction"]
            
            explanations = {
                "reg": explain_regression(ticker, live_data),
                "class": explain_classification(ticker, live_data, pred_class)
            }
            
            st.session_state.risk_result = {
                "ticker": ticker,
                "live_data": live_data,
                "risk": risk,
                "explanations": explanations
            }
        except Exception as exc:
            st.error(f"❌ Error computing risk dashboard: {exc}")
            st.stop()

if portfolio_clicked:
    st.session_state.prediction_result = None
    st.session_state.comparison_result = None
    st.session_state.risk_result = None
    with st.spinner("Computing Portfolio Risk..."):
        try:
            # We only have 'LT' right now, but the API supports list
            live_data_map = {"LT": get_live_features("LT")}
            port_risk = compute_portfolio_risk(["LT"], live_data_map)
            st.session_state.portfolio_result = port_risk
        except Exception as exc:
            st.error(f"❌ Error computing portfolio risk: {exc}")
            st.stop()

pred = st.session_state.prediction_result
if pred is not None:
    model_label = f"{ALGO_LABELS[pred['algorithm']]} — {TUNING_LABELS[pred['tuning']]}"
    live_data = pred["live_data"]
    result = pred["result"]
    task_val = pred["task"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Model Deployed</div>'
            f'<div class="value">{model_label}</div></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Last Data Date</div>'
            f'<div class="value">{live_data["last_date"]}</div></div>',
            unsafe_allow_html=True,
        )

    with col3:
        if task_val == "reg":
            pred_display = f"₹ {result['predicted_close']:.2f}"
            pred_label = "Predicted Close"
        else:
            pred_display = result["predicted_direction"]
            pred_label = "Predicted Direction"

        st.markdown(
            f'<div class="metric-card highlight"><div class="label">{pred_label}</div>'
            f'<div class="value">{pred_display}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    fig = generate_plotly_graph(
        pred["ticker"], pred["algorithm"], task_val, pred["tuning"], live_data, result
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, width="stretch")

comp = st.session_state.comparison_result
if comp is not None:
    st.markdown("---")
    st.subheader(f"⚔️ All 12 Models — {comp['ticker']} (Data as of {comp['last_date']})")

    df = pd.DataFrame(comp["rows"])

    reg_df = df[df["Task"] == TASK_LABELS["reg"]]
    class_df = df[df["Task"] == TASK_LABELS["class"]]

    col_reg, col_class = st.columns(2)

    with col_reg:
        st.markdown("##### 💰 Regression Models (Predicted Price)")
        st.dataframe(
            reg_df[["Algorithm", "Tuning", "Prediction", "Status"]],
            use_container_width=True,
            hide_index=True,
        )

    with col_class:
        st.markdown("##### 🧭 Classification Models (Predicted Direction)")
        st.dataframe(
            class_df[["Algorithm", "Tuning", "Prediction", "Status"]],
            use_container_width=True,
            hide_index=True,
        )

risk_res = st.session_state.get("risk_result")
if risk_res is not None:
    render_risk_dashboard(risk_res)

port_res = st.session_state.get("portfolio_result")
if port_res is not None:
    render_portfolio_dashboard(port_res)

if pred is None and comp is None and risk_res is None and port_res is None:
    st.info("👈 Configure your model in the sidebar and click **Generate Prediction** to get started, or explore the Risk dashboards.")
