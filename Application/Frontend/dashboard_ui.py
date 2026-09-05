import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd

Z_95_ONE_TAILED = 1.645
GAUGE_HIGH_THRESHOLD = 70
GAUGE_MED_THRESHOLD = 40
GAUGE_MAX = 100
GAUGE_HEIGHT = 250
TOP_SHAP_FEATURES = 5
PRICE_HISTORY_TAIL = 60
SURROGATE_FIDELITY_THRESHOLD = 0.8


def _gauge_color(score: float) -> str:
    if score > GAUGE_HIGH_THRESHOLD:
        return "red"
    if score > GAUGE_MED_THRESHOLD:
        return "orange"
    return "green"


def render_risk_dashboard(risk_res):
    st.markdown("---")
    ticker = risk_res["ticker"]
    risk = risk_res["risk"]
    expl = risk_res["explanations"]
    regime = risk["regime"]

    st.subheader(f"🛡️ Risk & Explainability Dashboard — {ticker}")

    col_reg, col_class = st.columns(2)

    with col_reg:
        st.markdown("### 💰 Regression (Price)")
        st.markdown(f"**Predicted Close (XGBoost):** ₹ {risk['regression']['predicted_close_xgb']:.2f}")

        reg_risk = risk["regression"]
        reg_final = reg_risk["final_risk"]
        fig_gauge_reg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=reg_final,
            title={'text': "Risk Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, GAUGE_MAX]},
                'bar': {'color': _gauge_color(reg_final)}
            }
        ))
        fig_gauge_reg.update_layout(height=GAUGE_HEIGHT, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge_reg, use_container_width=True)

        with st.expander("🔍 Risk Breakdown"):
            st.write(f"**Base Risk:** {reg_risk['base_risk']:.1f}")
            st.write(f"**Regime Multiplier:** {regime['regime_multiplier']:.2f}x")
            st.write(f"**VaR Contribution:** {reg_risk['var_score']:.1f}")
            st.write(f"**Disagreement:** {reg_risk['disagreement_score']:.1f}")
            if reg_risk.get("low_confidence"):
                st.warning("Low Confidence: Insufficient price history for stable VaR.")
            if regime.get("outlier_gate"):
                st.error("Outlier Gate Triggered: Extreme market conditions detected.")

        st.markdown("#### Top SHAP Features")
        shap_reg = pd.Series(expl["reg"]["feature_impacts"]).head(TOP_SHAP_FEATURES)
        st.bar_chart(shap_reg)

    with col_class:
        st.markdown("### 🧭 Classification (Direction)")
        st.markdown(f"**Predicted Direction (Stack):** {risk['classification']['predicted_direction']}")

        cls_risk = risk["classification"]
        cls_final = cls_risk["final_risk"]
        fig_gauge_cls = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cls_final,
            title={'text': "Risk Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, GAUGE_MAX]},
                'bar': {'color': _gauge_color(cls_final)}
            }
        ))
        fig_gauge_cls.update_layout(height=GAUGE_HEIGHT, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge_cls, use_container_width=True)

        with st.expander("🔍 Risk Breakdown"):
            st.write(f"**Base Risk:** {cls_risk['base_risk']:.1f}")
            st.write(f"**Regime Multiplier:** {regime['regime_multiplier']:.2f}x")
            st.write(f"**Entropy Contribution:** {cls_risk['entropy_score']:.1f}")
            st.write(f"**Disagreement:** {cls_risk['disagreement_score']:.1f}")
            if regime.get("outlier_gate"):
                st.error("Outlier Gate Triggered: Extreme market conditions detected.")

        st.markdown("#### Top SHAP Features (Stack Logit)")
        if expl["class"].get("low_fidelity_warning"):
            st.warning(f"LSTM Surrogate Proxy has low fidelity (R² = {expl['class']['lstm_surrogate_fidelity_r2']:.2f}). SHAP values for LSTM are included but may be noisy.")
        shap_cls = pd.Series(expl["class"]["feature_impacts_logit"]).head(TOP_SHAP_FEATURES)
        st.bar_chart(shap_cls)

    # VaR Band Chart
    st.markdown("### 📉 Recent Price History with VaR Band")
    df = risk_res["live_data"]["raw_df"].tail(PRICE_HISTORY_TAIL)
    pred_price = risk['regression']['predicted_close_xgb']

    current_vol = reg_risk.get("current_volatility", 0.0)
    var_lower = pred_price * (1 - Z_95_ONE_TAILED * current_vol)
    var_upper = pred_price * (1 + Z_95_ONE_TAILED * current_vol)

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical Close', line=dict(color='blue')))
    next_day = df.index[-1] + pd.Timedelta(days=1)
    fig_var.add_trace(go.Scatter(x=[next_day], y=[pred_price], name='Prediction', marker=dict(color='red', size=10)))

    fig_var.add_trace(go.Scatter(
        x=[df.index[-1], next_day, next_day, df.index[-1]],
        y=[df['Close'].iloc[-1], var_upper, var_lower, df['Close'].iloc[-1]],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name='95% VaR Band'
    ))
    st.plotly_chart(fig_var, use_container_width=True)


def render_portfolio_dashboard(port_res):
    st.markdown("---")
    st.subheader("🌐 Portfolio Risk View")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total VaR (95%)", f"{port_res['portfolio_var_95']:.6f}")
    with col2:
        st.metric("Portfolio Volatility", f"{port_res['portfolio_volatility']:.6f}")

    st.markdown("### Correlation Matrix")
    corr_df = pd.DataFrame(port_res['correlation_matrix'])
    fig = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Individual Asset Risk")
    asset_rows = []
    for t, ticker_risk in port_res["individual_risks"].items():
        asset_rows.append({
            "Ticker": t,
            "Predicted Price": f"₹ {ticker_risk['regression']['predicted_close_xgb']:.2f}",
            "Final Risk (Reg)": f"{ticker_risk['regression']['final_risk']:.1f}",
            "Final Risk (Class)": f"{ticker_risk['classification']['final_risk']:.1f}",
        })
    st.dataframe(pd.DataFrame(asset_rows), hide_index=True, use_container_width=True)
