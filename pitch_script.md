# StockSense AI - 5-Minute Pitch Script

### [0:00 - 0:30] The Problem
*(Camera: Center framed, talking directly to camera)*
"Hi, I'm presenting StockSense AI, an AI Risk Manager built for Track 2 of the Razorpay AI Builder Internship. 

In modern fintech, machine learning models output point predictions all the time—a price target, a binary 'buy' or 'sell'. And downstream systems treat those outputs as certainties. But models aren't certain. They have varying levels of confidence, they disagree with each other, and they degrade when the market enters volatile regimes. 

A real risk manager doesn't just ask 'what is the prediction?'. They ask 'how much should I trust this prediction right now?' That's the exact problem StockSense AI solves."

### [0:30 - 1:15] The Foundation
*(Screen Recording: Show the terminal running the model pipeline, specifically the RandomizedSearchCV progress bars)*
"To build a risk manager, I first needed a robust forecasting engine to manage. I built a comprehensive pipeline covering three major Indian equities: L&T, TCS, and Reliance. 

For each asset, the system trains 12 distinct models spanning Random Forests, XGBoost, and PyTorch LSTMs, simultaneously evaluating both regression price targets and directional classification. This isn't a toy dataset—it's trained and rigorously validated on 15 years of daily market data. The models achieve solid baseline performance, like a 57 RMSE on L&T, and 37% directional accuracy—which outperforms the majority baseline."

### [1:15 - 2:30] The Risk Architecture
*(Screen Recording: Display the `per_ticker_selection.md` tables briefly, then switch to the architecture diagram)*
"But having a prediction is just step one. Here's what the AI Risk Manager actually does with it.

First, I implemented a Meta-Learner Stack trained on Out-Of-Fold predictions to avoid data leakage. Interestingly, the stack only reliably beat the base models for L&T classification. For Reliance, a standalone LSTM actually performed better. Instead of forcing a complex architecture, I built a dynamic routing system that objectively serves the best architecture per-asset based on a strict 0.5% improvement margin.

Next is the Risk-Scoring Layer. The system runs an Isolation Forest against current market volatility to detect anomalous regimes. If the market is calm, risk relies on empirical Value-at-Risk and model disagreement. If the Isolation Forest triggers an anomaly, risk scales aggressively. 

Finally, every prediction routes through an Explainability Layer. Depending on which model the system dynamically selected, it extracts live SHAP values natively for tree models, reconstructs weighted SHAP for the ensemble stack, or utilizes a high-fidelity Decision Tree surrogate to crack open the LSTM."

### [2:30 - 4:00] Live Demo Walkthrough
*(Director's Note: Have the Streamlit dashboard open to the Single Ticker view for LT.NS)*

"Let's look at the live dashboard. I've selected L&T. Right at the top, you don't just see 'Up' or 'Down'. You see exactly which model generated this prediction—in this case, the Stacked Meta-Learner."

*(Director's Note: Scroll down to the Risk Breakdown section)*
"Here is the risk score breakdown. You can see the Base Risk, and right next to it, the Regime Multiplier. Because current volatility is normal, the multiplier is 1.0. But if a macro shock hit, this Isolation Forest gate would instantly throttle our confidence."

*(Director's Note: Switch tab to the SHAP Explainability view)*
"Because L&T uses the Stacked model, this SHAP chart isn't just looking at one algorithm. It's a coefficient-weighted reconstruction explaining the exact features driving the entire ensemble's logic."

*(Director's Note: Click over to the Portfolio View in the sidebar, selecting LT, TCS, and RELIANCE)*
"Finally, risk isn't isolated. In the Portfolio View, we aggregate the predictions. But rather than just summing up individual Value-at-Risk, the backend computes a live 15-year cross-asset correlation matrix. The total Portfolio VaR you see here mathematically reflects actual diversification benefits across the three assets."

### [4:00 - 5:00] Closing
*(Camera: Back to center framed, talking directly to camera)*
"The architecture here is built specifically for stock forecasting, but the framework generalizes to any AI pipeline. Whether you're predicting credit default, fraud, or loan approval, you shouldn't just be shipping raw predictions to production. You need a gatekeeper calculating disagreement, measuring VaR, and dynamically explaining the output.

StockSense AI bridges the gap between building an accurate model, and building a model you can actually trust with capital. Thank you."
