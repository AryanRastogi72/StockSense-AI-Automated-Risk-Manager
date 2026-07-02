# Project Context: Stock-Ticker Prediction App (LTTS Internship)

A complete brief for getting oriented and starting work — share this alongside the full **Project Reference Document.docx** (theory/concepts) and the LTTS NSE CSV (sample data).

---

## What This Project Is

An internship project at L&T Technology Services (LTTS), combining **Web Dev + AI/ML + Data Science**.

**Deliverable:** a mobile app where a user selects a stock ticker, and the app predicts its short-term price trend (1–2 days ahead). LTTS (NSE: LTTS) is the running example/test case throughout — but the whole pipeline is built to generalize to any ticker.

## Problem Statement

**"Stock-Ticker with ML and DL — A Comparative Predictive Analysis"**

The core idea: don't commit to one algorithm upfront. Implement multiple ML and DL approaches on the *same* LTTS data with the *same* evaluation criteria, and let the results decide which one is actually worth deploying — rather than assuming a more complex (deep learning) model automatically wins.

**Algorithms being compared** (simplest → most complex):
- Traditional ML: Moving Average, Linear Regression, ARIMA, Random Forest
- Deep Learning: LSTM, GRU, Transformer

Full pros/cons for each are in the reference doc (Section 7).

## Data

**Source:** NSE's official daily price history (the same CSV format banks/funds pull from) — already sourced and verified for LTTS: 1 year, 246 trading days.

**Columns available:** Open/High/Low/Close/Prev Close, Total Traded Quantity (volume), Turnover, No. of Trades, Deliverable Qty, % Delivery.

**Known gotchas (verified, not theoretical):**
- File arrives sorted **newest-first** — must sort ascending by date before computing anything time-based, or the math runs backwards.
- Prices/volume are stored as **comma-formatted text** (`"3,324.40"`) — strip commas and convert to numeric first.
- There's a column literally named `Average Price` that is **not** the moving-average feature — it's that single day's own session average, not a multi-day rolling average. Compute the real moving average as a new column.
- Volume spans roughly 8,400 to 3.4 million across the year — likely needs a log-transform before modeling.

**Other free sources researched (for swapping tickers or backup):** yfinance (free, no key, but unofficial/rate-limited), Alpha Vantage (free tier, 25 requests/day), Kaggle (static snapshots — fine for early practice, not for live data).

## Technical Parameters (Model Inputs)

Raw price history doesn't go straight into a model. These are the actual engineered features:
- Recent closing prices (lag values)
- Daily return (% change day-to-day)
- Moving average (rolling mean over a window)
- Volatility (rolling std dev of returns)
- Volume

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python (whole project) |
| Traditional ML | scikit-learn |
| Deep learning | PyTorch *or* TensorFlow/Keras |
| Model serving | FastAPI |
| Quick test UI | Streamlit |
| Packaging | Docker |
| Final delivery | Mobile app calling the served API |

## Core Concepts Already Documented

The full reference doc covers, with examples and a connected method-mapping for each:
- Data classification hierarchy (structured/semi-structured/unstructured → which methods apply to each)
- AI vs. ML vs. Deep Learning
- Univariate/Multivariate/Covariate analysis; Overfitting/Underfitting/Good fit
- Time-series fundamentals + cross-industry examples
- Pros/cons comparison across all 7 algorithms being tested

## Current Status

✅ **Done (concept/research only — no code written yet):**
- All theory above
- Real LTTS data sourced and verified against the technical parameters

🔲 **Pending (nothing implemented yet):**
- Data pipeline (clean/load/feature-engineer the CSV)
- Training all 7 algorithms
- Evaluating and comparing them against a naive baseline
- Model saving, FastAPI service, Streamlit test UI, Docker packaging
- Mobile UI integration

## Suggested First Steps

1. Clean + load the LTTS CSV in pandas (sort ascending, strip commas, parse dates).
2. Compute the 5 technical parameters as real dataframe columns.
3. Build Moving Average + Linear Regression first — simplest, fastest sanity check, and the baseline every other model has to beat.
4. Then ARIMA + Random Forest.
5. Then LSTM → GRU → Transformer, in that order (each is a smaller step once the previous one works).
6. Evaluate all seven against each other and the naive baseline (tomorrow = today).
7. Wrap the winner: save model → FastAPI → Streamlit test → Docker → mobile integration.

No fixed split of who does what yet — decide between yourselves based on who wants the modeling side vs. the API/mobile side. Steps 1–6 are the data/ML track; the tail end of step 7 (FastAPI/Streamlit/Docker/mobile) leans more web-dev.

## Reference Materials

- **Project_Reference_Document.docx** — full theory, classification trees, algorithm comparisons, data notes
- **LTTS NSE CSV** — 1 year of verified daily price data, ready to use once the sort/format gotchas above are handled
