import sys
from pathlib import Path

import yfinance as yf

from utils.paths import MODELS_DIR
sys.path.insert(0, str(MODELS_DIR))

from Feature_Engineering import add_lag_features, add_indicators


HISTORY_PERIOD = "6mo"


def get_live_features(ticker):
    nse_ticker = f"{ticker}.NS"

    try:
        stock = yf.Ticker(nse_ticker)
        hist = stock.history(period=HISTORY_PERIOD)
    except Exception as exc:
        raise ValueError(f"Failed to fetch data from yfinance for '{ticker}': {exc}")

    if hist.empty:
        raise ValueError(
            f"yfinance returned no data for '{ticker}' (tried '{nse_ticker}'). "
            "Check that the ticker is valid and that you have network access."
        )

    hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    hist.index.name = "Date"

    hist = add_lag_features(hist)
    hist = add_indicators(hist)
    hist = hist.dropna()

    if len(hist) < 30:
        raise ValueError(
            f"Not enough data rows after computing indicators for '{ticker}'. "
            f"Got {len(hist)} rows, need at least 30 for LSTM lookback."
        )

    feature_cols = [c for c in hist.columns if c not in [
        "Target_Close_Next", "Target_Pct_Change", "Target_Class_Next"
    ]]

    latest_row = hist[feature_cols].iloc[[-1]]
    last_date = hist.index[-1].strftime("%Y-%m-%d")
    latest_close = hist["Close"].iloc[-1]

    sequence = hist[feature_cols].iloc[-30:]

    return {
        "latest_row": latest_row,
        "sequence": sequence,
        "last_date": last_date,
        "latest_close": latest_close,
        "feature_cols": list(feature_cols),
        "raw_df": hist,
    }
