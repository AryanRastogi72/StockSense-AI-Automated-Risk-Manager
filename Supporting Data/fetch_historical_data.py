import yfinance as yf
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Models"))

from Feature_Engineering import add_lag_features, add_indicators

DATA_DIR = PROJECT_ROOT / "Supporting Data"

def fetch_and_clean(ticker):
    nse_ticker = f"{ticker}.NS"
    print(f"Fetching 15 years of data for {nse_ticker}...")
    
    stock = yf.Ticker(nse_ticker)
    hist = stock.history(period="15y")
    
    if hist.empty:
        print(f"Error: No data found for {nse_ticker}")
        return
        
    hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    hist.index.name = "Date"
    
    # Save raw
    hist.to_csv(DATA_DIR / f"{ticker}_NS_15yr_yfinance.csv")
    print(f"Saved raw data to {ticker}_NS_15yr_yfinance.csv")
    
    # Clean
    hist = add_lag_features(hist)
    hist = add_indicators(hist)
    hist = hist.dropna()
    
    # Add Targets
    hist["Target_Close_Next"] = hist["Close"].shift(-1)
    hist = hist.dropna()
    
    hist["Target_Pct_Change"] = (hist["Target_Close_Next"] - hist["Close"]) / hist["Close"]
    
    def classify_target(pct):
        if pct > 0.005:
            return "Up"
        elif pct < -0.005:
            return "Down"
        else:
            return "Flat"
            
    hist["Target_Class_Next"] = hist["Target_Pct_Change"].apply(classify_target)
    
    # Save cleaned
    hist.to_csv(DATA_DIR / f"{ticker}_cleaned_data.csv")
    print(f"Saved cleaned data to {ticker}_cleaned_data.csv")

if __name__ == "__main__":
    for ticker in ["TCS", "RELIANCE"]:
        fetch_and_clean(ticker)
