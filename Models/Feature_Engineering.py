import pandas as pd
import numpy as np
import pandas_ta_classic as ta

TICKER = "LT"
FLAT_THRESHOLD = 0.005


def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def remove_holiday_rows(df):
    same_price = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
    no_volume = df["Volume"] == 0
    return df[~(same_price & no_volume)]


def add_lag_features(df):
    for lag in range(1, 6):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
    return df


def add_indicators(df):
    df["Daily_Return"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Volatility_20"] = df["Close"].rolling(20).std()
    df["Log_Volume"] = np.log1p(df["Volume"])
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]

    df["ROC_10"] = ta.roc(df["Close"], length=10)

    bbands = ta.bbands(df["Close"], length=20, std=2)
    df["BB_Lower"] = bbands["BBL_20_2.0"]
    df["BB_Middle"] = bbands["BBM_20_2.0"]
    df["BB_Upper"] = bbands["BBU_20_2.0"]

    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["Day_of_Week"] = df.index.dayofweek
    return df


def add_targets(df, threshold):
    df["Target_Close_Next"] = df["Close"].shift(-1)
    df["Target_Pct_Change"] = (df["Target_Close_Next"] - df["Close"]) / df["Close"]

    conditions = [
        df["Target_Pct_Change"] > threshold,
        df["Target_Pct_Change"] < -threshold,
    ]
    choices = ["Up", "Down"]
    df["Target_Class_Next"] = np.select(conditions, choices, default="Flat")
    df.loc[df["Target_Close_Next"].isna(), "Target_Class_Next"] = np.nan
    return df


if __name__ == "__main__":
    df = load_data(f"{TICKER}_NS_15yr_yfinance.csv")
    print("Raw rows:", len(df))

    df = remove_holiday_rows(df)
    print("After removing holiday rows:", len(df))

    df = add_lag_features(df)
    df = add_indicators(df)
    df = add_targets(df, FLAT_THRESHOLD)

    df = df.dropna()
    print("After dropping warm-up rows:", len(df))
    print("Date range:", df.index.min().date(), "to", df.index.max().date())
    print()
    print(df["Target_Class_Next"].value_counts())

    df.to_csv(f"{TICKER}_cleaned_data.csv")
