import yfinance as yf
import pandas as pd
import numpy as np


def get_stock_data(symbol, period="2y"):
    """Fetch historical stock data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None, "No data found for this symbol."
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        return df, None
    except Exception as e:
        return None, str(e)


def get_stock_info(symbol):
    """Fetch company info."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "currency": info.get("currency", "USD"),
        }
    except:
        return {"name": symbol, "sector": "N/A"}


def compute_indicators(df):
    """Compute technical indicators."""
    df = df.copy()

    # Moving Averages
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # Bollinger Bands
    df["BB_mid"] = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * std
    df["BB_lower"] = df["BB_mid"] - 2 * std

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Volume MA
    df["Vol_MA20"] = df["Volume"].rolling(window=20).mean()

    return df
