import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


def prepare_features(df, window=10):
    """Create feature matrix from historical price data."""
    data = df["Close"].values
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_and_predict(df, days=30):
    """Train Random Forest model and predict future prices."""
    df = df.dropna(subset=["Close"])
    prices = df["Close"].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    window = 20
    X, y = prepare_features(pd.DataFrame({"Close": scaled}), window=window)

    if len(X) < 40:
        return None, None, "Not enough data to train the model."

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy on test set
    y_pred_test = model.predict(X_test)
    # Use R² score instead of MAPE for stability
    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-9))
    accuracy = round(max(0, min(r2, 1)) * 100, 2)

    # Predict future
    future_scaled = []
    last_window = list(scaled[-window:])
    for _ in range(days):
        x_input = np.array(last_window[-window:]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        future_scaled.append(pred)
        last_window.append(pred)

    future_prices = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=days, freq="B")

    prediction_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted": future_prices
    })

    return prediction_df, accuracy, None


def get_signal(df, prediction_df):
    """Generate buy/sell/hold signal."""
    current_price = df["Close"].iloc[-1]
    predicted_price = prediction_df["Predicted"].iloc[-1]
    rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 50
    ma20 = df["MA20"].iloc[-1] if "MA20" in df.columns else current_price
    ma50 = df["MA50"].iloc[-1] if "MA50" in df.columns else current_price

    change_pct = (predicted_price - current_price) / current_price * 100

    score = 0

    # Price prediction signal
    if change_pct > 3:
        score += 2
    elif change_pct > 1:
        score += 1
    elif change_pct < -3:
        score -= 2
    elif change_pct < -1:
        score -= 1

    # RSI signal
    if rsi < 35:
        score += 1  # oversold = buy
    elif rsi > 65:
        score -= 1  # overbought = sell

    # MA crossover
    if ma20 > ma50:
        score += 1
    else:
        score -= 1

    if score >= 2:
        signal = "BUY"
        color = "#00e676"
        reason = f"Model predicts +{change_pct:.1f}% growth. RSI and moving averages support upward momentum."
    elif score <= -2:
        signal = "SELL"
        color = "#ff1744"
        reason = f"Model predicts {change_pct:.1f}% decline. Technical indicators suggest downward pressure."
    else:
        signal = "HOLD"
        color = "#ffab00"
        reason = f"Mixed signals. Predicted change: {change_pct:.1f}%. Monitor closely before acting."

    return {
        "signal": signal,
        "color": color,
        "reason": reason,
        "predicted_change": round(change_pct, 2),
        "current_price": round(float(current_price), 2),
        "target_price": round(float(predicted_price), 2),
        "rsi": round(float(rsi), 1) if not np.isnan(rsi) else 50,
    }
