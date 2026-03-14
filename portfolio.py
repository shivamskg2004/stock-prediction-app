import json
import os

PORTFOLIO_FILE = "portfolio.json"


def load_portfolio():
    """Load portfolio from JSON file."""
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except:
        return []


def save_portfolio(portfolio):
    """Save portfolio to JSON file."""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


def add_stock(symbol, quantity, buy_price):
    """Add a stock to portfolio."""
    portfolio = load_portfolio()
    # Check if stock already exists
    for item in portfolio:
        if item["symbol"] == symbol:
            # Update existing - average the buy price
            total_qty = item["quantity"] + quantity
            item["buy_price"] = round(
                (item["buy_price"] * item["quantity"] + buy_price * quantity) / total_qty, 2
            )
            item["quantity"] = total_qty
            save_portfolio(portfolio)
            return portfolio
    # Add new stock
    portfolio.append({
        "symbol": symbol,
        "quantity": quantity,
        "buy_price": buy_price,
    })
    save_portfolio(portfolio)
    return portfolio


def remove_stock(symbol):
    """Remove a stock from portfolio."""
    portfolio = load_portfolio()
    portfolio = [p for p in portfolio if p["symbol"] != symbol]
    save_portfolio(portfolio)
    return portfolio


def get_portfolio_with_live_prices():
    """Fetch live prices for all portfolio stocks."""
    import yfinance as yf
    portfolio = load_portfolio()
    if not portfolio:
        return [], 0, 0, 0

    results = []
    total_invested = 0
    total_current = 0

    for item in portfolio:
        symbol = item["symbol"]
        qty = item["quantity"]
        buy_price = item["buy_price"]

        # Fetch live price
        try:
            sym = symbol if "." in symbol else symbol + ".NS"
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
            else:
                current_price = buy_price
        except:
            current_price = buy_price

        invested = round(qty * buy_price, 2)
        current_val = round(qty * current_price, 2)
        pnl = round(current_val - invested, 2)
        pnl_pct = round((pnl / invested) * 100, 2) if invested > 0 else 0

        total_invested += invested
        total_current += current_val

        results.append({
            "symbol": symbol,
            "quantity": qty,
            "buy_price": buy_price,
            "current_price": round(current_price, 2),
            "invested": invested,
            "current_value": current_val,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    total_pnl = round(total_current - total_invested, 2)
    total_pnl_pct = round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0

    return results, round(total_invested, 2), round(total_current, 2), total_pnl, total_pnl_pct
