# 📈 StockSense India — AI Stock Market Prediction App

> **ML-powered Indian stock market analysis web app** with real-time NSE/BSE data, 30-day price forecasting, technical indicators, buy/sell signals, and a portfolio tracker.

---

## 🚀 Live Demo
> Coming soon — deploying on Render.com

---

## 📸 Screenshots

### 🏠 Home Page
![Home Page](screenshots/home.png)

### 📊 Stock Analysis
![Stock Analysis](screenshots/analysis.png)

### 💼 Portfolio Tracker
![Portfolio](screenshots/portfolio.png)

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 **Stock Search** | Search any NSE/BSE listed Indian stock by symbol |
| 🤖 **ML Prediction** | Random Forest model predicts 30-day future prices |
| 📊 **Technical Indicators** | RSI, MACD, Bollinger Bands, MA20, MA50 |
| 🎯 **Buy/Sell/Hold Signals** | AI-powered trading recommendations |
| 📈 **Interactive Charts** | 4 Plotly charts — Price, RSI, MACD, Volume |
| 💼 **Portfolio Tracker** | Track your holdings with live P&L |
| 🇮🇳 **Indian Market Focus** | Auto NSE (.NS) and BSE (.BO) support |
| 💹 **Live Prices** | Real-time data via yfinance API |

---

## 🛠️ Tech Stack

**Backend**
- Python 3.x
- Flask — Web framework
- yfinance — Real-time NSE/BSE stock data
- scikit-learn — Random Forest ML model
- pandas & numpy — Data processing

**Frontend**
- HTML5, CSS3, JavaScript
- Plotly.js — Interactive charts
- Google Fonts (Outfit + JetBrains Mono)

---

## 📁 Project Structure

```
stock_prediction_project/
│
├── app.py              # Flask web app + full UI (HTML/CSS/JS)
├── data_fetch.py       # Stock data fetching + technical indicators
├── model.py            # ML model training + price prediction
├── portfolio.py        # Portfolio tracker logic
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stock-prediction-app.git
cd stock-prediction-app
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment
```bash
# Windows (Command Prompt)
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
python app.py
```

### 6. Open in browser
```
http://127.0.0.1:5000
```

---

## 📊 How It Works

```
User enters stock symbol (e.g. TCS)
        ↓
yfinance fetches 2 years of NSE data
        ↓
Technical indicators calculated (RSI, MACD, BB)
        ↓
Random Forest model trained on price history
        ↓
Model predicts next 30 days of prices
        ↓
Buy/Sell/Hold signal generated
        ↓
Results displayed with interactive charts
```

---

## 📈 Supported Stocks

Any stock listed on **NSE** (National Stock Exchange) or **BSE** (Bombay Stock Exchange).

**Popular examples:**

| Company | Symbol |
|---------|--------|
| Reliance Industries | RELIANCE |
| Tata Consultancy Services | TCS |
| Infosys | INFY |
| HDFC Bank | HDFCBANK |
| State Bank of India | SBIN |
| Wipro | WIPRO |
| ICICI Bank | ICICIBANK |
| Bajaj Finance | BAJFINANCE |
| Tata Motors | TATAMOTORS |
| Adani Enterprises | ADANIENT |

---

## 🤖 ML Model Details

- **Algorithm:** Random Forest Regressor (100 estimators)
- **Features:** Historical closing prices (20-day window)
- **Training Split:** 85% train / 15% test
- **Accuracy Metric:** R² Score
- **Forecast Horizon:** 30 business days
- **Data Source:** Yahoo Finance via yfinance

> ⚠️ **Disclaimer:** This app is for educational purposes only. Stock predictions are not financial advice. Always do your own research before investing.

---

## 💼 Portfolio Tracker

- Add stocks with **symbol, quantity, and buy price**
- View **live current price** from NSE
- Track **P&L (Profit & Loss)** in ₹ and %
- See **total portfolio value** vs invested amount
- Click **ANALYZE** on any holding to view detailed charts
- Data saved locally in `portfolio.json`

---

## 🔮 Future Improvements

- [ ] LSTM deep learning model for better accuracy
- [ ] News sentiment analysis
- [ ] More technical indicators (VWAP, ATR, Stochastic)
- [ ] Email/SMS price alerts
- [ ] Multi-user authentication
- [ ] Deploy on cloud (Render / Railway)

---

## 👨‍💻 Author

**Shivam Kumar Gupta**
- GitHub: [@shivamskg2004](https://github.com/shivamskg2004)
- LinkedIn: [Shivam Kumar Gupta](https://www.linkedin.com/in/shivam-kumar-gupta-23b48731b)
---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## ⭐ If you found this helpful, please give it a star!

> Built with ❤️ for learning Data Science and Web Development
