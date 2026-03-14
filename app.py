from flask import Flask, render_template_string, jsonify, request
import numpy as np
import pandas as pd
from data_fetch import get_stock_data, get_stock_info, compute_indicators
from model import train_and_predict, get_signal
from portfolio import add_stock, remove_stock, get_portfolio_with_live_prices

app = Flask(__name__)

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>StockSense India — AI Market Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
:root {
  --bg:#04050a; --bg2:#080c14; --surface:#0c1220; --surface2:#111827;
  --border:#1a2535; --border2:#243047;
  --gold:#f5a623; --gold2:#fbbf24; --gold-dim:rgba(245,166,35,0.12);
  --green:#10b981; --green-dim:rgba(16,185,129,0.12);
  --red:#ef4444; --red-dim:rgba(239,68,68,0.12);
  --blue:#3b82f6; --purple:#8b5cf6;
  --text:#f1f5f9; --text2:#94a3b8; --text3:#475569;
  --font:'Outfit',sans-serif; --mono:'JetBrains Mono',monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;inset:0;z-index:0;background:radial-gradient(ellipse 80% 50% at 20% 0%,rgba(245,166,35,0.06) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 100%,rgba(59,130,246,0.05) 0%,transparent 60%);pointer-events:none;}
body::after{content:'';position:fixed;inset:0;z-index:0;background-image:linear-gradient(rgba(245,166,35,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(245,166,35,0.025) 1px,transparent 1px);background-size:60px 60px;pointer-events:none;}
::-webkit-scrollbar{width:5px;} ::-webkit-scrollbar-track{background:var(--bg);} ::-webkit-scrollbar-thumb{background:var(--border2);border-radius:10px;}

/* HEADER */
header{position:sticky;top:0;z-index:100;background:rgba(4,5,10,0.88);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 48px;height:68px;display:flex;align-items:center;justify-content:space-between;}
.logo-wrap{display:flex;align-items:center;gap:12px;}
.logo-icon{width:38px;height:38px;background:linear-gradient(135deg,var(--gold),#e07b00);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;font-weight:900;color:#000;box-shadow:0 0 24px rgba(245,166,35,0.4);}
.logo-text{font-size:1.25rem;font-weight:800;letter-spacing:-0.3px;}
.logo-text span{color:var(--gold);}
.logo-sub{font-family:var(--mono);font-size:0.58rem;color:var(--text3);letter-spacing:2px;margin-top:1px;}
.nav-tabs{display:flex;gap:4px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:4px;}
.nav-tab{font-family:var(--font);font-size:0.82rem;font-weight:600;padding:8px 22px;border-radius:8px;border:none;cursor:pointer;transition:all 0.2s;color:var(--text2);background:transparent;letter-spacing:0.3px;}
.nav-tab:hover{color:var(--text);background:var(--border);}
.nav-tab.active{background:linear-gradient(135deg,var(--gold),#e07b00);color:#000;font-weight:700;box-shadow:0 2px 14px rgba(245,166,35,0.35);}
.header-right{display:flex;align-items:center;gap:20px;}
.market-badge{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:0.65rem;color:var(--text2);background:var(--surface);border:1px solid var(--border);padding:7px 14px;border-radius:10px;}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:0.3;}}
#clock{font-family:var(--mono);font-size:0.65rem;color:var(--text3);}

/* HERO */
#analyze-page{position:relative;z-index:1;}
.hero{max-width:880px;margin:0 auto;padding:80px 24px 56px;text-align:center;}
.hero-eyebrow{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:0.68rem;color:var(--gold);background:var(--gold-dim);border:1px solid rgba(245,166,35,0.25);padding:7px 18px;border-radius:100px;letter-spacing:2px;margin-bottom:28px;text-transform:uppercase;}
.hero-title{font-size:clamp(2.4rem,5vw,3.8rem);font-weight:900;letter-spacing:-2px;line-height:1.05;margin-bottom:18px;}
.hero-title .hl{background:linear-gradient(135deg,var(--gold),var(--gold2),#fff8e1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:1rem;color:var(--text2);font-weight:400;line-height:1.75;max-width:540px;margin:0 auto 44px;}

/* SEARCH */
.search-wrap-outer{max-width:680px;margin:0 auto;}
.search-box{background:var(--surface);border:1px solid var(--border2);border-radius:16px;padding:7px;display:flex;gap:8px;align-items:center;box-shadow:0 20px 60px rgba(0,0,0,0.4),0 0 0 1px rgba(245,166,35,0.08);transition:box-shadow 0.3s;}
.search-box:focus-within{box-shadow:0 20px 60px rgba(0,0,0,0.5),0 0 0 2px rgba(245,166,35,0.25);}
.search-icon{font-family:var(--mono);font-size:1.1rem;font-weight:700;color:var(--gold);background:var(--gold-dim);border:none;border-radius:10px;width:44px;height:44px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
#ticker-input{flex:1;background:transparent;border:none;outline:none;color:var(--text);font-family:var(--mono);font-size:1rem;font-weight:500;letter-spacing:2px;text-transform:uppercase;padding:10px 6px;}
#ticker-input::placeholder{color:var(--text3);letter-spacing:1px;text-transform:none;font-weight:400;font-family:var(--font);}
.period-pill{display:flex;gap:3px;background:var(--bg2);border-radius:8px;padding:3px;flex-shrink:0;}
.period-btn{font-family:var(--mono);font-size:0.68rem;font-weight:600;padding:6px 11px;border-radius:6px;border:none;cursor:pointer;color:var(--text3);background:transparent;transition:all 0.15s;}
.period-btn:hover,.period-btn.active{color:var(--text);background:var(--surface2);}
.period-btn.active{color:var(--gold);}
.search-btn{background:linear-gradient(135deg,var(--gold),#e07b00);border:none;color:#000;font-family:var(--font);font-weight:700;font-size:0.85rem;padding:12px 22px;border-radius:10px;cursor:pointer;transition:all 0.2s;white-space:nowrap;box-shadow:0 4px 16px rgba(245,166,35,0.3);}
.search-btn:hover{transform:translateY(-1px);box-shadow:0 6px 22px rgba(245,166,35,0.4);}
.quick-tags{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:22px;}
.qtag{font-family:var(--mono);font-size:0.68rem;font-weight:500;color:var(--text2);background:var(--surface);border:1px solid var(--border);padding:6px 14px;border-radius:100px;cursor:pointer;transition:all 0.2s;letter-spacing:1px;}
.qtag:hover{color:var(--gold);border-color:var(--gold);background:var(--gold-dim);transform:translateY(-1px);}
.features{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:44px;}
.fpill{display:flex;align-items:center;gap:7px;font-size:0.8rem;color:var(--text2);background:var(--surface);border:1px solid var(--border);padding:8px 16px;border-radius:100px;}

/* RESULTS */
#main-content{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:28px 24px 80px;display:none;}

.stock-hero{background:linear-gradient(135deg,var(--surface) 0%,var(--bg2) 100%);border:1px solid var(--border2);border-radius:20px;padding:30px 40px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap;position:relative;overflow:hidden;}
.stock-hero::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--gold),transparent);}
.exc-badge{display:inline-block;font-family:var(--mono);font-size:0.6rem;font-weight:700;color:var(--gold);background:var(--gold-dim);border:1px solid rgba(245,166,35,0.3);padding:3px 10px;border-radius:4px;letter-spacing:2px;margin-bottom:10px;}
.stock-name{font-size:2.1rem;font-weight:800;letter-spacing:-1px;line-height:1.1;}
.stock-sector{font-size:0.85rem;color:var(--text2);margin-top:6px;}
.stock-price{font-family:var(--mono);font-size:2.8rem;font-weight:700;letter-spacing:-2px;line-height:1;}
.stock-price-sub{font-family:var(--mono);font-size:0.75rem;color:var(--text2);margin-top:6px;}

.signal-card{border-radius:20px;padding:26px 30px;margin-bottom:18px;display:flex;align-items:center;gap:24px;flex-wrap:wrap;border:1px solid;position:relative;overflow:hidden;}
.signal-card.buy{background:linear-gradient(135deg,rgba(16,185,129,0.07),transparent);border-color:rgba(16,185,129,0.2);}
.signal-card.sell{background:linear-gradient(135deg,rgba(239,68,68,0.07),transparent);border-color:rgba(239,68,68,0.2);}
.signal-card.hold{background:linear-gradient(135deg,rgba(245,166,35,0.07),transparent);border-color:rgba(245,166,35,0.2);}
.sig-badge{font-family:var(--mono);font-size:1.5rem;font-weight:700;letter-spacing:4px;padding:14px 26px;border-radius:12px;border:2px solid;text-align:center;min-width:120px;flex-shrink:0;}
.sig-badge.buy{color:var(--green);border-color:var(--green);background:var(--green-dim);}
.sig-badge.sell{color:var(--red);border-color:var(--red);background:var(--red-dim);}
.sig-badge.hold{color:var(--gold);border-color:var(--gold);background:var(--gold-dim);}
.sig-body{flex:1;}
.sig-reason{font-size:0.9rem;color:var(--text2);line-height:1.65;margin-bottom:18px;}
.sig-stats{display:flex;gap:28px;flex-wrap:wrap;}
.sstat-label{font-family:var(--mono);font-size:0.58rem;color:var(--text3);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;}
.sstat-value{font-family:var(--mono);font-size:1.05rem;font-weight:700;}
.acc-badge{font-family:var(--mono);font-size:0.62rem;font-weight:600;letter-spacing:1px;color:var(--blue);background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);padding:8px 14px;border-radius:8px;white-space:nowrap;align-self:flex-start;}

.metrics-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}
@media(max-width:900px){.metrics-row{grid-template-columns:1fr 1fr;}}
.mcard{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;transition:border-color 0.2s,transform 0.2s;}
.mcard:hover{border-color:var(--border2);transform:translateY(-2px);}
.mcard-label{font-family:var(--mono);font-size:0.58rem;color:var(--text3);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;}
.mcard-value{font-family:var(--mono);font-size:1.25rem;font-weight:700;}
.mcard-sub{font-size:0.72rem;color:var(--text3);margin-top:4px;}

.charts-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;}
@media(max-width:900px){.charts-row{grid-template-columns:1fr;}}
.chart-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:20px;transition:border-color 0.2s;}
.chart-card:hover{border-color:var(--border2);}
.chart-card.full{margin-bottom:14px;}
.chart-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.chart-title{font-size:0.8rem;font-weight:600;color:var(--text2);}
.chart-badge{font-family:var(--mono);font-size:0.58rem;color:var(--text3);background:var(--bg2);border:1px solid var(--border);padding:3px 8px;border-radius:4px;letter-spacing:1px;}
.chart-area{width:100%;min-height:320px;}
.chart-sm{width:100%;min-height:220px;}

/* LOADING */
#loading{position:fixed;inset:0;z-index:1000;background:rgba(4,5,10,0.92);backdrop-filter:blur(10px);display:none;flex-direction:column;align-items:center;justify-content:center;gap:22px;}
.loader-ring{width:52px;height:52px;border-radius:50%;border:3px solid var(--border2);border-top-color:var(--gold);animation:spin 0.8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.loader-label{font-family:var(--mono);font-size:0.7rem;color:var(--gold);letter-spacing:3px;animation:blink 1.2s step-end infinite;}
@keyframes blink{50%{opacity:0;}}
.loader-sub{font-size:0.8rem;color:var(--text3);}

#toast{position:fixed;top:80px;right:24px;z-index:500;background:var(--surface);border:1px solid rgba(239,68,68,0.4);color:var(--red);font-family:var(--mono);font-size:0.75rem;padding:14px 20px;border-radius:12px;display:none;box-shadow:0 8px 32px rgba(0,0,0,0.4);max-width:360px;}

/* PORTFOLIO */
#portfolio-page{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:48px 24px 80px;display:none;}
.page-eyebrow{font-family:var(--mono);font-size:0.65rem;color:var(--gold);letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;}
.page-title{font-size:2.2rem;font-weight:800;letter-spacing:-1px;margin-bottom:30px;}
.page-title span{color:var(--gold);}
.port-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px;}
@media(max-width:800px){.port-summary{grid-template-columns:1fr 1fr;}}
.pcard{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:22px;position:relative;overflow:hidden;}
.pcard::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--gold),transparent);opacity:0.4;}
.pcard-label{font-family:var(--mono);font-size:0.58rem;color:var(--text3);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;}
.pcard-value{font-family:var(--mono);font-size:1.35rem;font-weight:700;}
.add-form{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px 28px;margin-bottom:18px;}
.add-form-title{font-size:0.9rem;font-weight:600;color:var(--text2);margin-bottom:16px;}
.form-row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;}
.fgroup{display:flex;flex-direction:column;gap:6px;}
.flabel{font-family:var(--mono);font-size:0.58rem;color:var(--text3);letter-spacing:2px;text-transform:uppercase;}
.finput{background:var(--bg2);border:1px solid var(--border);color:var(--text);font-family:var(--mono);font-size:0.88rem;padding:10px 14px;border-radius:10px;outline:none;transition:border-color 0.2s;}
.finput:focus{border-color:var(--gold);}
.btn-gold{background:linear-gradient(135deg,var(--gold),#e07b00);border:none;color:#000;font-family:var(--font);font-weight:700;font-size:0.82rem;padding:11px 22px;border-radius:10px;cursor:pointer;transition:all 0.2s;box-shadow:0 4px 12px rgba(245,166,35,0.3);}
.btn-gold:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(245,166,35,0.4);}
.btn-outline{background:var(--surface2);border:1px solid var(--border2);color:var(--text2);font-family:var(--font);font-weight:600;font-size:0.82rem;padding:11px 22px;border-radius:10px;cursor:pointer;transition:all 0.2s;}
.btn-outline:hover{color:var(--text);border-color:var(--gold);}
.holdings-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;}
.holdings-head{padding:18px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
.holdings-title{font-size:0.9rem;font-weight:600;}
.holdings-count{font-family:var(--mono);font-size:0.7rem;color:var(--text3);}
.htable{width:100%;border-collapse:collapse;}
.htable th{text-align:left;padding:11px 18px;font-family:var(--mono);font-size:0.58rem;color:var(--text3);letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid var(--border);background:var(--bg2);}
.htable td{padding:15px 18px;border-bottom:1px solid var(--border);font-size:0.875rem;}
.htable tr:last-child td{border-bottom:none;}
.htable tr:hover td{background:rgba(255,255,255,0.02);}
.sym-cell{font-family:var(--mono);font-weight:700;color:var(--gold);font-size:0.95rem;}
.pos{color:var(--green);} .neg{color:var(--red);}
.tbtn{font-family:var(--mono);font-size:0.62rem;font-weight:600;padding:5px 11px;border-radius:6px;border:1px solid;cursor:pointer;transition:all 0.2s;letter-spacing:0.5px;margin-right:5px;}
.tbtn-a{color:var(--gold);border-color:rgba(245,166,35,0.3);background:var(--gold-dim);}
.tbtn-a:hover{background:rgba(245,166,35,0.2);}
.tbtn-r{color:var(--red);border-color:rgba(239,68,68,0.3);background:var(--red-dim);}
.tbtn-r:hover{background:rgba(239,68,68,0.2);}
.empty-state{padding:56px 24px;text-align:center;color:var(--text3);font-size:0.9rem;line-height:1.9;}
.empty-icon{font-size:2.8rem;margin-bottom:14px;opacity:0.4;}

.fade-in{animation:fadeIn 0.4s ease both;}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
</style>
</head>
<body>

<div id="loading">
  <div class="loader-ring"></div>
  <div class="loader-label">ANALYZING MARKET DATA</div>
  <div class="loader-sub">Fetching NSE data · Running ML model...</div>
</div>
<div id="toast"></div>

<header>
  <div class="logo-wrap">
    <div class="logo-icon">S</div>
    <div>
      <div class="logo-text">Stock<span>Sense</span></div>
      <div class="logo-sub">NSE · BSE · AI PREDICTIONS</div>
    </div>
  </div>
  <div class="nav-tabs">
    <button class="nav-tab active" id="tab-analyze" onclick="switchTab('analyze')">📈 Analyze</button>
    <button class="nav-tab" id="tab-portfolio" onclick="switchTab('portfolio')">💼 Portfolio</button>
  </div>
  <div class="header-right">
    <div class="market-badge"><div class="dot"></div><span>NSE LIVE</span></div>
    <div id="clock"></div>
  </div>
</header>

<!-- ANALYZE PAGE -->
<div id="analyze-page">
  <div class="hero">
    <div class="hero-eyebrow">⚡ AI-Powered · Real-Time · NSE/BSE</div>
    <h1 class="hero-title">Predict India\'s<br/><span class="hl">Stock Market</span></h1>
    <p class="hero-sub">Get ML-powered 30-day price predictions, technical analysis, and intelligent buy/sell signals for any NSE or BSE listed stock.</p>

    <div class="search-wrap-outer">
      <div class="search-box">
        <div class="search-icon">₹</div>
        <input id="ticker-input" type="text" placeholder="Search any NSE stock — TCS, RELIANCE, INFY..." maxlength="20"/>
        <div class="period-pill">
          <button class="period-btn" data-period="1y" onclick="setPeriod(this)">1Y</button>
          <button class="period-btn active" data-period="2y" onclick="setPeriod(this)">2Y</button>
          <button class="period-btn" data-period="5y" onclick="setPeriod(this)">5Y</button>
        </div>
        <button class="search-btn" onclick="analyze()">Analyze →</button>
      </div>
    </div>

    <div class="quick-tags">
      <span class="qtag" onclick="quickSearch(\'RELIANCE\')">RELIANCE</span>
      <span class="qtag" onclick="quickSearch(\'TCS\')">TCS</span>
      <span class="qtag" onclick="quickSearch(\'INFY\')">INFY</span>
      <span class="qtag" onclick="quickSearch(\'HDFCBANK\')">HDFCBANK</span>
      <span class="qtag" onclick="quickSearch(\'WIPRO\')">WIPRO</span>
      <span class="qtag" onclick="quickSearch(\'ICICIBANK\')">ICICIBANK</span>
      <span class="qtag" onclick="quickSearch(\'SBIN\')">SBIN</span>
      <span class="qtag" onclick="quickSearch(\'BAJFINANCE\')">BAJFINANCE</span>
      <span class="qtag" onclick="quickSearch(\'TATAMOTORS\')">TATAMOTORS</span>
      <span class="qtag" onclick="quickSearch(\'NVDA\')">NVDA</span>
    </div>

    <div class="features">
      <div class="fpill">🤖 Random Forest ML</div>
      <div class="fpill">📊 RSI · MACD · Bollinger</div>
      <div class="fpill">🎯 Buy/Sell Signals</div>
      <div class="fpill">💹 30-Day Forecast</div>
      <div class="fpill">💼 Portfolio Tracker</div>
    </div>
  </div>

  <!-- Results -->
  <div id="main-content">
    <div class="stock-hero fade-in">
      <div>
        <div id="disp-exchange" class="exc-badge">NSE</div>
        <div id="disp-name" class="stock-name">—</div>
        <div id="disp-sector" class="stock-sector">—</div>
      </div>
      <div style="text-align:right;">
        <div id="disp-price" class="stock-price">—</div>
        <div id="disp-symbol" class="stock-price-sub">—</div>
      </div>
    </div>

    <div id="signal-card" class="signal-card fade-in">
      <div id="signal-badge" class="sig-badge">—</div>
      <div class="sig-body">
        <div id="signal-reason" class="sig-reason">—</div>
        <div class="sig-stats">
          <div><div class="sstat-label">Current Price</div><div class="sstat-value" id="sig-current">—</div></div>
          <div><div class="sstat-label">30-Day Target</div><div class="sstat-value" id="sig-target">—</div></div>
          <div><div class="sstat-label">Expected Move</div><div class="sstat-value" id="sig-change">—</div></div>
          <div><div class="sstat-label">RSI</div><div class="sstat-value" id="sig-rsi">—</div></div>
        </div>
      </div>
      <div id="acc-badge" class="acc-badge">MODEL ACCURACY: —</div>
    </div>

    <div class="metrics-row">
      <div class="mcard"><div class="mcard-label">52W High</div><div class="mcard-value" id="m-high">—</div><div class="mcard-sub">52-week peak</div></div>
      <div class="mcard"><div class="mcard-label">52W Low</div><div class="mcard-value" id="m-low">—</div><div class="mcard-sub">52-week trough</div></div>
      <div class="mcard"><div class="mcard-label">P/E Ratio</div><div class="mcard-value" id="m-pe">—</div><div class="mcard-sub">Price to earnings</div></div>
      <div class="mcard"><div class="mcard-label">Avg Volume</div><div class="mcard-value" id="m-vol">—</div><div class="mcard-sub">Daily avg shares</div></div>
    </div>

    <div class="chart-card full fade-in">
      <div class="chart-header"><div class="chart-title">📈 Price History + ML Prediction</div><div class="chart-badge">BOLLINGER BANDS · MA20 · MA50</div></div>
      <div id="price-chart" class="chart-area"></div>
    </div>

    <div class="charts-row">
      <div class="chart-card fade-in">
        <div class="chart-header"><div class="chart-title">⚡ RSI — Relative Strength Index</div><div class="chart-badge">14-PERIOD</div></div>
        <div id="rsi-chart" class="chart-sm"></div>
      </div>
      <div class="chart-card fade-in">
        <div class="chart-header"><div class="chart-title">📉 MACD — Moving Avg Convergence</div><div class="chart-badge">12·26·9</div></div>
        <div id="macd-chart" class="chart-sm"></div>
      </div>
    </div>

    <div class="chart-card full fade-in">
      <div class="chart-header"><div class="chart-title">📊 Volume Analysis</div><div class="chart-badge">20-DAY MA</div></div>
      <div id="volume-chart" class="chart-sm"></div>
    </div>
  </div>
</div>

<!-- PORTFOLIO PAGE -->
<div id="portfolio-page">
  <div class="page-eyebrow">💼 MY INVESTMENTS</div>
  <div class="page-title">Portfolio <span>Tracker</span></div>

  <div class="port-summary">
    <div class="pcard"><div class="pcard-label">Total Invested</div><div class="pcard-value" id="p-invested">₹0</div></div>
    <div class="pcard"><div class="pcard-label">Current Value</div><div class="pcard-value" id="p-current">₹0</div></div>
    <div class="pcard"><div class="pcard-label">Total P&L</div><div class="pcard-value" id="p-pnl">₹0</div></div>
    <div class="pcard"><div class="pcard-label">Overall Return</div><div class="pcard-value" id="p-pnl-pct">0%</div></div>
  </div>

  <div class="add-form">
    <div class="add-form-title">➕ Add Stock to Portfolio</div>
    <div class="form-row">
      <div class="fgroup"><div class="flabel">Symbol</div><input class="finput" id="p-symbol" placeholder="TCS" style="text-transform:uppercase;width:120px;"/></div>
      <div class="fgroup"><div class="flabel">Quantity</div><input class="finput" id="p-qty" type="number" placeholder="10" min="1" style="width:110px;"/></div>
      <div class="fgroup"><div class="flabel">Buy Price (₹)</div><input class="finput" id="p-price" type="number" placeholder="3500" min="0" style="width:150px;"/></div>
      <button class="btn-gold" onclick="addToPortfolio()">Add Stock</button>
      <button class="btn-outline" onclick="refreshPortfolio()">↻ Refresh</button>
    </div>
  </div>

  <div class="holdings-card">
    <div class="holdings-head">
      <div class="holdings-title">Holdings</div>
      <div class="holdings-count" id="holdings-count">0 stocks</div>
    </div>
    <div id="portfolio-table-wrap">
      <div class="empty-state"><div class="empty-icon">📭</div>No stocks yet. Add your first holding above.</div>
    </div>
  </div>
</div>

<script>
const PL = () => ({
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  font:{family:'JetBrains Mono,monospace',color:'#475569',size:10.5},
  margin:{t:8,r:16,b:36,l:64},
  xaxis:{gridcolor:'#1a2535',linecolor:'#1a2535',tickcolor:'transparent',color:'#475569',showgrid:true,zeroline:false},
  yaxis:{gridcolor:'#1a2535',linecolor:'#1a2535',tickcolor:'transparent',color:'#475569',showgrid:true,zeroline:false},
  showlegend:true,
  legend:{bgcolor:'rgba(8,12,20,0.85)',bordercolor:'#1a2535',borderwidth:1,font:{size:10,color:'#94a3b8'},x:1,xanchor:'right',y:1},
  hovermode:'x unified',
  hoverlabel:{bgcolor:'#0c1220',bordercolor:'#243047',font:{family:'JetBrains Mono',size:11,color:'#f1f5f9'}}
});
const CFG={displayModeBar:false,responsive:true};
let activePeriod='2y';

function setPeriod(btn){
  document.querySelectorAll('.period-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active'); activePeriod=btn.dataset.period;
}
function quickSearch(s){document.getElementById('ticker-input').value=s;analyze();}
function showError(msg){const t=document.getElementById('toast');t.textContent='⚠ '+msg;t.style.display='block';setTimeout(()=>t.style.display='none',4500);}
function fmt(n,d=2){if(n===null||n===undefined)return'N/A';return Number(n).toLocaleString('en-IN',{minimumFractionDigits:d,maximumFractionDigits:d});}
function fmtVol(n){if(!n)return'N/A';if(n>=1e7)return(n/1e7).toFixed(1)+'Cr';if(n>=1e5)return(n/1e5).toFixed(1)+'L';if(n>=1e3)return(n/1e3).toFixed(0)+'K';return n;}

function switchTab(tab){
  document.getElementById('tab-analyze').classList.toggle('active',tab==='analyze');
  document.getElementById('tab-portfolio').classList.toggle('active',tab==='portfolio');
  document.getElementById('analyze-page').style.display=tab==='analyze'?'block':'none';
  document.getElementById('portfolio-page').style.display=tab==='portfolio'?'block':'none';
  if(tab==='portfolio') refreshPortfolio();
}

async function analyze(){
  const sym=document.getElementById('ticker-input').value.trim().toUpperCase();
  if(!sym)return;
  document.getElementById('loading').style.display='flex';
  document.getElementById('main-content').style.display='none';
  try{
    const res=await fetch(`/analyze?symbol=${sym}&period=${activePeriod}`);
    const data=await res.json();
    if(data.error){showError(data.error);return;}
    renderAll(data);
  }catch(e){showError('Network error. Check connection.');}
  finally{document.getElementById('loading').style.display='none';}
}

function renderAll(d){
  const{info,signal,historical:h,prediction:pred}=d;
  document.getElementById('disp-exchange').textContent=d.exchange||'NSE';
  document.getElementById('disp-name').textContent=info.name||d.symbol;
  document.getElementById('disp-sector').textContent=info.sector||'N/A';
  document.getElementById('disp-price').textContent='₹'+fmt(signal.current_price);
  document.getElementById('disp-symbol').textContent=d.symbol+' · '+(d.exchange||'NSE');
  const sl=signal.signal.toLowerCase();
  const badge=document.getElementById('signal-badge');
  badge.textContent=signal.signal; badge.className='sig-badge '+sl;
  document.getElementById('signal-card').className='signal-card '+sl+' fade-in';
  document.getElementById('signal-reason').textContent=signal.reason;
  document.getElementById('sig-current').textContent='₹'+fmt(signal.current_price);
  document.getElementById('sig-target').textContent='₹'+fmt(signal.target_price);
  const chg=signal.predicted_change;
  const ce=document.getElementById('sig-change');
  ce.textContent=(chg>=0?'+':'')+fmt(chg)+'%';
  ce.style.color=chg>=0?'var(--green)':'var(--red)';
  document.getElementById('sig-rsi').textContent=fmt(signal.rsi,1);
  document.getElementById('acc-badge').textContent='🎯 Model Accuracy: '+d.accuracy+'%';
  document.getElementById('m-high').textContent=info['52w_high']?'₹'+fmt(info['52w_high']):'N/A';
  document.getElementById('m-low').textContent=info['52w_low']?'₹'+fmt(info['52w_low']):'N/A';
  document.getElementById('m-pe').textContent=info.pe_ratio?fmt(info.pe_ratio,1)+'x':'N/A';
  document.getElementById('m-vol').textContent=fmtVol(info.avg_volume);

  const dates=h.dates, lastDate=dates[dates.length-1], lastClose=h.close[h.close.length-1];

  Plotly.newPlot('price-chart',[
    {x:dates,y:h.bb_upper,name:'BB Upper',line:{color:'#243047',width:1,dash:'dot'},showlegend:false,hoverinfo:'skip'},
    {x:dates,y:h.bb_lower,name:'BB Band',fill:'tonexty',fillcolor:'rgba(245,166,35,0.04)',line:{color:'#243047',width:1,dash:'dot'},hoverinfo:'skip'},
    {x:dates,y:h.close,name:'Close',line:{color:'#f5a623',width:2.5},hovertemplate:'<b>%{x}</b><br>₹%{y:.2f}<extra>Close</extra>'},
    {x:dates,y:h.ma20,name:'MA 20',line:{color:'#3b82f6',width:1.5,dash:'dot'},hovertemplate:'MA20: ₹%{y:.2f}<extra></extra>'},
    {x:dates,y:h.ma50,name:'MA 50',line:{color:'#8b5cf6',width:1.5,dash:'dash'},hovertemplate:'MA50: ₹%{y:.2f}<extra></extra>'},
    {x:[lastDate,...pred.dates],y:[lastClose,...pred.prices],name:'ML Forecast',line:{color:'#10b981',width:2,dash:'dot'},hovertemplate:'<b>Forecast</b><br>%{x}<br>₹%{y:.2f}<extra></extra>'}
  ],{...PL(),margin:{t:8,r:16,b:36,l:72},yaxis:{...PL().yaxis,tickprefix:'₹'},
    shapes:[{type:'line',x0:lastDate,x1:lastDate,y0:0,y1:1,yref:'paper',line:{color:'#10b981',width:1,dash:'dot'}}],
    annotations:[{x:lastDate,y:1,yref:'paper',text:'FORECAST →',showarrow:false,font:{color:'#10b981',size:9,family:'JetBrains Mono'},yanchor:'bottom',xanchor:'left'}]
  },CFG);

  Plotly.newPlot('rsi-chart',[
    {x:dates,y:h.rsi,name:'RSI',line:{color:'#f5a623',width:2},fill:'tozeroy',fillcolor:'rgba(245,166,35,0.05)',hovertemplate:'RSI: %{y:.1f}<extra></extra>'}
  ],{...PL(),yaxis:{...PL().yaxis,range:[0,100]},
    shapes:[
      {type:'rect',x0:dates[0],x1:dates[dates.length-1],y0:70,y1:100,fillcolor:'rgba(239,68,68,0.05)',line:{width:0}},
      {type:'rect',x0:dates[0],x1:dates[dates.length-1],y0:0,y1:30,fillcolor:'rgba(16,185,129,0.05)',line:{width:0}},
      {type:'line',x0:dates[0],x1:dates[dates.length-1],y0:70,y1:70,line:{color:'#ef4444',width:1,dash:'dot'}},
      {type:'line',x0:dates[0],x1:dates[dates.length-1],y0:30,y1:30,line:{color:'#10b981',width:1,dash:'dot'}}
    ],
    annotations:[
      {x:dates[10],y:72,text:'Overbought',showarrow:false,font:{color:'#ef4444',size:9,family:'JetBrains Mono'}},
      {x:dates[10],y:28,text:'Oversold',showarrow:false,yanchor:'top',font:{color:'#10b981',size:9,family:'JetBrains Mono'}}
    ]
  },CFG);

  const mhist=h.macd.map((m,i)=>m-(h.macd_signal[i]||0));
  Plotly.newPlot('macd-chart',[
    {x:dates,y:mhist,name:'Histogram',type:'bar',marker:{color:mhist.map(v=>v>=0?'rgba(16,185,129,0.6)':'rgba(239,68,68,0.6)')},hovertemplate:'Hist: %{y:.3f}<extra></extra>'},
    {x:dates,y:h.macd,name:'MACD',line:{color:'#3b82f6',width:1.5},hovertemplate:'MACD: %{y:.3f}<extra></extra>'},
    {x:dates,y:h.macd_signal,name:'Signal',line:{color:'#f5a623',width:1.5,dash:'dot'},hovertemplate:'Signal: %{y:.3f}<extra></extra>'}
  ],PL(),CFG);

  Plotly.newPlot('volume-chart',[
    {x:dates,y:h.volume,name:'Volume',type:'bar',marker:{color:h.close.map((c,i)=>i===0||c>=h.close[i-1]?'rgba(245,166,35,0.5)':'rgba(239,68,68,0.5)')},hovertemplate:'%{x}<br>%{y:,.0f}<extra>Volume</extra>'},
    {x:dates,y:h.vol_ma20,name:'MA 20',line:{color:'#8b5cf6',width:1.5},hovertemplate:'MA20: %{y:,.0f}<extra></extra>'}
  ],{...PL(),yaxis:{...PL().yaxis,tickformat:'.2s'}},CFG);

  document.getElementById('main-content').style.display='block';
  document.getElementById('main-content').scrollIntoView({behavior:'smooth',block:'start'});
}

async function addToPortfolio(){
  const symbol=document.getElementById('p-symbol').value.trim().toUpperCase();
  const qty=parseFloat(document.getElementById('p-qty').value);
  const price=parseFloat(document.getElementById('p-price').value);
  if(!symbol||!qty||!price){showError('Please fill all fields.');return;}
  const res=await fetch('/portfolio/add',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol,quantity:qty,buy_price:price})});
  const data=await res.json();
  if(data.error){showError(data.error);return;}
  document.getElementById('p-symbol').value='';
  document.getElementById('p-qty').value='';
  document.getElementById('p-price').value='';
  await refreshPortfolio();
}

async function removeFromPortfolio(symbol){
  await fetch('/portfolio/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol})});
  await refreshPortfolio();
}

async function refreshPortfolio(){
  const res=await fetch('/portfolio');
  renderPortfolio(await res.json());
}

function renderPortfolio(data){
  const wrap=document.getElementById('portfolio-table-wrap');
  const count=data.holdings?data.holdings.length:0;
  document.getElementById('holdings-count').textContent=count+' stock'+(count!==1?'s':'');
  if(!count){
    wrap.innerHTML='<div class="empty-state"><div class="empty-icon">📭</div>No stocks yet. Add your first holding above.</div>';
    ['p-invested','p-current','p-pnl','p-pnl-pct'].forEach(id=>document.getElementById(id).textContent='₹0');
    return;
  }
  document.getElementById('p-invested').textContent='₹'+fmt(data.total_invested);
  document.getElementById('p-current').textContent='₹'+fmt(data.total_current);
  const pnl=data.total_pnl,pct=data.total_pnl_pct;
  const pe=document.getElementById('p-pnl');
  pe.textContent=(pnl>=0?'+₹':'-₹')+fmt(Math.abs(pnl));
  pe.className='pcard-value '+(pnl>=0?'pos':'neg');
  const pp=document.getElementById('p-pnl-pct');
  pp.textContent=(pct>=0?'+':'')+fmt(pct)+'%';
  pp.className='pcard-value '+(pct>=0?'pos':'neg');
  let html=`<table class="htable"><thead><tr><th>Symbol</th><th>Qty</th><th>Buy Price</th><th>Current</th><th>Invested</th><th>Value</th><th>P&L</th><th>Return</th><th>Actions</th></tr></thead><tbody>`;
  for(const h of data.holdings){
    const pc=h.pnl>=0?'pos':'neg';
    html+=`<tr><td class="sym-cell">${h.symbol}</td><td>${h.quantity}</td><td>₹${fmt(h.buy_price)}</td><td>₹${fmt(h.current_price)}</td><td>₹${fmt(h.invested)}</td><td>₹${fmt(h.current_value)}</td><td class="${pc}">${h.pnl>=0?'+':'-'}₹${fmt(Math.abs(h.pnl))}</td><td class="${pc}">${h.pnl_pct>=0?'+':''}${fmt(h.pnl_pct)}%</td><td><button class="tbtn tbtn-a" onclick="analyzeFromPortfolio('${h.symbol}')">ANALYZE</button><button class="tbtn tbtn-r" onclick="removeFromPortfolio('${h.symbol}')">REMOVE</button></td></tr>`;
  }
  wrap.innerHTML=html+'</tbody></table>';
}

function analyzeFromPortfolio(s){switchTab('analyze');document.getElementById('ticker-input').value=s;analyze();}

function updateClock(){
  const ist=new Date(new Date().toLocaleString('en-US',{timeZone:'Asia/Kolkata'}));
  document.getElementById('clock').textContent=ist.toLocaleTimeString('en-IN',{hour12:false})+' IST';
}
updateClock(); setInterval(updateClock,1000);

document.getElementById('ticker-input').addEventListener('keydown',e=>{if(e.key==='Enter')analyze();});
document.getElementById('p-symbol').addEventListener('keydown',e=>{if(e.key==='Enter')addToPortfolio();});
</script>
</body>
</html>'''


def safe_list(series, round_digits=4):
    return [
        None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), round_digits)
        for v in series
    ]


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze")
def analyze():
    symbol = request.args.get("symbol", "").upper().strip()
    period = request.args.get("period", "2y")
    if not symbol:
        return jsonify({"error": "Please enter a stock symbol."})
    if "." not in symbol:
        symbol_to_fetch = symbol + ".NS"
    else:
        symbol_to_fetch = symbol
    df, err = get_stock_data(symbol_to_fetch, period=period)
    if (err or df is None) and "." not in symbol:
        symbol_to_fetch = symbol + ".BO"
        df, err = get_stock_data(symbol_to_fetch, period=period)
    if err or df is None:
        return jsonify({"error": f"Could not find '{symbol}' on NSE or BSE."})
    df = compute_indicators(df)
    info = get_stock_info(symbol_to_fetch)
    pred_df, accuracy, model_err = train_and_predict(df, days=30)
    if model_err or pred_df is None:
        return jsonify({"error": model_err or "Model training failed."})
    signal = get_signal(df, pred_df)
    hist = {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "close": safe_list(df["Close"]), "volume": safe_list(df["Volume"], 0),
        "ma20": safe_list(df["MA20"]), "ma50": safe_list(df["MA50"]),
        "bb_upper": safe_list(df["BB_upper"]), "bb_lower": safe_list(df["BB_lower"]),
        "rsi": safe_list(df["RSI"]), "macd": safe_list(df["MACD"]),
        "macd_signal": safe_list(df["MACD_signal"]),
        "macd_hist": safe_list(df["MACD"] - df["MACD_signal"]),
        "vol_ma20": safe_list(df["Vol_MA20"], 0),
    }
    prediction = {
        "dates": [d.strftime("%Y-%m-%d") for d in pred_df["Date"]],
        "prices": safe_list(pred_df["Predicted"]),
    }
    return jsonify({
        "symbol": symbol, "exchange": "NSE" if symbol_to_fetch.endswith(".NS") else "BSE",
        "info": info, "historical": hist, "prediction": prediction,
        "signal": signal, "accuracy": accuracy,
    })


@app.route("/portfolio")
def portfolio():
    holdings, total_invested, total_current, total_pnl, total_pnl_pct = get_portfolio_with_live_prices()
    return jsonify({
        "holdings": holdings, "total_invested": total_invested,
        "total_current": total_current, "total_pnl": total_pnl, "total_pnl_pct": total_pnl_pct,
    })


@app.route("/portfolio/add", methods=["POST"])
def portfolio_add():
    data = request.get_json()
    symbol = data.get("symbol", "").upper().strip()
    quantity = float(data.get("quantity", 0))
    buy_price = float(data.get("buy_price", 0))
    if not symbol or quantity <= 0 or buy_price <= 0:
        return jsonify({"error": "Invalid input."})
    add_stock(symbol, quantity, buy_price)
    return jsonify({"success": True})


@app.route("/portfolio/remove", methods=["POST"])
def portfolio_remove():
    data = request.get_json()
    symbol = data.get("symbol", "").upper().strip()
    remove_stock(symbol)
    return jsonify({"success": True})


if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
