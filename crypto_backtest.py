"""H1 EMA(10/30) backtest on major cryptocurrencies via Yahoo Finance.
OANDA practice account has no crypto -- this tests whether the strategy
has edge on crypto in principle. Live trading would need a different broker.
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
import yfinance as yf
from strategies import ema, atr

CRYPTOS = {
    "BTC-USD":  "Bitcoin",
    "ETH-USD":  "Ethereum",
    "BNB-USD":  "BNB",
    "SOL-USD":  "Solana",
    "XRP-USD":  "Ripple",
    "ADA-USD":  "Cardano",
    "AVAX-USD": "Avalanche",
    "DOGE-USD": "Dogecoin",
    "LTC-USD":  "Litecoin",
    "LINK-USD": "Chainlink",
}

RISK_PCT = 0.01
MAX_LEV  = 5.0
INITIAL  = 10_000


def load_crypto(ticker, period):
    df = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close"]].dropna()


def backtest(ticker, period):
    df = load_crypto(ticker, period)
    if df is None or len(df) < 35:
        return None

    fast_s = ema(df["Close"], 10)
    slow_s = ema(df["Close"], 30)
    atr_s  = atr(df["High"], df["Low"], df["Close"], 14)

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []
    for i in range(1, len(df)):
        price = df["Close"].iloc[i]; av = atr_s.iloc[i]
        fn, fp = fast_s.iloc[i], fast_s.iloc[i-1]
        sn, sp = slow_s.iloc[i], slow_s.iloc[i-1]
        if np.isnan(av) or av <= 0: continue
        if position != 0:
            sl_hit = (position==1 and price<=sl) or (position==-1 and price>=sl)
            tp_hit = (position==1 and price>=tp) or (position==-1 and price<=tp)
            if sl_hit or tp_hit:
                ep = sl if sl_hit else tp
                trades.append(position*(ep-entry_price)/abs(entry_price-sl)*equity*RISK_PCT)
                equity += trades[-1]; position = 0
        cross_up = fp<=sp and fn>sn; cross_dn = fp>=sp and fn<sn
        if not (cross_up or cross_dn) or position != 0: continue
        sd = 1.5 * av
        units = min(equity*RISK_PCT/sd, equity*MAX_LEV/price)
        if units <= 0: continue
        if cross_up:  position=1;  entry_price=price; sl=price-sd; tp=price+2.5*av
        else:         position=-1; entry_price=price; sl=price+sd; tp=price-2.5*av

    if len(trades) < 10:
        return None
    t   = pd.Series(trades)
    sharpe = t.mean()/t.std()*np.sqrt(252*24) if t.std()>0 else 0  # crypto trades 24/7
    dd     = (t.cumsum()+INITIAL).pipe(lambda s: ((s-s.cummax())/s.cummax()).min())
    ann    = (equity/INITIAL)**(1/2)-1  # yfinance caps at ~2y for H1
    return {"n": len(t), "wr": (t>0).mean(), "sharpe": sharpe, "dd": dd, "ann": ann}


print("H1 EMA(10/30) — Cryptocurrency (Yahoo Finance, ~2y history)\n")
print(f"{'Ticker':<12} {'Name':<12} {'Sharpe':>7} {'Trades':>7} {'WR':>5} {'MaxDD':>7} {'Ann%':>7}  Verdict")
print("-" * 72)

good = []
for ticker, name in CRYPTOS.items():
    r = backtest(ticker, "2y")
    if r is None:
        print(f"{ticker:<12} {name:<12}  -- insufficient data")
        continue
    verdict = "PASS" if r["sharpe"] > 0 else "FAIL"
    if verdict == "PASS":
        good.append((ticker, name, r["sharpe"]))
    print(f"{ticker:<12} {name:<12} {r['sharpe']:>7.2f} {r['n']:>7} {r['wr']:>5.0%} "
          f"{r['dd']:>7.1%} {r['ann']:>7.0%}  {verdict}")

print(f"\nPassing ({len(good)}): {', '.join(n for _,n,_ in good)}")
print("\nNote: live trading crypto requires a different broker (not OANDA)")
