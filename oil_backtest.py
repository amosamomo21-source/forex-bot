"""Backtest WTI and Brent crude oil on all three intraday strategies:
- H1 EMA(10/30) crossover
- H1 MACD(12,26,9) crossover
- ORB M30 (London + NY session opens)
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import ema, macd, atr

INSTRUMENTS = {"WTICO_USD": "WTI Crude", "BCO_USD": "Brent Crude"}
RISK_PCT = 0.01
MAX_LEV  = 5.0
INITIAL  = 10_000


# ── H1 EMA ────────────────────────────────────────────────────────────────────
def backtest_ema_h1(ticker, period):
    df = load_oanda_data(ticker, period=period, interval="1h")
    if df is None or len(df) < 35:
        return None

    fast_s = ema(df["Close"], 10)
    slow_s = ema(df["Close"], 30)
    atr_s  = atr(df["High"], df["Low"], df["Close"], 14)

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []
    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        av    = atr_s.iloc[i]
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
        sd = 1.5*av
        units = min(equity*RISK_PCT/sd, equity*MAX_LEV/price)
        if units <= 0: continue
        if cross_up:  position=1;  entry_price=price; sl=price-sd; tp=price+2.5*av
        else:         position=-1; entry_price=price; sl=price+sd; tp=price-2.5*av

    return _stats(trades, equity, period)


# ── H1 MACD ───────────────────────────────────────────────────────────────────
def backtest_macd_h1(ticker, period):
    df = load_oanda_data(ticker, period=period, interval="1h")
    if df is None or len(df) < 40:
        return None

    ml, sig = macd(df["Close"], 12, 26, 9)
    atr_s   = atr(df["High"], df["Low"], df["Close"], 14)

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []
    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        av    = atr_s.iloc[i]
        mn, mp = ml.iloc[i], ml.iloc[i-1]
        sn, sp = sig.iloc[i], sig.iloc[i-1]
        if np.isnan(av) or av <= 0 or np.isnan(mn): continue
        if position != 0:
            sl_hit = (position==1 and price<=sl) or (position==-1 and price>=sl)
            tp_hit = (position==1 and price>=tp) or (position==-1 and price<=tp)
            if sl_hit or tp_hit:
                ep = sl if sl_hit else tp
                trades.append(position*(ep-entry_price)/abs(entry_price-sl)*equity*RISK_PCT)
                equity += trades[-1]; position = 0
        cross_up = mp<=sp and mn>sn; cross_dn = mp>=sp and mn<sn
        if not (cross_up or cross_dn) or position != 0: continue
        sd = 1.5*av
        units = min(equity*RISK_PCT/sd, equity*MAX_LEV/price)
        if units <= 0: continue
        if cross_up:  position=1;  entry_price=price; sl=price-sd; tp=price+2.5*av
        else:         position=-1; entry_price=price; sl=price+sd; tp=price-2.5*av

    return _stats(trades, equity, period)


# ── ORB M30 ───────────────────────────────────────────────────────────────────
SESSION_HOURS = {8, 13}
TP_MULT = 1.5

def backtest_orb(ticker, period):
    df = load_oanda_data(ticker, period=period, interval="30m")
    if df is None or len(df) < 100:
        return None

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []
    for i in range(1, len(df)):
        price      = df["Close"].iloc[i]
        prev_hour  = df.index[i-1].hour

        if position != 0:
            sl_hit = (position==1 and price<=sl) or (position==-1 and price>=sl)
            tp_hit = (position==1 and price>=tp) or (position==-1 and price<=tp)
            if sl_hit or tp_hit:
                ep = sl if sl_hit else tp
                trades.append(position*(ep-entry_price)/abs(entry_price-sl)*equity*RISK_PCT)
                equity += trades[-1]; position = 0

        if prev_hour in SESSION_HOURS and position == 0:
            or_high = df["High"].iloc[i-1]; or_low = df["Low"].iloc[i-1]
            rng = or_high - or_low
            if rng <= 0: continue
            units = min(equity*RISK_PCT/rng, equity*MAX_LEV/price)
            if units <= 0: continue
            if price > or_high:
                position=1;  entry_price=price; sl=or_low;  tp=price+TP_MULT*rng
            elif price < or_low:
                position=-1; entry_price=price; sl=or_high; tp=price-TP_MULT*rng

    return _stats(trades, equity, period)


def _stats(trades, equity, period):
    if len(trades) < 10:
        return None
    t   = pd.Series(trades)
    yrs = int(period[0])
    sharpe = t.mean()/t.std()*np.sqrt(252) if t.std()>0 else 0
    dd     = (t.cumsum()+INITIAL).pipe(lambda s: ((s-s.cummax())/s.cummax()).min())
    ann    = (equity/INITIAL)**(1/yrs)-1
    return {"n": len(t), "wr": (t>0).mean(), "sharpe": sharpe, "dd": dd, "ann": ann}


def row(r, label):
    if r is None:
        return f"  {label:<12}  {'--':>7}  {'--':>6}  {'--':>5}  {'--':>7}"
    return (f"  {label:<12}  {r['sharpe']:>7.2f}  {r['n']:>6}  "
            f"{r['wr']:>5.0%}  {r['dd']:>7.1%}")


print("Oil backtest — WTI and Brent\n")
print(f"{'Strategy':<14}  {'Sharpe':>7}  {'Trades':>6}  {'WR':>5}  {'MaxDD':>7}")
print("-" * 50)

for ticker, name in INSTRUMENTS.items():
    print(f"\n{name} ({ticker})")
    for period in ("5y", "10y"):
        r_ema  = backtest_ema_h1(ticker, period)
        r_macd = backtest_macd_h1(ticker, period)
        r_orb  = backtest_orb(ticker, period)
        print(f"  [{period}]")
        print(row(r_ema,  "H1 EMA"))
        print(row(r_macd, "H1 MACD"))
        print(row(r_orb,  "ORB M30"))
