"""MACD(12,26,9) crossover on H1 bars. SL=1.5xATR, TP=2.5xATR.
Tests same pair list as H1 EMA sleeves + a few extras.
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import macd, atr

FAST     = 12
SLOW     = 26
SIGNAL   = 9
SL_MULT  = 1.5
TP_MULT  = 2.5
RISK_PCT = 0.01
MAX_LEV  = 5.0
INITIAL  = 10_000

PAIRS = [
    "GBP_USD", "EUR_JPY", "CHF_JPY", "CAD_JPY", "AUD_JPY",
    "GBP_JPY", "NZD_JPY", "AUD_CHF", "EUR_AUD", "AUD_SGD",
    "EUR_USD", "USD_JPY", "EUR_CAD", "GBP_CAD", "GBP_CHF",
]


def backtest(pair, period):
    df = load_oanda_data(pair, period=period, interval="1h")
    if df is None or len(df) < SLOW + SIGNAL + 5:
        return None

    ml, sl_line = macd(df["Close"], FAST, SLOW, SIGNAL)
    atr_s = atr(df["High"], df["Low"], df["Close"], 14)

    equity = INITIAL
    position = 0
    entry_price = sl = tp = 0.0
    trades = []

    for i in range(1, len(df)):
        price    = df["Close"].iloc[i]
        ml_now   = ml.iloc[i];     ml_prev   = ml.iloc[i-1]
        sig_now  = sl_line.iloc[i]; sig_prev  = sl_line.iloc[i-1]
        av       = atr_s.iloc[i]

        if np.isnan(av) or av <= 0 or np.isnan(ml_now) or np.isnan(sig_now):
            continue

        if position != 0:
            sl_hit = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            tp_hit = (position == 1 and price >= tp) or (position == -1 and price <= tp)
            if sl_hit or tp_hit:
                exit_p = sl if sl_hit else tp
                pnl_r  = position * (exit_p - entry_price) / abs(entry_price - sl)
                trades.append(pnl_r * equity * RISK_PCT)
                equity += trades[-1]
                position = 0

        cross_up = ml_prev <= sig_prev and ml_now > sig_now
        cross_dn = ml_prev >= sig_prev and ml_now < sig_now
        if not (cross_up or cross_dn) or position != 0:
            continue

        stop_dist = SL_MULT * av
        units = min(equity * RISK_PCT / stop_dist, equity * MAX_LEV / price)
        if units <= 0:
            continue

        if cross_up:
            position = 1;  entry_price = price; sl = price - stop_dist; tp = price + TP_MULT * av
        else:
            position = -1; entry_price = price; sl = price + stop_dist; tp = price - TP_MULT * av

    if len(trades) < 10:
        return None
    t   = pd.Series(trades)
    yrs = int(period[0])
    sharpe = t.mean() / t.std() * np.sqrt(252) if t.std() > 0 else 0
    dd     = (t.cumsum() + INITIAL).pipe(lambda s: ((s - s.cummax()) / s.cummax()).min())
    ann    = (equity / INITIAL) ** (1 / yrs) - 1
    return {"n": len(t), "wr": (t > 0).mean(), "sharpe": sharpe, "dd": dd, "ann": ann}


print(f"MACD({FAST},{SLOW},{SIGNAL}) H1 backtest — {len(PAIRS)} pairs\n")
print(f"{'Pair':<14} {'5y Sh':>7} {'5y N':>6} {'10y Sh':>7} {'10y N':>6} {'WR':>5} {'DD':>7} {'Ann%':>7}  Verdict")
print("-" * 75)

good = []
for pair in PAIRS:
    r5  = backtest(pair, "5y")
    r10 = backtest(pair, "10y")
    sh5  = r5["sharpe"]  if r5  else float("nan")
    sh10 = r10["sharpe"] if r10 else float("nan")
    n5   = r5["n"]       if r5  else 0
    n10  = r10["n"]      if r10 else 0
    wr   = r10["wr"]     if r10 else float("nan")
    dd   = r10["dd"]     if r10 else float("nan")
    ann  = r10["ann"]    if r10 else float("nan")

    verdict = "PASS" if (not np.isnan(sh5) and sh5 > 0) and (not np.isnan(sh10) and sh10 > 0) else \
              "FAIL" if (not np.isnan(sh5) and sh5 < 0) or (not np.isnan(sh10) and sh10 < 0) else "MARGINAL"
    if verdict == "PASS":
        good.append(pair)

    s5  = f"{sh5:>7.2f}"  if not np.isnan(sh5)  else "     --"
    s10 = f"{sh10:>7.2f}" if not np.isnan(sh10) else "     --"
    w   = f"{wr:>5.0%}"   if not np.isnan(wr)   else "   --"
    d   = f"{dd:>7.1%}"   if not np.isnan(dd)   else "     --"
    a   = f"{ann:>7.1%}"  if not np.isnan(ann)  else "     --"
    print(f"{pair:<14} {s5} {n5:>6} {s10} {n10:>6} {w} {d} {a}  {verdict}")

print(f"\nPassing pairs ({len(good)}): {', '.join(good)}")
print(f"Expected trades/month across passing pairs: ~{len(good) * 5}")
