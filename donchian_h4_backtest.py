"""Donchian channel breakout on H4 bars. 20-period channel.
Entry: close above upper channel → BUY, close below lower → SELL.
Exit: ATR trailing stop or 10-period exit channel.
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import atr

ENTRY_N  = 20    # channel period for entry
EXIT_N   = 10    # shorter channel for exit
SL_MULT  = 2.0   # initial ATR stop
TRAIL_MULT = 3.0 # trailing ATR stop
RISK_PCT = 0.01
MAX_LEV  = 5.0
INITIAL  = 10_000

PAIRS = [
    "GBP_USD", "EUR_JPY", "CHF_JPY", "CAD_JPY", "AUD_JPY",
    "GBP_JPY", "NZD_JPY", "AUD_CHF", "EUR_AUD", "AUD_SGD",
    "EUR_USD", "USD_JPY", "EUR_CAD", "GBP_CAD", "GBP_CHF",
    "EUR_USD", "AUD_USD", "NZD_USD",
]
PAIRS = list(dict.fromkeys(PAIRS))  # deduplicate


def backtest(pair, period):
    df = load_oanda_data(pair, period=period, interval="4h")
    if df is None or len(df) < ENTRY_N + 5:
        return None

    atr_s    = atr(df["High"], df["Low"], df["Close"], 14)
    upper    = df["High"].shift(1).rolling(ENTRY_N).max()  # prior-bar channel avoids look-ahead
    lower    = df["Low"].shift(1).rolling(ENTRY_N).min()
    ex_high  = df["High"].shift(1).rolling(EXIT_N).max()
    ex_low   = df["Low"].shift(1).rolling(EXIT_N).min()

    equity   = INITIAL
    position = 0
    entry_price = sl = trail = 0.0
    trades   = []

    for i in range(ENTRY_N + 1, len(df)):
        price  = df["Close"].iloc[i]
        av     = atr_s.iloc[i]
        u      = upper.iloc[i]
        lo     = lower.iloc[i]
        ex_h   = ex_high.iloc[i]
        ex_l   = ex_low.iloc[i]

        if np.isnan(av) or np.isnan(u) or av <= 0:
            continue

        stop_dist = SL_MULT * av

        if position != 0:
            # Update trailing stop
            if position == 1:
                trail = max(trail, price - TRAIL_MULT * av)
                if price <= trail or price <= ex_l:
                    pnl_r = (price - entry_price) / stop_dist
                    trades.append(pnl_r * equity * RISK_PCT)
                    equity += trades[-1]
                    position = 0
            else:
                trail = min(trail, price + TRAIL_MULT * av)
                if price >= trail or price >= ex_h:
                    pnl_r = (entry_price - price) / stop_dist
                    trades.append(pnl_r * equity * RISK_PCT)
                    equity += trades[-1]
                    position = 0
            if position != 0:
                continue

        units = min(equity * RISK_PCT / stop_dist, equity * MAX_LEV / price)
        if units <= 0:
            continue

        if price > u and position == 0:
            position = 1
            entry_price = price
            sl = price - stop_dist
            trail = price - TRAIL_MULT * av
        elif price < lo and position == 0:
            position = -1
            entry_price = price
            sl = price + stop_dist
            trail = price + TRAIL_MULT * av

    if len(trades) < 10:
        return None
    t   = pd.Series(trades)
    yrs = int(period[0])
    sharpe = t.mean() / t.std() * np.sqrt(252 * 6) if t.std() > 0 else 0  # 6 H4 bars/day
    dd     = (t.cumsum() + INITIAL).pipe(lambda s: ((s - s.cummax()) / s.cummax()).min())
    ann    = (equity / INITIAL) ** (1 / yrs) - 1
    return {"n": len(t), "wr": (t > 0).mean(), "sharpe": sharpe, "dd": dd, "ann": ann}


print(f"Donchian Breakout H4 (entry={ENTRY_N}, exit={EXIT_N}) — {len(PAIRS)} pairs\n")
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
print(f"Expected trades/month across passing pairs: ~{len(good) * 3}")
