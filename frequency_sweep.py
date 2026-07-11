"""Sweep across timeframes to find where trade frequency meets positive edge.
Same EMA crossover signal at each timeframe, fixed 1% risk with MAX_LEVERAGE=5 cap.
"""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import atr, ema

FAST       = 10
SLOW       = 30
SL_MULT    = 1.5
TP_MULT    = 2.5
RISK_PCT   = 0.01
MAX_LEV    = 5.0
INITIAL_EQ = 10_000
PAIRS      = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_JPY"]

INTERVALS  = [
    ("M5",  "5m"),
    ("M15", "15m"),
    ("M30", "30m"),
    ("H1",  "1h"),
    ("H4",  "4h"),
]

def backtest(pair, interval_oanda, period="2y"):
    df = load_oanda_data(pair, period=period, interval=interval_oanda)
    if df is None or len(df) < SLOW + 5:
        return None

    fast_s = ema(df["Close"], FAST)
    slow_s = ema(df["Close"], SLOW)
    atr_s  = atr(df["High"], df["Low"], df["Close"], 14)

    equity = INITIAL_EQ
    position = 0
    entry_price = sl = tp = 0.0
    trades = []

    for i in range(1, len(df)):
        price    = df["Close"].iloc[i]
        fast_now = fast_s.iloc[i];  fast_prev = fast_s.iloc[i-1]
        slow_now = slow_s.iloc[i];  slow_prev = slow_s.iloc[i-1]
        av       = atr_s.iloc[i]

        if np.isnan(av) or av <= 0:
            continue

        if position != 0:
            sl_hit = (position==1 and price<=sl) or (position==-1 and price>=sl)
            tp_hit = (position==1 and price>=tp) or (position==-1 and price<=tp)
            if sl_hit or tp_hit:
                exit_p = sl if sl_hit else tp
                pnl_r  = position * (exit_p - entry_price) / abs(entry_price - sl)
                trades.append(pnl_r * equity * RISK_PCT)
                equity += trades[-1]
                position = 0

        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_dn = fast_prev >= slow_prev and fast_now < slow_now
        if not (cross_up or cross_dn) or position != 0:
            continue

        stop_dist    = SL_MULT * av
        risk_amount  = equity * RISK_PCT
        risk_units   = risk_amount / stop_dist
        max_units    = (equity * MAX_LEV) / price
        units        = min(risk_units, max_units)
        if units <= 0:
            continue

        if cross_up:
            position=1;  entry_price=price; sl=price-stop_dist; tp=price+TP_MULT*av
        else:
            position=-1; entry_price=price; sl=price+stop_dist; tp=price-TP_MULT*av

    if len(trades) < 5:
        return None
    t = pd.Series(trades)
    sharpe = t.mean() / t.std() * np.sqrt(252) if t.std() > 0 else 0
    dd = (t.cumsum()+INITIAL_EQ).pipe(lambda s: ((s-s.cummax())/s.cummax()).min())
    return {"n": len(trades), "wr": (t>0).mean(), "sharpe": sharpe, "dd": dd,
            "pnl": t.sum()}

print(f"\nTimeframe sweep — EMA({FAST}/{SLOW}), SL={SL_MULT}×ATR, TP={TP_MULT}×ATR, 2y, 5 pairs\n")
print(f"{'TF':<6} {'Trades/yr':>10} {'Trades/mo':>10} {'Win%':>6} {'Sharpe':>8} {'MaxDD':>7} {'P&L':>8}")
print("-" * 62)

for label, interval in INTERVALS:
    all_trades = []
    for pair in PAIRS:
        r = backtest(pair, interval)
        if r:
            all_trades.append(r)

    if not all_trades:
        print(f"{label:<6} {'no data':>10}")
        continue

    total_n  = sum(r["n"] for r in all_trades)
    avg_wr   = np.mean([r["wr"] for r in all_trades])
    avg_sh   = np.mean([r["sharpe"] for r in all_trades])
    avg_dd   = np.mean([r["dd"] for r in all_trades])
    total_pnl= sum(r["pnl"] for r in all_trades)
    per_yr   = int(total_n / 2)       # 2y period
    per_mo   = int(per_yr / 12)

    print(f"{label:<6} {per_yr:>10,} {per_mo:>10,} {avg_wr:>6.0%} {avg_sh:>8.2f} {avg_dd:>7.1%} ${total_pnl:>7,.0f}")
