"""1-year out-of-sample confirmation for top BbmrtM30 candidates.
Data up to 1 year ago = in-sample (used for screening). Last 1 year = never seen before.
"""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import atr, ema, rsi, rolling_std, sma, BollingerMeanReversionTrendFilter as Strat


def backtest_m30(instrument, period):
    daily = load_oanda_data(instrument, period=period, interval="1d")
    m30   = load_oanda_data(instrument, period=period, interval="30m")
    if daily is None or m30 is None or len(daily) < Strat.trend_period + 5 or len(m30) < 50:
        return None

    bb_mid = sma(daily["Close"], Strat.bb_period)
    bb_std = rolling_std(daily["Close"], Strat.bb_period)
    trend  = ema(daily["Close"], Strat.trend_period)
    atr_d  = atr(daily["High"], daily["Low"], daily["Close"], Strat.atr_period)
    lower  = bb_mid - Strat.bb_k * bb_std
    upper  = bb_mid + Strat.bb_k * bb_std

    long_cond  = (daily["Close"].shift(1) < lower.shift(1)) & (daily["Close"].shift(1) > trend.shift(1))
    short_cond = (daily["Close"].shift(1) > upper.shift(1)) & (daily["Close"].shift(1) < trend.shift(1))
    bias = pd.Series(0, index=daily.index)
    bias[long_cond]  =  1
    bias[short_cond] = -1

    daily_bias = bias.reindex(m30.index, method="ffill")
    daily_atr  = atr_d.shift(1).reindex(m30.index, method="ffill")
    daily_mid  = bb_mid.shift(1).reindex(m30.index, method="ffill")

    m30_rsi  = rsi(m30["Close"], 14)
    rsi_prev = m30_rsi.shift(1)

    # restrict to last 1 year only
    cutoff = m30.index[-1] - pd.DateOffset(years=1)
    mask   = m30.index >= cutoff
    start_i = mask.argmax()

    equity = 10_000
    position = 0
    entry_price = sl = tp = 0.0
    trades = []

    for i in range(max(1, start_i), len(m30)):
        price = m30["Close"].iloc[i]
        b     = daily_bias.iloc[i]
        a     = daily_atr.iloc[i]
        mid   = daily_mid.iloc[i]
        rn    = m30_rsi.iloc[i]
        rp    = rsi_prev.iloc[i]

        if np.isnan(a) or a <= 0 or np.isnan(rn) or np.isnan(rp):
            continue

        if position != 0:
            sl_hit   = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            tp_hit   = (position == 1 and price >= tp) or (position == -1 and price <= tp)
            mean_hit = (position == 1 and price >= mid) or (position == -1 and price <= mid)
            if sl_hit or tp_hit or mean_hit:
                exit_p = sl if sl_hit else (tp if tp_hit else price)
                pnl_r  = position * (exit_p - entry_price) / abs(entry_price - sl)
                trades.append(pnl_r * equity * Strat.risk_pct)
                equity += trades[-1]
                position = 0
            continue

        stop_dist = 2.0 * a
        if b == 1 and rp < 40 and rn >= 40:
            position = 1; entry_price = price; sl = price - stop_dist; tp = price + 3.0 * a
        elif b == -1 and rp > 60 and rn <= 60:
            position = -1; entry_price = price; sl = price + stop_dist; tp = price - 3.0 * a

    if not trades:
        return None
    t = pd.Series(trades)
    sharpe = t.mean() / t.std() * np.sqrt(252) if t.std() > 0 else 0
    total_pnl = t.sum()
    return {"inst": instrument, "n": len(t), "wr": (t > 0).mean(),
            "sharpe": sharpe, "pnl": total_pnl}


candidates = ["CHF_JPY", "EUR_CAD", "EUR_JPY", "AUD_CHF"]

print("1-year out-of-sample test (last 12 months, never seen in screening)")
print(f"{'Instrument':<14} {'Trades':>6} {'Win%':>6} {'Sharpe':>7} {'P&L (R)':>8}  Verdict")
print("-" * 60)
for inst in candidates:
    r = backtest_m30(inst, "5y")
    if r:
        verdict = "PASS" if r["sharpe"] > 0 and r["n"] >= 3 else "FAIL" if r["sharpe"] < 0 else "MARGINAL"
        print(f"{inst:<14} {r['n']:>6} {r['wr']:>6.0%} {r['sharpe']:>7.2f} {r['pnl']:>8.1f}  {verdict}")
    else:
        print(f"{inst:<14} {'no trades':>6}                              INSUFFICIENT DATA")
