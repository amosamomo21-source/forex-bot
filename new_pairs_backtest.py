from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import atr, ema, rsi, rolling_std, sma, BollingerMeanReversionTrendFilter as Strat

def backtest_m30(instrument, period="5y"):
    daily = load_oanda_data(instrument, period=period, interval="1d")
    m30   = load_oanda_data(instrument, period=period, interval="30m")
    if daily is None or m30 is None or len(daily) < Strat.trend_period + 5:
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

    equity = 10_000
    position = 0
    entry_price = sl = tp = 0.0
    trades = []

    for i in range(1, len(m30)):
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
    dd = (t.cumsum() + 10_000).pipe(lambda s: ((s - s.cummax()) / s.cummax()).min())
    ann = (equity / 10_000) ** (1 / int(period[0])) - 1
    return {"inst": instrument, "period": period, "n": len(t),
            "wr": (t > 0).mean(), "sharpe": sharpe, "ann": ann, "dd": dd}

print(f"\n{'--- BbmrtM30 on new pairs ---':}")
print(f"{'Instrument':<14} {'Period':<6} {'Trades':>6} {'Win%':>6} {'Sharpe':>7} {'Ann%':>7} {'MaxDD':>7}")
print("-" * 60)
for inst in ["EUR_JPY", "USD_CAD", "GBP_JPY"]:
    for period in ["5y", "10y"]:
        r = backtest_m30(inst, period)
        if r:
            print(f"{r['inst']:<14} {r['period']:<6} {r['n']:>6} {r['wr']:>6.0%} {r['sharpe']:>7.2f} {r['ann']:>7.1%} {r['dd']:>7.1%}")
        else:
            print(f"{inst:<14} {period:<6} {'no trades':>40}")
