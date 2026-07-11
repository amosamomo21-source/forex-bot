"""Screen all 68 OANDA forex pairs for BbmrtM30 edge.
WARNING: this is a screening exercise only. Top results need separate
out-of-sample confirmation before being trusted -- multiple testing inflates
apparent Sharpe on the best-looking pairs.
"""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import oandapyV20.endpoints.accounts as accounts

import broker
from data import load_oanda_data
from strategies import atr, ema, rsi, rolling_std, sma, BollingerMeanReversionTrendFilter as Strat


def backtest_m30(instrument, period="5y"):
    try:
        daily = load_oanda_data(instrument, period=period, interval="1d")
        m30   = load_oanda_data(instrument, period=period, interval="30m")
    except Exception:
        return None
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

    if len(trades) < 3:
        return None
    t = pd.Series(trades)
    sharpe = t.mean() / t.std() * np.sqrt(252) if t.std() > 0 else 0
    dd = (t.cumsum() + 10_000).pipe(lambda s: ((s - s.cummax()) / s.cummax()).min())
    ann = (equity / 10_000) ** (1 / int(period[0])) - 1
    return {"inst": instrument, "period": period, "n": len(t),
            "wr": (t > 0).mean(), "sharpe": sharpe, "ann": ann, "dd": dd}


# get full instrument list
b = broker.from_env()
r = accounts.AccountInstruments(accountID=b.account_id)
b.api.request(r)
all_pairs = sorted(i['name'] for i in r.response['instruments'] if i['type'] == 'CURRENCY')

already_live = {"EUR_USD", "GBP_USD", "XAU_USD"}

results_5y  = {}
results_10y = {}

print(f"Screening {len(all_pairs)} pairs...")
for i, inst in enumerate(all_pairs, 1):
    print(f"  [{i}/{len(all_pairs)}] {inst}", flush=True)
    r5  = backtest_m30(inst, "5y")
    r10 = backtest_m30(inst, "10y")
    if r5:  results_5y[inst]  = r5
    if r10: results_10y[inst] = r10

# merge and sort by 10y Sharpe
rows = []
for inst in all_pairs:
    s5  = results_5y.get(inst)
    s10 = results_10y.get(inst)
    if s5 is None and s10 is None:
        continue
    rows.append({
        "inst":     inst,
        "live":     "LIVE" if inst in already_live else "",
        "5y_sh":    s5["sharpe"]  if s5  else float("nan"),
        "5y_n":     s5["n"]       if s5  else 0,
        "10y_sh":   s10["sharpe"] if s10 else float("nan"),
        "10y_n":    s10["n"]      if s10 else 0,
        "10y_wr":   s10["wr"]     if s10 else float("nan"),
        "10y_dd":   s10["dd"]     if s10 else float("nan"),
    })

df = pd.DataFrame(rows).sort_values("10y_sh", ascending=False)

print(f"\n{'Instrument':<14} {'Live':<5} {'5y Sh':>7} {'5y N':>5} {'10y Sh':>7} {'10y N':>5} {'WR':>5} {'MaxDD':>7}")
print("-" * 65)
for _, row in df.iterrows():
    s5  = f"{row['5y_sh']:>7.2f}"  if not np.isnan(row['5y_sh'])  else "     --"
    s10 = f"{row['10y_sh']:>7.2f}" if not np.isnan(row['10y_sh']) else "     --"
    wr  = f"{row['10y_wr']:>5.0%}" if not np.isnan(row['10y_wr']) else "   --"
    dd  = f"{row['10y_dd']:>7.1%}" if not np.isnan(row['10y_dd']) else "     --"
    print(f"{row['inst']:<14} {row['live']:<5} {s5} {int(row['5y_n']):>5} {s10} {int(row['10y_n']):>5} {wr} {dd}")
