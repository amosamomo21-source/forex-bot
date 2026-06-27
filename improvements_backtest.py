"""
Backtest comparing 4 variants of H1 EMA strategy across all 13 live pairs:
  BASELINE : trailing stop + vol filter + weekly filter  (current live)
  +ADX     : baseline + ADX(14) > 20 entry gate
  +BE      : baseline + break-even stop (SL floored at entry after 1R profit)
  +BOTH    : baseline + ADX + break-even

Shows per-pair Sharpe for each variant, then aggregate summary.
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import ema, atr, adx as adx_fn

RISK_PCT     = 0.01
MAX_LEV      = 5.0
INITIAL      = 10_000
SL_MULT      = 1.5
TRAIL_MULT   = 2.0
MIN_ATR_PCT  = 0.0008
W_EMA_PERIOD = 10
ADX_MIN      = 20

PAIRS = [
    "GBP_USD", "CAD_JPY", "AUD_JPY",
    "WTICO_USD", "BCO_USD", "XAU_USD", "XAG_USD", "NATGAS_USD", "CORN_USD", "WHEAT_USD",
    "NAS100_USD", "DE30_EUR", "JP225_USD",
]


def _weekly_filter(df):
    try:
        wc = df["Close"].resample("W").last().dropna()
        w = ema(wc, W_EMA_PERIOD)
        wp = w.shift(1)
        rising  = (w > wp).reindex(df.index, method="ffill").fillna(False)
        falling = (w < wp).reindex(df.index, method="ffill").fillna(False)
        return rising, falling
    except Exception:
        t = pd.Series(True, index=df.index)
        return t, t


def backtest(pair, period, use_adx=False, use_be=False):
    df = load_oanda_data(pair, period=period, interval="1h")
    if df is None or len(df) < 35:
        return None

    fast_s = ema(df["Close"], 10)
    slow_s = ema(df["Close"], 30)
    atr_s  = atr(df["High"], df["Low"], df["Close"], 14)
    adx_s  = adx_fn(df["High"], df["Low"], df["Close"], 14) if use_adx else None
    w_rising, w_falling = _weekly_filter(df)

    equity = INITIAL; position = 0
    entry_price = sl = initial_sl_dist = 0.0
    be_triggered = False; trades = []

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        av    = atr_s.iloc[i]
        if np.isnan(av) or av <= 0:
            continue

        if position != 0:
            # Break-even: floor SL at entry once 1R profit hit
            if use_be and not be_triggered:
                if position * (price - entry_price) >= initial_sl_dist:
                    be_triggered = True

            if position == 1:
                new_sl = price - TRAIL_MULT * av
                if use_be and be_triggered:
                    new_sl = max(new_sl, entry_price)
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = price + TRAIL_MULT * av
                if use_be and be_triggered:
                    new_sl = min(new_sl, entry_price)
                if new_sl < sl:
                    sl = new_sl

            sl_hit = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            if sl_hit:
                pnl_r = position * (price - entry_price) / initial_sl_dist if initial_sl_dist > 0 else 0
                trades.append(pnl_r * equity * RISK_PCT)
                equity += trades[-1]; position = 0; be_triggered = False
            continue

        fn, fp = fast_s.iloc[i], fast_s.iloc[i-1]
        sn, sp = slow_s.iloc[i], slow_s.iloc[i-1]
        cross_up = fp <= sp and fn > sn
        cross_dn = fp >= sp and fn < sn
        if not (cross_up or cross_dn):
            continue
        if av < price * MIN_ATR_PCT:
            continue
        if use_adx:
            av2 = adx_s.iloc[i]
            if np.isnan(av2) or av2 < ADX_MIN:
                continue
        if cross_up and not w_rising.iloc[i]:
            continue
        if cross_dn and not w_falling.iloc[i]:
            continue

        sd = SL_MULT * av
        units = min(equity * RISK_PCT / sd, equity * MAX_LEV / price)
        if units <= 0:
            continue

        if cross_up:
            position = 1;  entry_price = price; sl = price - sd; initial_sl_dist = sd
        else:
            position = -1; entry_price = price; sl = price + sd; initial_sl_dist = sd

    if len(trades) < 5:
        return None
    t = pd.Series(trades)
    yrs    = int(period[0])
    sharpe = t.mean() / t.std() * np.sqrt(252) if t.std() > 0 else 0
    dd     = (t.cumsum() + INITIAL).pipe(lambda s: ((s - s.cummax()) / s.cummax()).min())
    wr     = (t > 0).mean()
    equity_final = INITIAL + t.sum()
    ann    = (equity_final / INITIAL) ** (1 / yrs) - 1
    return {"n": len(t), "sharpe": sharpe, "dd": dd, "wr": wr, "ann": ann}


VARIANTS = [
    ("BASELINE", dict(use_adx=False, use_be=False)),
    ("+ADX",     dict(use_adx=True,  use_be=False)),
    ("+BE",      dict(use_adx=False, use_be=True)),
    ("+BOTH",    dict(use_adx=True,  use_be=True)),
]
PERIOD = "10y"

# Collect all results
print(f"Running 4 variants × {len(PAIRS)} pairs × 2 periods...\n")
results = {}  # variant -> pair -> result
for vname, vkwargs in VARIANTS:
    results[vname] = {}
    for pair in PAIRS:
        r5  = backtest(pair, "5y", **vkwargs)
        r10 = backtest(pair, "10y", **vkwargs)
        results[vname][pair] = (r5, r10)
    print(f"  {vname} done")

# Per-pair Sharpe comparison table
print(f"\n{'='*78}")
print(f"10y Sharpe by variant (all 13 live H1 EMA pairs)")
print(f"{'='*78}")
print(f"{'Pair':<14} {'BASELINE':>9} {'  +ADX':>9} {'   +BE':>9} {' +BOTH':>9}  Best")
print("-" * 78)
for pair in PAIRS:
    row = {}
    for vname, _ in VARIANTS:
        r10 = results[vname][pair][1]
        row[vname] = r10["sharpe"] if r10 else float("nan")
    best = max(row, key=lambda k: row[k] if not np.isnan(row[k]) else -999)
    vals = " ".join(f"{row[v]:>9.2f}" if not np.isnan(row[v]) else "       --" for v, _ in VARIANTS)
    print(f"{pair:<14} {vals}  {best}")

# Aggregate summary per variant
print(f"\n{'='*78}")
print(f"Aggregate stats (10y, all 13 pairs combined)")
print(f"{'='*78}")
print(f"{'Variant':<10} {'Avg Sharpe':>10} {'Avg WR':>8} {'Avg DD':>8} {'Pairs>0':>8} {'AvgTrades':>10}")
print("-" * 78)
for vname, _ in VARIANTS:
    sharpes, wrs, dds, ns = [], [], [], []
    pass_count = 0
    for pair in PAIRS:
        r10 = results[vname][pair][1]
        if r10:
            sharpes.append(r10["sharpe"])
            wrs.append(r10["wr"])
            dds.append(r10["dd"])
            ns.append(r10["n"])
            if r10["sharpe"] > 0:
                pass_count += 1
    avg_sh = np.mean(sharpes) if sharpes else 0
    avg_wr = np.mean(wrs) if wrs else 0
    avg_dd = np.mean(dds) if dds else 0
    avg_n  = int(np.mean(ns)) if ns else 0
    print(f"{vname:<10} {avg_sh:>10.2f} {avg_wr:>8.1%} {avg_dd:>8.1%} {pass_count:>8}/{len(PAIRS)} {avg_n:>10}")

# Win rate improvement detail
print(f"\n{'='*78}")
print(f"Win rate per pair: BASELINE vs +BOTH")
print(f"{'='*78}")
print(f"{'Pair':<14} {'BASE WR':>9} {'+BOTH WR':>9} {'Trades BASE':>12} {'Trades +BOTH':>13}  Delta WR")
print("-" * 78)
for pair in PAIRS:
    rb = results["BASELINE"][pair][1]
    ra = results["+BOTH"][pair][1]
    bwr = rb["wr"] if rb else float("nan")
    awr = ra["wr"] if ra else float("nan")
    bn  = rb["n"]  if rb else 0
    an  = ra["n"]  if ra else 0
    delta = awr - bwr if not (np.isnan(bwr) or np.isnan(awr)) else float("nan")
    d_str = f"{delta:>+.1%}" if not np.isnan(delta) else "  --"
    print(f"{pair:<14} {bwr:>9.1%} {awr:>9.1%} {bn:>12} {an:>13}  {d_str}")

print("\nDone.")
