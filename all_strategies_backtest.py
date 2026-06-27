"""
All-strategies backtest — matches current live parameters exactly.

H1 EMA  : trailing stop (TRAIL_MULT=2.0), volatility filter (MIN_ATR_PCT=0.0008),
           weekly EMA-10 trend filter
MACD H1 : trailing stop (TRAIL_MULT=2.0), volatility filter
ORB M30 : London 08:00 + NY 13:00 UTC, TP=1.5x range
PDH/PDL : SL=range midpoint, TP=1.5x range
M30 BBMRT: daily BB(20,1.5)+EMA(100) bias, M30 RSI trigger
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import ema, atr, sma, rolling_std, rsi, macd

RISK_PCT     = 0.01
MAX_LEV      = 5.0
INITIAL      = 10_000
SL_MULT      = 1.5
TRAIL_MULT   = 2.0
MIN_ATR_PCT  = 0.0008
W_EMA_PERIOD = 10


def _result_row(trades, period, initial=INITIAL):
    if len(trades) < 10:
        return None
    t   = pd.Series(trades)
    yrs = int(period[0])
    sharpe = t.mean() / t.std() * np.sqrt(252) if t.std() > 0 else 0
    dd     = (t.cumsum() + initial).pipe(lambda s: ((s - s.cummax()) / s.cummax()).min())
    equity = initial + t.sum()
    ann    = (equity / initial) ** (1 / yrs) - 1
    return {"n": len(t), "wr": (t > 0).mean(), "sharpe": sharpe, "dd": dd, "ann": ann}


def _print_table(results, header):
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")
    print(f"{'Pair':<16} {'5y Sh':>7} {'5y N':>6} {'10y Sh':>7} {'10y N':>6} {'WR':>5} {'DD':>7} {'Ann%':>7}  Verdict")
    print("-" * 70)
    good = []
    for pair, r5, r10 in results:
        sh5  = r5["sharpe"]  if r5  else float("nan")
        sh10 = r10["sharpe"] if r10 else float("nan")
        n5   = r5["n"]       if r5  else 0
        n10  = r10["n"]      if r10 else 0
        wr   = r10["wr"]     if r10 else float("nan")
        dd   = r10["dd"]     if r10 else float("nan")
        ann  = r10["ann"]    if r10 else float("nan")
        if np.isnan(sh5) and np.isnan(sh10):
            print(f"{pair:<16}  -- no data"); continue
        verdict = "PASS"    if (not np.isnan(sh5) and sh5 > 0) and (not np.isnan(sh10) and sh10 > 0) else \
                  "FAIL"    if (not np.isnan(sh5) and sh5 < 0) or (not np.isnan(sh10) and sh10 < 0) else \
                  "MARGINAL"
        if verdict == "PASS":
            good.append(pair)
        s5 = f"{sh5:>7.2f}" if not np.isnan(sh5) else "     --"
        s10= f"{sh10:>7.2f}"if not np.isnan(sh10) else "     --"
        w  = f"{wr:>5.0%}"  if not np.isnan(wr)  else "   --"
        d  = f"{dd:>7.1%}"  if not np.isnan(dd)  else "     --"
        a  = f"{ann:>7.1%}" if not np.isnan(ann) else "     --"
        print(f"{pair:<16} {s5} {n5:>6} {s10} {n10:>6} {w} {d} {a}  {verdict}")
    print(f"\nPASSING ({len(good)}): {', '.join(good)}")
    return good


# ─────────────────────────────────────────────────────────────
# 1. H1 EMA  (trailing stop + vol filter + weekly trend filter)
# ─────────────────────────────────────────────────────────────

H1_EMA_PAIRS = [
    "GBP_USD", "EUR_JPY", "CHF_JPY", "CAD_JPY", "AUD_JPY",
    "GBP_JPY", "NZD_JPY", "AUD_CHF", "EUR_AUD", "AUD_SGD",
    "WTICO_USD", "BCO_USD", "XAU_USD", "XAG_USD", "XCU_USD",
    "XPT_USD", "NATGAS_USD", "CORN_USD", "SOYBN_USD", "WHEAT_USD", "SUGAR_USD",
    "SPX500_USD", "NAS100_USD", "US30_USD", "US2000_USD",
    "DE30_EUR", "EU50_EUR", "JP225_USD", "AU200_AUD",
]


def _h1_ema_backtest(pair, period):
    df = load_oanda_data(pair, period=period, interval="1h")
    if df is None or len(df) < 35:
        return None

    fast_s = ema(df["Close"], 10)
    slow_s = ema(df["Close"], 30)
    atr_s  = atr(df["High"], df["Low"], df["Close"], 14)

    # Weekly trend filter: resample H1 → weekly, compute EMA(10)
    try:
        weekly_close = df["Close"].resample("W").last().dropna()
        w_ema = ema(weekly_close, W_EMA_PERIOD)
        w_ema_prev = w_ema.shift(1)
        w_rising  = (w_ema > w_ema_prev).reindex(df.index, method="ffill").fillna(False)
        w_falling = (w_ema < w_ema_prev).reindex(df.index, method="ffill").fillna(False)
    except Exception:
        w_rising  = pd.Series(True,  index=df.index)
        w_falling = pd.Series(True,  index=df.index)

    equity = INITIAL; position = 0; entry_price = sl = initial_sl_dist = 0.0; trades = []

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        av    = atr_s.iloc[i]
        if np.isnan(av) or av <= 0:
            continue

        if position != 0:
            # Trail the stop
            if position == 1:
                new_sl = price - TRAIL_MULT * av
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = price + TRAIL_MULT * av
                if new_sl < sl:
                    sl = new_sl
            sl_hit = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            if sl_hit:
                pnl_r = position * (price - entry_price) / initial_sl_dist if initial_sl_dist > 0 else 0
                trades.append(pnl_r * equity * RISK_PCT)
                equity += trades[-1]; position = 0
            continue

        fn, fp = fast_s.iloc[i], fast_s.iloc[i-1]
        sn, sp = slow_s.iloc[i], slow_s.iloc[i-1]
        cross_up = fp <= sp and fn > sn
        cross_dn = fp >= sp and fn < sn
        if not (cross_up or cross_dn):
            continue
        if av < price * MIN_ATR_PCT:
            continue

        sd = SL_MULT * av
        units = min(equity * RISK_PCT / sd, equity * MAX_LEV / price)
        if units <= 0:
            continue

        if cross_up and w_rising.iloc[i]:
            position = 1;  entry_price = price; sl = price - sd; initial_sl_dist = sd
        elif cross_dn and w_falling.iloc[i]:
            position = -1; entry_price = price; sl = price + sd; initial_sl_dist = sd

    return _result_row(trades, period)


print("Running H1 EMA backtest (trailing stop + vol filter + weekly filter)...")
h1_ema_results = []
for pair in H1_EMA_PAIRS:
    r5  = _h1_ema_backtest(pair, "5y")
    r10 = _h1_ema_backtest(pair, "10y")
    h1_ema_results.append((pair, r5, r10))
    print(f"  {pair} done")

h1_ema_good = _print_table(h1_ema_results, "H1 EMA (10/30) — trailing stop, vol filter, weekly filter [29 live pairs]")


# ─────────────────────────────────────────────────────────────
# 2. MACD H1  (trailing stop + vol filter)
# ─────────────────────────────────────────────────────────────

MACD_PAIRS = ["USD_JPY", "EUR_CAD", "GBP_CAD", "GBP_CHF"]


def _macd_backtest(pair, period):
    df = load_oanda_data(pair, period=period, interval="1h")
    if df is None or len(df) < 40:
        return None

    ml, sig = macd(df["Close"], 12, 26, 9)
    atr_s   = atr(df["High"], df["Low"], df["Close"], 14)

    equity = INITIAL; position = 0; entry_price = sl = initial_sl_dist = 0.0; trades = []

    for i in range(1, len(df)):
        price   = df["Close"].iloc[i]
        av      = atr_s.iloc[i]
        ml_now  = ml.iloc[i];  ml_prev  = ml.iloc[i-1]
        sig_now = sig.iloc[i]; sig_prev = sig.iloc[i-1]
        if np.isnan(av) or av <= 0 or np.isnan(ml_now) or np.isnan(sig_now):
            continue

        if position != 0:
            if position == 1:
                new_sl = price - TRAIL_MULT * av
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = price + TRAIL_MULT * av
                if new_sl < sl:
                    sl = new_sl
            sl_hit = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            if sl_hit:
                pnl_r = position * (price - entry_price) / initial_sl_dist if initial_sl_dist > 0 else 0
                trades.append(pnl_r * equity * RISK_PCT)
                equity += trades[-1]; position = 0
            continue

        cross_up = ml_prev <= sig_prev and ml_now > sig_now
        cross_dn = ml_prev >= sig_prev and ml_now < sig_now
        if not (cross_up or cross_dn):
            continue
        if av < price * MIN_ATR_PCT:
            continue

        sd = SL_MULT * av
        units = min(equity * RISK_PCT / sd, equity * MAX_LEV / price)
        if units <= 0:
            continue

        if cross_up:
            position = 1;  entry_price = price; sl = price - sd; initial_sl_dist = sd
        else:
            position = -1; entry_price = price; sl = price + sd; initial_sl_dist = sd

    return _result_row(trades, period)


print("\nRunning MACD H1 backtest (trailing stop + vol filter)...")
macd_results = []
for pair in MACD_PAIRS:
    r5  = _macd_backtest(pair, "5y")
    r10 = _macd_backtest(pair, "10y")
    macd_results.append((pair, r5, r10))
    print(f"  {pair} done")

macd_good = _print_table(macd_results, "MACD H1 (12,26,9) — trailing stop, vol filter [4 live pairs]")


# ─────────────────────────────────────────────────────────────
# 3. ORB M30  (unchanged from live)
# ─────────────────────────────────────────────────────────────

ORB_PAIRS = [
    "EUR_JPY", "CHF_JPY", "CAD_JPY", "AUD_JPY", "GBP_JPY",
    "NZD_JPY", "AUD_CHF", "EUR_AUD", "USD_JPY", "EUR_CAD",
]
SESSION_HOURS = {8, 13}
ORB_TP_MULT   = 1.5


def _orb_backtest(pair, period):
    df = load_oanda_data(pair, period=period, interval="30m")
    if df is None or len(df) < 100:
        return None

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []

    for i in range(1, len(df)):
        price     = df["Close"].iloc[i]
        prev_hour = df.index[i - 1].hour

        if position != 0:
            sl_hit = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            tp_hit = (position == 1 and price >= tp) or (position == -1 and price <= tp)
            if sl_hit or tp_hit:
                exit_p = sl if sl_hit else tp
                pnl_r  = position * (exit_p - entry_price) / abs(entry_price - sl)
                trades.append(pnl_r * equity * RISK_PCT)
                equity += trades[-1]; position = 0

        if prev_hour in SESSION_HOURS and position == 0:
            or_high = df["High"].iloc[i - 1]
            or_low  = df["Low"].iloc[i - 1]
            rng     = or_high - or_low
            if rng <= 0:
                continue
            units = min(equity * RISK_PCT / rng, equity * MAX_LEV / price)
            if units <= 0:
                continue
            if price > or_high:
                position = 1;  entry_price = price; sl = or_low;  tp = price + ORB_TP_MULT * rng
            elif price < or_low:
                position = -1; entry_price = price; sl = or_high; tp = price - ORB_TP_MULT * rng

    return _result_row(trades, period)


print("\nRunning ORB M30 backtest...")
orb_results = []
for pair in ORB_PAIRS:
    r5  = _orb_backtest(pair, "5y")
    r10 = _orb_backtest(pair, "10y")
    orb_results.append((pair, r5, r10))
    print(f"  {pair} done")

orb_good = _print_table(orb_results, "ORB M30 — London 08:00 + NY 13:00 UTC [10 live pairs]")


# ─────────────────────────────────────────────────────────────
# 4. PDH/PDL H1  (unchanged from live)
# ─────────────────────────────────────────────────────────────

PDHL_PAIRS = [
    "GBP_USD", "EUR_JPY", "CHF_JPY", "AUD_JPY", "GBP_JPY",
    "NZD_JPY", "USD_JPY", "BCO_USD", "XAU_USD", "XAG_USD",
    "NATGAS_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]
PDHL_TP_MULT = 1.5


def _pdhl_backtest(pair, period):
    try:
        h1 = load_oanda_data(pair, period=period, interval="1h")
        d1 = load_oanda_data(pair, period=period, interval="1d")
    except Exception:
        return None
    if h1 is None or d1 is None or len(h1) < 50 or len(d1) < 10:
        return None

    d1 = d1.copy()
    d1["prev_high"] = d1["High"].shift(1)
    d1["prev_low"]  = d1["Low"].shift(1)
    d1["prev_mid"]  = (d1["prev_high"] + d1["prev_low"]) / 2
    d1["prev_rng"]  = d1["prev_high"] - d1["prev_low"]
    d1_dict = d1[["prev_high", "prev_low", "prev_mid", "prev_rng"]].to_dict("index")

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []
    day_long = None; day_short = None

    for i in range(1, len(h1)):
        price = h1["Close"].iloc[i]
        date  = h1.index[i].date()
        candidates = [k for k in d1_dict if k.date() == date]
        if not candidates:
            continue
        prev = d1_dict[candidates[0]]
        ph, pl, pm, pr = prev["prev_high"], prev["prev_low"], prev["prev_mid"], prev["prev_rng"]
        if np.isnan(ph) or pr <= 0:
            continue

        if position != 0:
            sl_hit = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            tp_hit = (position == 1 and price >= tp) or (position == -1 and price <= tp)
            if sl_hit or tp_hit:
                exit_p = sl if sl_hit else tp
                pnl_r  = position * (exit_p - entry_price) / abs(entry_price - sl)
                trades.append(pnl_r * equity * RISK_PCT)
                equity += trades[-1]; position = 0

        if price > ph and position == 0 and day_long != date:
            sd = abs(price - pm)
            if sd <= 0: continue
            units = min(equity * RISK_PCT / sd, equity * MAX_LEV / price)
            if units <= 0: continue
            position = 1;  entry_price = price; sl = pm; tp = price + PDHL_TP_MULT * pr
            day_long = date
        elif price < pl and position == 0 and day_short != date:
            sd = abs(price - pm)
            if sd <= 0: continue
            units = min(equity * RISK_PCT / sd, equity * MAX_LEV / price)
            if units <= 0: continue
            position = -1; entry_price = price; sl = pm; tp = price - PDHL_TP_MULT * pr
            day_short = date

    return _result_row(trades, period)


print("\nRunning PDH/PDL H1 backtest...")
pdhl_results = []
for pair in PDHL_PAIRS:
    r5  = _pdhl_backtest(pair, "5y")
    r10 = _pdhl_backtest(pair, "10y")
    pdhl_results.append((pair, r5, r10))
    print(f"  {pair} done")

pdhl_good = _print_table(pdhl_results, "PDH/PDL H1 — SL=range mid, TP=1.5x range [14 live pairs]")


# ─────────────────────────────────────────────────────────────
# 5. M30 BBMRT  (daily bias + M30 RSI trigger)
# ─────────────────────────────────────────────────────────────

M30_PAIRS = [
    "EUR_USD", "GBP_USD", "EUR_CAD", "EUR_JPY", "CHF_JPY",
    "AUD_CHF", "EUR_SGD", "GBP_AUD", "CAD_JPY", "AUD_SGD",
    "EUR_AUD", "GBP_CAD", "GBP_SGD", "GBP_JPY", "GBP_CHF",
    "AUD_JPY", "NZD_JPY",
]
BB_K     = 1.5
BB_PER   = 20
TREND_P  = 100
ATR_P    = 14
M30_SL   = 2.0
M30_TP   = 3.0
M30_RISK = 0.01


def _m30_backtest(pair, period):
    daily = load_oanda_data(pair, period=period, interval="1d")
    m30   = load_oanda_data(pair, period=period, interval="30m")
    if daily is None or m30 is None or len(daily) < TREND_P + 5:
        return None

    bb_mid = sma(daily["Close"], BB_PER)
    bb_std = rolling_std(daily["Close"], BB_PER)
    trend  = ema(daily["Close"], TREND_P)
    atr_d  = atr(daily["High"], daily["Low"], daily["Close"], ATR_P)
    lower  = bb_mid - BB_K * bb_std
    upper  = bb_mid + BB_K * bb_std

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

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []

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
            sl_hit   = (position ==  1 and price <= sl) or (position == -1 and price >= sl)
            tp_hit   = (position ==  1 and price >= tp) or (position == -1 and price <= tp)
            mean_hit = (position ==  1 and price >= mid) or (position == -1 and price <= mid)
            if sl_hit or tp_hit or mean_hit:
                exit_p = sl if sl_hit else (tp if tp_hit else price)
                pnl_r  = position * (exit_p - entry_price) / abs(entry_price - sl)
                trades.append(pnl_r * equity * M30_RISK)
                equity += trades[-1]; position = 0
            continue

        stop_dist = M30_SL * a
        if b == 1 and rp < 40 and rn >= 40:
            position = 1;  entry_price = price; sl = price - stop_dist; tp = price + M30_TP * a
        elif b == -1 and rp > 60 and rn <= 60:
            position = -1; entry_price = price; sl = price + stop_dist; tp = price - M30_TP * a

    return _result_row(trades, period)


print("\nRunning M30 BBMRT backtest...")
m30_results = []
for pair in M30_PAIRS:
    r5  = _m30_backtest(pair, "5y")
    r10 = _m30_backtest(pair, "10y")
    m30_results.append((pair, r5, r10))
    print(f"  {pair} done")

m30_good = _print_table(m30_results, "M30 BBMRT — daily BB(20,1.5)+EMA(100) bias, M30 RSI trigger [17 live pairs]")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("MASTER SUMMARY")
print("="*70)
total_live = 5 + 17 + 29 + 4 + 10 + 14  # daily BBMRT not retested here
print(f"Daily BBMRT+EMA  :  5 sleeves  (not re-run — uses backtesting library)")
print(f"M30 BBMRT        : {len(m30_good):>2}/{len(M30_PAIRS)} pairs PASS  | live: 17 sleeves")
print(f"H1 EMA           : {len(h1_ema_good):>2}/{len(H1_EMA_PAIRS)} pairs PASS  | live: 29 sleeves")
print(f"MACD H1          : {len(macd_good):>2}/{len(MACD_PAIRS)} pairs PASS  | live:  4 sleeves")
print(f"ORB M30          : {len(orb_good):>2}/{len(ORB_PAIRS)} pairs PASS  | live: 10 sleeves")
print(f"PDH/PDL H1       : {len(pdhl_good):>2}/{len(PDHL_PAIRS)} pairs PASS  | live: 14 sleeves")
total_pass = len(m30_good) + len(h1_ema_good) + len(macd_good) + len(orb_good) + len(pdhl_good)
print(f"\nTotal tested: {total_pass} pass  (+ 5 daily BBMRT not re-run)")
