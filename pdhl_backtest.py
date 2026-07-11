"""Previous Day High/Low breakout on H1 bars.
Entry when H1 close breaks above yesterday's high (BUY) or below yesterday's low (SELL).
SL = midpoint of yesterday's range. TP = entry + 1.5x yesterday's range.
Only one trade per direction per day.
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import atr

TP_MULT  = 1.5
RISK_PCT = 0.01
MAX_LEV  = 5.0
INITIAL  = 10_000

# Test across the same pairs as H1 EMA
PAIRS = [
    "GBP_USD", "EUR_JPY", "CHF_JPY", "CAD_JPY", "AUD_JPY",
    "GBP_JPY", "NZD_JPY", "AUD_CHF", "EUR_AUD", "AUD_SGD",
    "USD_JPY", "EUR_CAD", "GBP_CAD", "GBP_CHF",
    "WTICO_USD", "BCO_USD", "XAU_USD", "XAG_USD", "NATGAS_USD",
    "SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR",
]


def backtest(pair, period):
    try:
        h1 = load_oanda_data(pair, period=period, interval="1h")
        d1 = load_oanda_data(pair, period=period, interval="1d")
    except Exception:
        return None
    if h1 is None or d1 is None or len(h1) < 50 or len(d1) < 10:
        return None

    # Build a lookup: for each date, yesterday's high/low/mid
    d1 = d1.copy()
    d1["prev_high"] = d1["High"].shift(1)
    d1["prev_low"]  = d1["Low"].shift(1)
    d1["prev_mid"]  = (d1["prev_high"] + d1["prev_low"]) / 2
    d1["prev_rng"]  = d1["prev_high"] - d1["prev_low"]
    d1_dict = d1[["prev_high","prev_low","prev_mid","prev_rng"]].to_dict("index")

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []
    day_traded_long = None; day_traded_short = None

    for i in range(1, len(h1)):
        price = h1["Close"].iloc[i]
        ts    = h1.index[i]
        date  = ts.date()

        # Get yesterday's levels -- match by date
        d1_date = pd.Timestamp(date, tz=d1.index.tz)
        # find the daily bar for today's date (which gives yesterday's OHLC via shift)
        candidates = [k for k in d1_dict if k.date() == date]
        if not candidates:
            continue
        prev = d1_dict[candidates[0]]
        ph, pl, pm, pr = prev["prev_high"], prev["prev_low"], prev["prev_mid"], prev["prev_rng"]
        if np.isnan(ph) or pr <= 0:
            continue

        # Manage open position
        if position != 0:
            sl_hit = (position==1 and price<=sl) or (position==-1 and price>=sl)
            tp_hit = (position==1 and price>=tp) or (position==-1 and price<=tp)
            if sl_hit or tp_hit:
                ep = sl if sl_hit else tp
                trades.append(position*(ep-entry_price)/abs(entry_price-sl)*equity*RISK_PCT)
                equity += trades[-1]; position = 0

        stop_dist = abs(price - pm) if position == 0 else 0

        # BUY breakout -- only once per day per direction
        if price > ph and position == 0 and day_traded_long != date:
            sd = abs(price - pm)
            if sd <= 0: continue
            units = min(equity*RISK_PCT/sd, equity*MAX_LEV/price)
            if units <= 0: continue
            position=1; entry_price=price; sl=pm; tp=price+TP_MULT*pr
            day_traded_long = date

        # SELL breakout
        elif price < pl and position == 0 and day_traded_short != date:
            sd = abs(price - pm)
            if sd <= 0: continue
            units = min(equity*RISK_PCT/sd, equity*MAX_LEV/price)
            if units <= 0: continue
            position=-1; entry_price=price; sl=pm; tp=price-TP_MULT*pr
            day_traded_short = date

    if len(trades) < 10:
        return None
    t   = pd.Series(trades)
    yrs = int(period[0])
    sharpe = t.mean()/t.std()*np.sqrt(252) if t.std()>0 else 0
    dd     = (t.cumsum()+INITIAL).pipe(lambda s: ((s-s.cummax())/s.cummax()).min())
    ann    = (equity/INITIAL)**(1/yrs)-1
    return {"n": len(t), "wr": (t>0).mean(), "sharpe": sharpe, "dd": dd, "ann": ann}


print("Previous Day High/Low Breakout — H1 entry\n")
print(f"{'Pair':<14} {'5y Sh':>7} {'5y N':>6} {'10y Sh':>7} {'10y N':>6} {'WR':>5} {'DD':>7}  Verdict")
print("-" * 68)

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

    if np.isnan(sh5) and np.isnan(sh10):
        print(f"{pair:<14}  -- no data"); continue

    verdict = "PASS" if (not np.isnan(sh5) and sh5>0) and (not np.isnan(sh10) and sh10>0) else \
              "FAIL" if (not np.isnan(sh5) and sh5<0) or (not np.isnan(sh10) and sh10<0) else "MARGINAL"
    if verdict == "PASS":
        good.append(pair)

    s5 = f"{sh5:>7.2f}" if not np.isnan(sh5) else "     --"
    s10= f"{sh10:>7.2f}" if not np.isnan(sh10) else "     --"
    w  = f"{wr:>5.0%}" if not np.isnan(wr) else "   --"
    d  = f"{dd:>7.1%}" if not np.isnan(dd) else "     --"
    print(f"{pair:<14} {s5} {n5:>6} {s10} {n10:>6} {w} {d}  {verdict}")

print(f"\nPassing ({len(good)}): {', '.join(good)}")
