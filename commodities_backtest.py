"""H1 EMA(10/30) backtest across all OANDA commodities.
Skips pairs already in live (WTICO_USD, BCO_USD).
"""
from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import ema, atr

COMMODITIES = {
    "XAU_USD":   "Gold",
    "XAG_USD":   "Silver",
    "XCU_USD":   "Copper",
    "XPT_USD":   "Platinum",
    "XPD_USD":   "Palladium",
    "NATGAS_USD":"Natural Gas",
    "CORN_USD":  "Corn",
    "SOYBN_USD": "Soybeans",
    "WHEAT_USD": "Wheat",
    "SUGAR_USD": "Sugar",
}

RISK_PCT = 0.01
MAX_LEV  = 5.0
INITIAL  = 10_000


def backtest(ticker, period):
    try:
        df = load_oanda_data(ticker, period=period, interval="1h")
    except Exception as e:
        return None, str(e)
    if df is None or len(df) < 35:
        return None, "insufficient data"

    fast_s = ema(df["Close"], 10)
    slow_s = ema(df["Close"], 30)
    atr_s  = atr(df["High"], df["Low"], df["Close"], 14)

    equity = INITIAL; position = 0; entry_price = sl = tp = 0.0; trades = []
    for i in range(1, len(df)):
        price = df["Close"].iloc[i]; av = atr_s.iloc[i]
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

    if len(trades) < 10:
        return None, f"too few trades ({len(trades)})"
    t   = pd.Series(trades)
    yrs = int(period[0])
    sharpe = t.mean()/t.std()*np.sqrt(252) if t.std()>0 else 0
    dd     = (t.cumsum()+INITIAL).pipe(lambda s: ((s-s.cummax())/s.cummax()).min())
    ann    = (equity/INITIAL)**(1/yrs)-1
    return {"n": len(t), "wr": (t>0).mean(), "sharpe": sharpe, "dd": dd, "ann": ann}, None


print("H1 EMA(10/30) — All OANDA Commodities\n")
print(f"{'Instrument':<14} {'Name':<13} {'5y Sh':>7} {'5y N':>6} {'10y Sh':>7} {'10y N':>6} {'WR':>5} {'DD':>7}  Verdict")
print("-" * 80)

good = []
for ticker, name in COMMODITIES.items():
    r5,  e5  = backtest(ticker, "5y")
    r10, e10 = backtest(ticker, "10y")

    sh5  = r5["sharpe"]  if r5  else float("nan")
    sh10 = r10["sharpe"] if r10 else float("nan")
    n5   = r5["n"]       if r5  else 0
    n10  = r10["n"]      if r10 else 0
    wr   = r10["wr"]     if r10 else float("nan")
    dd   = r10["dd"]     if r10 else float("nan")

    if not r5 or not r10:
        note = e5 or e10 or "no data"
        print(f"{ticker:<14} {name:<13}  -- skipped: {note}")
        continue

    verdict = "PASS" if sh5 > 0 and sh10 > 0 else \
              "FAIL" if sh5 < 0 or sh10 < 0 else "MARGINAL"
    if verdict == "PASS":
        good.append((ticker, name, sh5, sh10))

    s5 = f"{sh5:>7.2f}"; s10 = f"{sh10:>7.2f}"
    w  = f"{wr:>5.0%}";  d   = f"{dd:>7.1%}"
    print(f"{ticker:<14} {name:<13} {s5} {n5:>6} {s10} {n10:>6} {w} {d}  {verdict}")

print(f"\nPassing ({len(good)}): {', '.join(t for t,_,_,_ in good)}")
