"""
Energy/commodity EMA crossover across M15, M30, H1, D1 — side-by-side comparison.
"""
from dotenv import load_dotenv
load_dotenv()

from backtesting import Backtest
from data import load_oanda_data
from strategies import EmaCrossoverAtr

INSTRUMENTS = [
    ("BCO/USD  Brent",    "BCO_USD"),
    ("WTICO/USD WTI",     "WTICO_USD"),
    ("NATGAS   NatGas",   "NATGAS_USD"),
    ("CORN/USD  Corn",    "CORN_USD"),
    ("WHEAT/USD Wheat",   "WHEAT_USD"),
    ("SOYBN/USD Soybean", "SOYBN_USD"),
    ("SUGAR/USD Sugar",   "SUGAR_USD"),
    ("XCU/USD   Copper",  "XCU_USD"),
    ("XPT/USD   Platinum","XPT_USD"),
    ("XPD/USD   Palladium","XPD_USD"),
]

TIMEFRAMES = [("15m", "M15"), ("30m", "M30"), ("1h", "H1"), ("1d", "D1")]
RISK_PCT   = 0.0025
PERIOD     = "5y"

print(f"\nEnergy/Commodity — EMA Crossover across timeframes (5y, 0.25% risk)")
print(f"\n{'Instrument':<26}", end="")
for _, tf_label in TIMEFRAMES:
    print(f"  {tf_label:>16}", end="")
print()
print(f"{'':26}", end="")
for _ in TIMEFRAMES:
    print(f"  {'Sharpe / AnnRet':>16}", end="")
print()
print("-" * 100)

for label, instr in INSTRUMENTS:
    print(f"{label:<26}", end="", flush=True)
    for interval, tf_label in TIMEFRAMES:
        try:
            df = load_oanda_data(instr, period=PERIOD, interval=interval)
            bt = Backtest(df, EmaCrossoverAtr, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
            s  = bt.run(risk_pct=RISK_PCT)
            sharpe  = float(s.get("Sharpe Ratio", 0) or 0)
            ann_ret = float(s.get("Return (Ann.) [%]", 0) or 0)
            flag = "✅" if sharpe >= 1.0 and ann_ret > 0 else ("⚠" if sharpe >= 0.5 and ann_ret > 0 else "❌")
            print(f"  {flag} {sharpe:+.2f}/{ann_ret:+5.1f}%", end="", flush=True)
        except Exception as e:
            print(f"  {'ERR':>16}", end="")
    print()

print()
