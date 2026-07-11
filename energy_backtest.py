"""
Commodity expansion backtest — EMA H1 crossover (our live strategy) on all
OANDA CFD commodities not yet fully covered.

Groups:
  BASELINE  — already in both EMA H1 + PDHL (benchmark)
  EMA_ONLY  — in EMA H1 but not PDHL (test whether PDHL adds value)
  NEW       — not trading at all yet

Reports Sharpe, annual return, max DD, win rate over 5y.
"""
from dotenv import load_dotenv
load_dotenv()

from backtesting import Backtest
from data import load_oanda_data
from strategies import EmaCrossoverAtr

INSTRUMENTS = [
    # --- BASELINE (already in EMA + PDHL) ---
    ("BCO/USD  Brent Crude",   "BCO_USD",   "BASELINE"),
    ("NATGAS   Natural Gas",   "NATGAS_USD","BASELINE"),
    # --- EMA ONLY (not yet in PDHL) ---
    ("WTICO/USD WTI Crude",    "WTICO_USD", "EMA_ONLY"),
    ("CORN/USD  Corn",         "CORN_USD",  "EMA_ONLY"),
    ("WHEAT/USD Wheat",        "WHEAT_USD", "EMA_ONLY"),
    # --- NEW (not trading at all) ---
    ("SOYBN/USD Soybeans",     "SOYBN_USD", "NEW"),
    ("SUGAR/USD Sugar",        "SUGAR_USD", "NEW"),
    ("XCU/USD   Copper",       "XCU_USD",   "NEW"),
    ("XPD/USD   Palladium",    "XPD_USD",   "NEW"),
    ("XPT/USD   Platinum",     "XPT_USD",   "NEW"),
]

PERIOD   = "5y"
RISK_PCT = 0.0025   # 0.25% per sleeve — same as live RISK_MODE="demo"

print(f"\nCommodity EMA H1 Backtest — 5y, 0.25% risk/sleeve")
print(f"\n{'Instrument':<28} {'Group':<10} {'Sharpe':>7} {'Ann Ret%':>9} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7}  Verdict")
print("-" * 95)

results = []

for label, instr, group in INSTRUMENTS:
    try:
        df = load_oanda_data(instr, period=PERIOD, interval="1h")
        bt = Backtest(df, EmaCrossoverAtr, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
        s = bt.run(risk_pct=RISK_PCT)

        sharpe   = float(s.get("Sharpe Ratio", 0) or 0)
        ann_ret  = float(s.get("Return (Ann.) [%]", 0) or 0)
        max_dd   = float(s.get("Max. Drawdown [%]", 0) or 0)
        win_rate = float(s.get("Win Rate [%]", 0) or 0)
        n_trades = int(s.get("# Trades", 0) or 0)

        if sharpe >= 1.0 and ann_ret > 0:
            verdict = "✅ ADD"
        elif sharpe >= 0.5 and ann_ret > 0:
            verdict = "⚠️  BORDERLINE"
        else:
            verdict = "❌ SKIP"

        results.append((label, group, sharpe, ann_ret, max_dd, win_rate, n_trades, verdict))
        print(f"{label:<28} {group:<10} {sharpe:>7.2f} {ann_ret:>8.1f}% {max_dd:>7.1f}% {win_rate:>8.1f}% {n_trades:>7}  {verdict}")

    except Exception as e:
        print(f"{label:<28} {group:<10}  ERROR: {e}")

print("\n")
print("Summary — candidates to add:")
for label, group, sharpe, ann_ret, max_dd, win_rate, n_trades, verdict in results:
    if "ADD" in verdict or "BORDER" in verdict:
        print(f"  {verdict}  {label}  (Sharpe={sharpe:.2f}, AnnRet={ann_ret:+.1f}%, MaxDD={max_dd:.1f}%)")
