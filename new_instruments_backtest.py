"""
Backtest untested OANDA instruments — stock indices, bonds, and anything
not already in the live sleeves. Tests EMA H1 (our primary live strategy).

Already trading (skipped here):
  FX: GBP/USD, EUR/USD, EUR/JPY, CHF/JPY, GBP/JPY, AUD/JPY, NZD/JPY, USD/JPY + M30 BBMRT pairs
  CFD: NAS100, SPX500, US30, DE30, JP225, XAU, XAG, BCO, WTICO, NATGAS, CORN, WHEAT
"""
from dotenv import load_dotenv
load_dotenv()

from backtesting import Backtest
from data import load_oanda_data
from strategies import EmaCrossoverAtr

CANDIDATES = [
    # --- Stock indices not yet trading ---
    ("UK100     FTSE 100",      "UK100_GBP",  "INDEX"),
    ("EU50      EuroStoxx 50",  "EU50_EUR",   "INDEX"),
    ("FR40      CAC 40",        "FR40_EUR",   "INDEX"),
    ("AU200     ASX 200",       "AU200_AUD",  "INDEX"),
    ("US2000    Russell 2000",  "US2000_USD", "INDEX"),
    ("HK33      Hang Seng",     "HK33_HKD",  "INDEX"),
    ("CH20      SMI",           "CH20_CHF",   "INDEX"),
    ("ESPIX     IBEX 35",       "ESPIX_EUR",  "INDEX"),
    ("CN50      China A50",     "CN50_USD",   "INDEX"),
    # --- Bonds ---
    ("USB10Y    US 10Y T-Note", "USB10Y_USD", "BOND"),
    ("USB30Y    US T-Bond",     "USB30Y_USD", "BOND"),
    ("USB05Y    US 5Y T-Note",  "USB05Y_USD", "BOND"),
    ("DE10YB    Bund",          "DE10YB_EUR", "BOND"),
    ("UK10YB    Gilt",          "UK10YB_GBP", "BOND"),
]

RISK_PCT = 0.0025
PERIOD   = "5y"

print(f"\nNew Instrument Candidates — EMA H1 Backtest (5y, 0.25% risk/sleeve)")
print(f"\n{'Instrument':<30} {'Cat':<7} {'Sharpe':>7} {'Ann Ret%':>9} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7}  Verdict")
print("-" * 100)

by_cat = {}

for label, instr, cat in CANDIDATES:
    try:
        df = load_oanda_data(instr, period=PERIOD, interval="1h")
        bt = Backtest(df, EmaCrossoverAtr, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
        s  = bt.run(risk_pct=RISK_PCT)

        sharpe   = float(s.get("Sharpe Ratio", 0) or 0)
        ann_ret  = float(s.get("Return (Ann.) [%]", 0) or 0)
        max_dd   = float(s.get("Max. Drawdown [%]", 0) or 0)
        win_rate = float(s.get("Win Rate [%]", 0) or 0)
        n_trades = int(s.get("# Trades", 0) or 0)

        if sharpe >= 1.0 and ann_ret > 0:
            verdict = "✅ ADD"
        elif sharpe >= 0.5 and ann_ret > 0:
            verdict = "⚠️  WATCH"
        else:
            verdict = "❌ SKIP"

        by_cat.setdefault(cat, []).append((label, sharpe, ann_ret, max_dd, win_rate, n_trades, verdict))
        print(f"{label:<30} {cat:<7} {sharpe:>7.2f} {ann_ret:>8.1f}% {max_dd:>7.1f}% {win_rate:>8.1f}% {n_trades:>7}  {verdict}")

    except Exception as e:
        print(f"{label:<30} {cat:<7}  ERROR: {e}")

print("\n")
print("=== SUMMARY — instruments worth adding ===")
for cat, rows in by_cat.items():
    adds = [(l, sh, ar, dd) for l, sh, ar, dd, wr, n, v in rows if "ADD" in v or "WATCH" in v]
    if adds:
        print(f"\n{cat}:")
        for l, sh, ar, dd in adds:
            print(f"  {l}  Sharpe={sh:.2f}  AnnRet={ar:+.1f}%  MaxDD={dd:.1f}%")
    else:
        print(f"\n{cat}: nothing clears the bar")
