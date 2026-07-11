"""
Find the right risk% per sleeve for an FTMO $100k challenge.

FTMO rules:
  - Profit target: +10% ($10k) in 30 days
  - Max daily loss:  5% ($5k)
  - Max total drawdown: 10% ($10k)

Tests the validated pairs at 0.01%, 0.05%, 0.10%, 0.25% risk per sleeve,
then projects portfolio-level monthly return and worst-case drawdown.
"""
from dotenv import load_dotenv
load_dotenv()

from backtesting import Backtest
from data import load_oanda_data
from strategies import BollingerMeanReversionTrendFilter, EmaCrossoverAtr

# Validated pairs (Sharpe > 1.0 on both 5y and 10y from prior backtests)
VALIDATED = [
    # (label, instrument, timeframe, strategy)
    ("M30 BBMRT EUR/USD",  "EUR_USD",    "30m", BollingerMeanReversionTrendFilter),
    ("M30 BBMRT GBP/USD",  "GBP_USD",    "30m", BollingerMeanReversionTrendFilter),
    ("M30 BBMRT EUR/JPY",  "EUR_JPY",    "30m", BollingerMeanReversionTrendFilter),
    ("M30 BBMRT CHF/JPY",  "CHF_JPY",    "30m", BollingerMeanReversionTrendFilter),
    ("M30 BBMRT GBP/JPY",  "GBP_JPY",    "30m", BollingerMeanReversionTrendFilter),
    ("H1 EMA  GBP/USD",    "GBP_USD",    "1h",  EmaCrossoverAtr),
    ("H1 EMA  AUD/JPY",    "AUD_JPY",    "1h",  EmaCrossoverAtr),
    ("H1 EMA  XAU/USD",    "XAU_USD",    "1h",  EmaCrossoverAtr),
    ("H1 EMA  NATGAS",     "NATGAS_USD", "1h",  EmaCrossoverAtr),
]

ACCOUNT   = 100_000
RISK_LEVELS = [0.0001, 0.0005, 0.001, 0.0025]  # 0.01%, 0.05%, 0.10%, 0.25%
PERIOD    = "5y"

print(f"\nFTMO $100k Challenge — Risk Calibration")
print(f"Profit target: +$10,000 in 30 days | Max DD: -$10,000\n")

# Collect per-sleeve stats at each risk level
all_results = {r: [] for r in RISK_LEVELS}

for label, instr, tf, cls in VALIDATED:
    df = load_oanda_data(instr, period=PERIOD, interval=tf)
    for rpct in RISK_LEVELS:
        bt = Backtest(df, cls, cash=ACCOUNT, commission=0.0002, margin=1/30, finalize_trades=True)
        stats = bt.run(risk_pct=rpct)
        all_results[rpct].append({
            "label": label,
            "ann_return_pct": float(stats.get("Return (Ann.) [%]", 0)),
            "max_dd_pct":     float(stats.get("Max. Drawdown [%]", 0)),
            "n_trades":       int(stats.get("# Trades", 0)),
            "sharpe":         float(stats.get("Sharpe Ratio", 0)),
        })

# Portfolio projection: sum annualised returns, worst single-sleeve drawdown
print(f"{'Risk/sleeve':<14} {'Sleeves':<9} {'Avg ann return':<18} {'Portfolio ann $':<18} {'Monthly est $':<16} {'Worst DD $':<14} {'FTMO safe?'}")
print("-" * 110)

FTMO_TARGET_MONTHLY = ACCOUNT * 0.10
FTMO_MAX_DD         = ACCOUNT * 0.10

for rpct in RISK_LEVELS:
    rows = all_results[rpct]
    n = len(rows)
    avg_ann = sum(r["ann_return_pct"] for r in rows) / n
    # Portfolio return: each sleeve trades independently on its own equity share
    # Conservative estimate: sum of individual sleeve returns scaled to full account
    portfolio_ann_pct = avg_ann  # each sleeve already runs on full ACCOUNT cash
    portfolio_ann_dollar = ACCOUNT * portfolio_ann_pct / 100
    monthly_dollar = portfolio_ann_dollar / 12
    worst_dd_dollar = min(r["max_dd_pct"] for r in rows) * ACCOUNT / 100

    hits_target = monthly_dollar >= FTMO_TARGET_MONTHLY
    within_dd   = abs(worst_dd_dollar) <= FTMO_MAX_DD
    safe = "✅" if hits_target and within_dd else ("⚠️ DD" if not within_dd else "❌ low")

    print(f"{rpct*100:.2f}%         {n:<9} {avg_ann:>+8.1f}%          "
          f"${portfolio_ann_dollar:>+10,.0f}       ${monthly_dollar:>+8,.0f}        "
          f"${worst_dd_dollar:>+8,.0f}      {safe}")

print()
print("Notes:")
print("  - Per-sleeve return is % of $100k (not 1/N share) so numbers are conservative")
print("  - Worst DD = worst single sleeve; real portfolio DD diversified lower")
print("  - FTMO requires +$10k/month and drawdown never exceeds -$10k total")
