from dotenv import load_dotenv
load_dotenv()

from backtesting import Backtest
from data import load_oanda_data
from strategies import BollingerMeanReversionTrendFilter, EmaCrossoverAtr

STATS = ["Return (Ann.) [%]", "Sharpe Ratio", "Max. Drawdown [%]", "Win Rate [%]", "Profit Factor"]

tests = [
    ("GBP/USD PDHL proxy (EMA H1)",  "GBP_USD",    "1h",  EmaCrossoverAtr),
    ("EUR/JPY EMA H1",               "EUR_JPY",    "1h",  EmaCrossoverAtr),
    ("GBP/JPY EMA H1",               "GBP_JPY",    "1h",  EmaCrossoverAtr),
    ("USD/JPY EMA H1",               "USD_JPY",    "1h",  EmaCrossoverAtr),
    ("XAU/USD EMA H1",               "XAU_USD",    "1h",  EmaCrossoverAtr),
    ("EUR/USD M30 BBMRT",            "EUR_USD",    "30m", BollingerMeanReversionTrendFilter),
    ("GBP/USD M30 BBMRT",            "GBP_USD",    "30m", BollingerMeanReversionTrendFilter),
]

print(f"\n{'Pair':<28} {'Metric':<25} {'1x (0.01%)':<14} {'2x (0.02%)':<14} {'Change'}")
print("-" * 90)

for label, instr, tf, cls in tests:
    df = load_oanda_data(instr, period="5y", interval=tf)
    results = {}
    for mult, rpct in [(1, 0.01), (2, 0.02)]:
        bt = Backtest(df, cls, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
        stats = bt.run(risk_pct=rpct)
        results[mult] = stats

    for stat in STATS:
        v1 = results[1].get(stat, float("nan"))
        v2 = results[2].get(stat, float("nan"))
        if isinstance(v1, float) and isinstance(v2, float):
            chg = f"{v2-v1:+.2f}"
        else:
            chg = "n/a"
        print(f"{label:<28} {stat:<25} {v1:<14.2f} {v2:<14.2f} {chg}")
    print()
