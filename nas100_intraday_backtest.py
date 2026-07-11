from dotenv import load_dotenv
load_dotenv()

from backtesting import Backtest
from data import load_oanda_data
from strategies import BollingerMeanReversion, BollingerMeanReversionTrendFilter, EmaCrossoverAtr

SCALAR_STATS = [
    "Return (Ann.) [%]", "Sharpe Ratio", "Max. Drawdown [%]",
    "# Trades", "Win Rate [%]", "Profit Factor", "SQN",
]

strategies = [
    ("BB MeanRev",        BollingerMeanReversion,           {}),
    ("BB MeanRev+Trend",  BollingerMeanReversionTrendFilter, {}),
    ("EMA Crossover",     EmaCrossoverAtr,                   {}),
]

for interval in ["1d"]:
    for period in ["5y", "10y"]:
        df = load_oanda_data("NAS100_USD", period=period, interval=interval)
        print(f"\n{'='*60}")
        print(f"NAS100  D1  {period}")
        print(f"{'='*60}")
        for name, cls, params in strategies:
            try:
                bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                              margin=1/30, finalize_trades=True)
                stats = bt.run(**params)
                vals = {k: stats[k] for k in SCALAR_STATS if k in stats}
                print(f"\n  [{name}]")
                for k, v in vals.items():
                    print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
            except Exception as e:
                print(f"\n  [{name}] ERROR: {e}")
