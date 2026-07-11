from dotenv import load_dotenv
load_dotenv()

from backtesting import Backtest
from data import load_oanda_data
from strategies import BollingerMeanReversion

SCALAR_STATS = [
    "Return [%]", "Buy & Hold Return [%]", "Return (Ann.) [%]",
    "Volatility (Ann.) [%]", "Sharpe Ratio", "Sortino Ratio",
    "Max. Drawdown [%]", "# Trades", "Win Rate [%]",
    "Best Trade [%]", "Worst Trade [%]", "Avg. Trade [%]",
    "Profit Factor", "Expectancy [%]", "SQN",
]

for label, period in [("5y", "5y"), ("10y", "10y")]:
    df = load_oanda_data("NAS100_USD", period=period, interval="1d")
    bt = Backtest(df, BollingerMeanReversion, cash=10_000, commission=0.0002, margin=1/30, finalize_trades=True)
    stats = bt.run()
    print(f"\n=== NAS100 Daily BB Mean Reversion — {label} ===")
    for k in SCALAR_STATS:
        if k in stats:
            print(f"  {k}: {stats[k]}")
