import argparse
import json

from backtesting import Backtest
from dotenv import load_dotenv

load_dotenv()

from data import load_fx_data, load_oanda_data, load_oanda_intraday_with_htf_trend  # noqa: E402
from strategies import (
    BollingerMeanReversion,
    BollingerMeanReversionTrendFilter,
    DonchianBreakout,
    EmaCrossoverAtr,
    GoldScalperPro,
)

STRATEGIES = {
    "ema_crossover": EmaCrossoverAtr,
    "bb_meanrev": BollingerMeanReversion,
    "bb_meanrev_trend": BollingerMeanReversionTrendFilter,
    "donchian_breakout": DonchianBreakout,
    "gold_scalper_pro": GoldScalperPro,
}

DATA_SOURCES = {
    "oanda": load_oanda_data,
    "yfinance": load_fx_data,
    "oanda_htf": load_oanda_intraday_with_htf_trend,
}

SCALAR_STATS = [
    "Return [%]",
    "Buy & Hold Return [%]",
    "Return (Ann.) [%]",
    "Volatility (Ann.) [%]",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Calmar Ratio",
    "Max. Drawdown [%]",
    "# Trades",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg. Trade [%]",
    "Profit Factor",
    "Expectancy [%]",
    "SQN",
    "Kelly Criterion",
]


def run_backtest(
    strategy_name: str = "ema_crossover",
    ticker: str = "EURUSD=X",
    period: str = "5y",
    interval: str = "1d",
    source: str = "oanda",
    cash: float = 10_000,
    commission: float = 0.0002,
    margin: float = 1 / 30,
    **strategy_params,
):
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Known: {list(STRATEGIES)}")
    if source not in DATA_SOURCES:
        raise ValueError(f"Unknown source '{source}'. Known: {list(DATA_SOURCES)}")
    strategy_cls = STRATEGIES[strategy_name]
    data = DATA_SOURCES[source](ticker, period, interval)
    bt = Backtest(
        data, strategy_cls, cash=cash, commission=commission, margin=margin, finalize_trades=True
    )
    stats = bt.run(**strategy_params)
    return stats, bt


def stats_to_dict(stats) -> dict:
    out = {}
    for key in SCALAR_STATS:
        if key in stats:
            value = stats[key]
            out[key] = float(value) if hasattr(value, "__float__") else value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", nargs="?", default="ema_crossover")
    parser.add_argument("--ticker", default="EURUSD=X")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--source", default="oanda", choices=list(DATA_SOURCES))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    stats, _ = run_backtest(
        strategy_name=args.strategy,
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        source=args.source,
    )

    if args.json:
        print(json.dumps(stats_to_dict(stats), indent=2))
    else:
        print(stats)


if __name__ == "__main__":
    main()
