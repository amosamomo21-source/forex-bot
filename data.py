import pandas as pd
import yfinance as yf

_OANDA_GRANULARITY = {
    "1d": "D", "1wk": "W", "1h": "H1", "4h": "H4", "30m": "M30", "15m": "M15", "5m": "M5", "1m": "M1",
}

_BARS_PER_DAY = {
    "D": 1, "W": 1 / 7, "H4": 6, "H1": 24, "M30": 48, "M15": 96, "M5": 288, "M1": 1440,
}


def load_fx_data(ticker: str = "EURUSD=X", period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download historical FX candles from Yahoo Finance.

    yfinance's EURUSD=X-style tickers are spot rate proxies, not a real
    broker feed -- fine for prototyping a strategy, not for production
    backtests. Prefer load_oanda_data for anything meant to inform a real
    trading decision.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({period}, {interval})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def _to_oanda_instrument(ticker: str) -> str:
    """Converts a yfinance-style FX ticker ("EURUSD=X" or "EURUSD") to OANDA's
    instrument code ("EUR_USD"), so callers can pass the same ticker regardless
    of data source."""
    code = ticker.upper().removesuffix("=X")
    if "_" in code:
        return code
    if len(code) != 6:
        raise ValueError(f"Can't convert ticker '{ticker}' to an OANDA instrument code")
    return f"{code[:3]}_{code[3:]}"


def _period_to_count(period: str, granularity: str = "D") -> int:
    """Converts a yfinance-style period ("5y", "18mo", "90d") to a number of
    candles, using ~252 trading days/year and ~21/month for "y"/"mo" units so
    the requested calendar span actually covers that much trading history.
    Scaled by granularity's bars/day for intraday intervals -- "90d" at "M5"
    means 90 trading days' worth of 5-minute bars, not 90 candles."""
    period = period.strip().lower()
    if period.endswith("mo"):
        days = float(period[:-2]) * 21
    elif period.endswith("y"):
        days = float(period[:-1]) * 252
    elif period.endswith("d"):
        days = float(period[:-1])
    else:
        raise ValueError(f"Unsupported period '{period}'")
    return round(days * _BARS_PER_DAY.get(granularity, 1))


def _row_from_candle(c: dict) -> dict:
    mid = c["mid"]
    return {
        "time": c["time"],
        "Open": float(mid["o"]),
        "High": float(mid["h"]),
        "Low": float(mid["l"]),
        "Close": float(mid["c"]),
        "Volume": c["volume"],
    }


def load_oanda_data(ticker: str = "EURUSD=X", period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Historical FX/CFD candles from OANDA's own feed -- the real broker this
    project trades through -- via the same practice-account credentials
    broker.py uses, not a third-party proxy like yfinance.

    OANDA's InstrumentsCandles caps a single request at 5000 candles, counted
    backward from a given `to` time (or now, if omitted). At "D" granularity
    that's ~19 years, comfortably covering this project's daily backtests in
    one request. Intraday granularities (e.g. "5m") need far more candles to
    cover the same calendar span, so this paginates backward in 5000-candle
    batches via repeated `to` cursors until the requested period is covered or
    OANDA's history runs out.
    """
    import broker

    if interval not in _OANDA_GRANULARITY:
        raise ValueError(f"Unsupported interval '{interval}' for OANDA. Known: {list(_OANDA_GRANULARITY)}")
    instrument = _to_oanda_instrument(ticker)
    granularity = _OANDA_GRANULARITY[interval]
    remaining = _period_to_count(period, granularity)

    b = broker.from_env()
    rows = []
    to = None
    while remaining > 0:
        batch_count = min(remaining, 5000)
        candles = b.get_candles(instrument, granularity=granularity, count=batch_count, to=to)
        if not candles:
            break
        rows = [_row_from_candle(c) for c in candles if c["complete"]] + rows
        to = candles[0]["time"]
        remaining -= len(candles)
        if len(candles) < batch_count:
            break  # ran out of available history before covering the requested period

    if not rows:
        raise RuntimeError(f"No OANDA candles returned for {instrument} ({granularity})")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset="time").set_index("time").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_oanda_intraday_with_htf_trend(
    ticker: str = "XAUUSD=X",
    period: str = "180d",
    interval: str = "5m",
    htf_interval: str = "1h",
    htf_ema_period: int = 50,
) -> pd.DataFrame:
    """Like load_oanda_data, but adds an "HtfEmaPrev" column: the EMA(htf_ema_period)
    of htf_interval closes, as of the last FULLY COMPLETED htf_interval bar
    before each row's own timestamp.

    Exists because the `backtesting` library backtests a single timeframe at a
    time; this precomputes the higher-timeframe trend value per-row so a
    Strategy can read it like any other indicator column, without needing
    multi-timeframe support inside the backtest loop itself.

    Note on what "EMA as of the last completed HTF bar" means live: a strategy
    that reads a higher-timeframe EMA mid-bar is really reading
    alpha*latest_price + (1-alpha)*EMA_prev_completed (the EMA recursion, with
    the still-forming bar's "close" approximated by the latest tick) -- which
    algebraically reduces to comparing the latest price against
    EMA_prev_completed directly. So a strategy can just compare its own
    intraday close to this column rather than reimplementing that recursion.
    """
    df = load_oanda_data(ticker, period, interval)
    htf_df = load_oanda_data(ticker, period, htf_interval)

    htf_ema = htf_df["Close"].ewm(span=htf_ema_period, adjust=False).mean()
    floor_str, offset = {
        "1h": ("h", pd.Timedelta(hours=1)),
        "4h": ("4h", pd.Timedelta(hours=4)),
        "1d": ("D", pd.Timedelta(days=1)),
    }.get(htf_interval, ("h", pd.Timedelta(hours=1)))
    prev_htf_start = df.index.floor(floor_str) - offset
    df["HtfEmaPrev"] = htf_ema.reindex(prev_htf_start).values
    return df
