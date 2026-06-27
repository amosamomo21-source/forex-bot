"""Daily forward (paper) trading runner for the validated multi-sleeve
portfolio: BollingerMeanReversionTrendFilter on EUR_USD/GBP_USD, and
EmaCrossoverAtr on GBP_USD/USD_JPY/AUD_USD.

Run once per day, after OANDA's daily candle rolls over (~5pm New York time),
against the OANDA practice account only -- see schedule via cron, e.g.:
    15 22 * * 1-5  cd /Users/bamznizzy/forex-bot && uv run python3 live_runner.py >> live_runner.log 2>&1

Re-implements each strategy's exact entry/exit rules from strategies.py,
applied to one new live bar at a time -- this is what actually produces a
genuine forward (out-of-sample) test: none of this data existed when the
strategies were designed or backtested. Trade history shows up automatically
in dashboard.py's OANDA tab.

GBP_USD is the only instrument two sleeves both trade. This OANDA account
has hedgingEnabled=False (confirmed via account_summary()), so it can't hold
two genuinely opposite-direction trades on the same instrument at once --
OANDA's default REDUCE_FIRST behavior would net a new opposite order against
the other sleeve's existing trade, silently corrupting both sleeves'
bookkeeping. _opposite_direction_conflict() guards against this by skipping
(and logging) a new entry whenever another sleeve already holds the
opposite side on the same instrument. Same-direction overlap is fine --
OANDA tracks each trade by ID regardless of hedging mode -- so this only
ever blocks the one broker-unsafe case.

Every order is tagged (tradeClientExtensions) with its sleeve name so each
sleeve's trades stay individually identifiable and closable via
close_trade(), even when sharing an instrument with another sleeve.

Each sleeve risks risk_pct of a 1/N share of total account equity (N =
number of sleeves), matching the equal-weighted blend this portfolio was
validated with in backtest -- not risk_pct of the full account balance.

Always uses broker.from_env()'s default practice environment -- never pass
allow_live=True here.
"""

from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import broker  # noqa: E402
import journal  # noqa: E402
from strategies import BollingerMeanReversionTrendFilter, EmaCrossoverAtr  # noqa: E402
from strategies import atr, ema, macd, rsi, rolling_std, sma  # noqa: E402

SLEEVES = [
    ("bbmrt_eurusd", "EUR_USD", "bbmrt"),
    ("bbmrt_gbpusd", "GBP_USD", "bbmrt"),
    ("ema_gbpusd", "GBP_USD", "ema"),
    ("ema_usdjpy", "USD_JPY", "ema"),
    ("ema_audusd", "AUD_USD", "ema"),
]
M30_SLEEVES = [
    # Core validated pairs (10y Sharpe > 2.0, positive both periods)
    ("bbmrt_m30_eurusd", "EUR_USD"),
    ("bbmrt_m30_gbpusd", "GBP_USD"),
    ("bbmrt_m30_eurcad", "EUR_CAD"),
    ("bbmrt_m30_eurjpy", "EUR_JPY"),
    ("bbmrt_m30_chfjpy", "CHF_JPY"),
    ("bbmrt_m30_audchf", "AUD_CHF"),
    ("bbmrt_m30_eursgd", "EUR_SGD"),
    ("bbmrt_m30_gbpaud", "GBP_AUD"),
    # Additional positive pairs (10y Sharpe 1.0-2.0)
    ("bbmrt_m30_cadjpy", "CAD_JPY"),
    ("bbmrt_m30_audsgd", "AUD_SGD"),
    ("bbmrt_m30_euraud", "EUR_AUD"),
    ("bbmrt_m30_gbpcad", "GBP_CAD"),
    ("bbmrt_m30_gbpsgd", "GBP_SGD"),
    ("bbmrt_m30_gbpjpy", "GBP_JPY"),
    # Marginal pairs (10y Sharpe 0.2-1.0, included for frequency)
    ("bbmrt_m30_gbpchf", "GBP_CHF"),
    ("bbmrt_m30_audjpy", "AUD_JPY"),
    ("bbmrt_m30_nzdjpy", "NZD_JPY"),
]
H1_SLEEVES = [
    ("ema_h1_gbpusd",  "GBP_USD"),
    ("ema_h1_eurjpy",  "EUR_JPY"),
    ("ema_h1_chfjpy",  "CHF_JPY"),
    ("ema_h1_cadjpy",  "CAD_JPY"),
    ("ema_h1_audjpy",  "AUD_JPY"),
    ("ema_h1_gbpjpy",  "GBP_JPY"),
    ("ema_h1_nzdjpy",  "NZD_JPY"),
    ("ema_h1_audchf",  "AUD_CHF"),
    ("ema_h1_euraud",  "EUR_AUD"),
    ("ema_h1_audsgd",  "AUD_SGD"),
    ("ema_h1_wticousd",  "WTICO_USD"),
    ("ema_h1_bcousd",    "BCO_USD"),
    ("ema_h1_xauusd",    "XAU_USD"),
    ("ema_h1_xagusd",    "XAG_USD"),
    ("ema_h1_xcuusd",    "XCU_USD"),
    ("ema_h1_xptusd",    "XPT_USD"),
    ("ema_h1_natgasusd", "NATGAS_USD"),
    ("ema_h1_cornusd",   "CORN_USD"),
    ("ema_h1_soybused",  "SOYBN_USD"),
    ("ema_h1_wheatusd",  "WHEAT_USD"),
    ("ema_h1_sugarusd",  "SUGAR_USD"),
    # Stock indices -- session hours naturally limit signals (flat candles outside market hours)
    ("ema_h1_spx500",   "SPX500_USD"),
    ("ema_h1_nas100",   "NAS100_USD"),
    ("ema_h1_us30",     "US30_USD"),
    ("ema_h1_us2000",   "US2000_USD"),
    ("ema_h1_de30",     "DE30_EUR"),
    ("ema_h1_eu50",     "EU50_EUR"),
    ("ema_h1_jp225",    "JP225_USD"),
    ("ema_h1_au200",    "AU200_AUD"),
]
MACD_H1_SLEEVES = [
    # Pairs passing MACD H1 backtest that are NOT already covered by EMA H1
    ("macd_h1_usdjpy", "USD_JPY"),
    ("macd_h1_eurcad", "EUR_CAD"),
    ("macd_h1_gbpcad", "GBP_CAD"),
    ("macd_h1_gbpchf", "GBP_CHF"),
]
ORB_SLEEVES = [
    # Opening Range Breakout -- London (08:00 UTC) + NY (13:00 UTC) sessions
    ("orb_m30_eurjpy", "EUR_JPY"),
    ("orb_m30_chfjpy", "CHF_JPY"),
    ("orb_m30_cadjpy", "CAD_JPY"),
    ("orb_m30_audjpy", "AUD_JPY"),
    ("orb_m30_gbpjpy", "GBP_JPY"),
    ("orb_m30_nzdjpy", "NZD_JPY"),
    ("orb_m30_audchf", "AUD_CHF"),
    ("orb_m30_euraud", "EUR_AUD"),
    ("orb_m30_usdjpy", "USD_JPY"),
    ("orb_m30_eurcad", "EUR_CAD"),
]
PDHL_SLEEVES = [
    # Previous Day High/Low breakout -- H1 entry, fires hourly
    ("pdhl_gbpusd",   "GBP_USD"),
    ("pdhl_eurjpy",   "EUR_JPY"),
    ("pdhl_chfjpy",   "CHF_JPY"),
    ("pdhl_audjpy",   "AUD_JPY"),
    ("pdhl_gbpjpy",   "GBP_JPY"),
    ("pdhl_nzdjpy",   "NZD_JPY"),
    ("pdhl_usdjpy",   "USD_JPY"),
    ("pdhl_bcousd",   "BCO_USD"),
    ("pdhl_xauusd",   "XAU_USD"),
    ("pdhl_xagusd",   "XAG_USD"),
    ("pdhl_natgas",   "NATGAS_USD"),
    ("pdhl_spx500",   "SPX500_USD"),
    ("pdhl_nas100",   "NAS100_USD"),
    ("pdhl_us30",     "US30_USD"),
]
ALLOCATION_FRACTION = 1 / (len(SLEEVES) + len(M30_SLEEVES) + len(H1_SLEEVES) + len(MACD_H1_SLEEVES) + len(ORB_SLEEVES) + len(PDHL_SLEEVES))
RISK_PCT     = 0.25   # M30 BBMRT risk -- revert to 0.01 for funded account
H1_RISK_PCT  = 0.25   # H1 EMA + MACD H1 risk
ORB_RISK_PCT = 0.25   # ORB intraday risk
ORB_TP_MULT  = 1.5    # TP = 1.5x the opening range width
_ORB_SESSION_HOURS = {8, 13}  # UTC: London open, NY open
TRAIL_MULT   = 2.0    # ATR multiplier for trailing stop on H1 EMA/MACD
MIN_ATR_PCT  = 0.0008 # volatility filter: skip if ATR < 0.08% of price
WARMUP_CANDLES = 500
M30_WARMUP = 100
H1_WARMUP  = 200     # H1 bars needed for EMA(30)/MACD(26) warmup
W_WARMUP   = 15      # weekly bars for trend filter


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


def _load_h1_bars(b: broker.OandaBroker, instrument: str) -> pd.DataFrame:
    raw = b.get_candles(instrument, granularity="H1", count=H1_WARMUP)
    rows = []
    for c in raw:
        if not c["complete"]:
            continue
        mid = c["mid"]
        rows.append({"time": c["time"], "Open": float(mid["o"]),
                     "High": float(mid["h"]), "Low": float(mid["l"]), "Close": float(mid["c"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _load_weekly_bars(b: broker.OandaBroker, instrument: str) -> pd.DataFrame:
    raw = b.get_candles(instrument, granularity="W", count=W_WARMUP)
    rows = []
    for c in raw:
        if not c["complete"]:
            continue
        mid = c["mid"]
        rows.append({"time": c["time"], "Close": float(mid["c"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _load_m30_bars(b: broker.OandaBroker, instrument: str) -> pd.DataFrame:
    raw = b.get_candles(instrument, granularity="M30", count=M30_WARMUP)
    rows = []
    for c in raw:
        if not c["complete"]:
            continue
        mid = c["mid"]
        rows.append(
            {
                "time": c["time"],
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _load_candles(b: broker.OandaBroker, instrument: str) -> pd.DataFrame:
    raw = b.get_candles(instrument, granularity="D", count=WARMUP_CANDLES)
    rows = []
    for c in raw:
        if not c["complete"]:
            continue
        mid = c["mid"]
        rows.append(
            {
                "time": c["time"],
                "Open": float(mid["o"]),
                "High": float(mid["h"]),
                "Low": float(mid["l"]),
                "Close": float(mid["c"]),
            }
        )
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def _bars_held_since(df: pd.DataFrame, open_time: str) -> int:
    """Number of completed daily bars between the bar that was current when
    the trade was opened and the latest bar now -- matches the backtest's
    self._bars_held, which starts at 0 on the entry bar and increments once
    per bar thereafter."""
    entry_idx = df.index.get_indexer([pd.Timestamp(open_time)], method="pad")[0]
    return (len(df) - 1) - entry_idx


def _tagged_trade(b: broker.OandaBroker, instrument: str, tag: str) -> dict | None:
    for t in b.get_open_trades(instrument):
        if t.get("clientExtensions", {}).get("tag") == tag:
            return t
    return None


def _opposite_direction_conflict(b: broker.OandaBroker, instrument: str, own_tag: str, direction: str) -> bool:
    """True if another sleeve already holds the opposite side of
    `instrument` -- opening into it would have OANDA net/reduce the other
    sleeve's trade on this non-hedging account. See module docstring."""
    for t in b.get_open_trades(instrument):
        if t.get("clientExtensions", {}).get("tag") == own_tag:
            continue
        other_is_long = float(t["currentUnits"]) > 0
        if (direction == "long") != other_is_long:
            other_tag = t.get("clientExtensions", {}).get("tag", "untagged")
            _log(
                f"{own_tag}: wants {direction}, but {other_tag} already holds the opposite "
                f"side of {instrument} -- skipping to avoid netting on this non-hedging account"
            )
            return True
    return False


def _open_and_journal(
    b: broker.OandaBroker,
    tag: str,
    instrument: str,
    units: int,
    direction: str,
    sl: float,
    tp: float | None = None,
) -> None:
    resp = b.place_market_order(instrument, units, stop_loss_price=sl, take_profit_price=tp, client_tag=tag)
    fill = resp.get("orderFillTransaction")
    if fill is None:
        _log(f"{tag}: order did not fill -- {resp.get('orderCancelTransaction', resp)}")
        return
    journal.record_open(tag, instrument, direction, units, float(fill["price"]), sl, tp)


def _close_and_journal(b: broker.OandaBroker, trade_id: str, tag: str, instrument: str, reason: str) -> None:
    resp = b.close_trade(trade_id)
    fill = resp.get("orderFillTransaction")
    pl = float(fill["pl"]) if fill else None
    if pl is not None:
        _log(f"{tag}: closed ({reason}), realized P/L = {pl:+.2f}")
    journal.record_close(tag, instrument, reason, pl)


def run_bbmrt_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    Strat = BollingerMeanReversionTrendFilter
    df = _load_candles(b, instrument)
    if len(df) < Strat.trend_period + 5:
        _log(f"{tag}: not enough history ({len(df)} bars), skipping")
        return

    mid = sma(df["Close"], Strat.bb_period)
    sd = rolling_std(df["Close"], Strat.bb_period)
    a = atr(df["High"], df["Low"], df["Close"], Strat.atr_period)
    trend = ema(df["Close"], Strat.trend_period)

    price = df["Close"].iloc[-1]
    m, s, av, t = mid.iloc[-1], sd.iloc[-1], a.iloc[-1], trend.iloc[-1]
    upper, lower = m + Strat.bb_k * s, m - Strat.bb_k * s
    stop_distance = Strat.sl_atr_mult * av

    trade = _tagged_trade(b, instrument, tag)

    if trade is not None:
        is_long = float(trade["currentUnits"]) > 0
        bars_held = _bars_held_since(df, trade["openTime"])
        _log(
            f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']} "
            f"({bars_held} bars held), price={price:.5f} mid={m:.5f}"
        )
        if is_long and price >= m:
            _log(f"{tag}: price reverted to mean -- closing long")
            _close_and_journal(b, trade["id"], tag, instrument, "mean_reversion")
        elif not is_long and price <= m:
            _log(f"{tag}: price reverted to mean -- closing short")
            _close_and_journal(b, trade["id"], tag, instrument, "mean_reversion")
        elif bars_held >= Strat.max_hold:
            _log(f"{tag}: time stop ({bars_held} >= {Strat.max_hold} bars) -- closing")
            _close_and_journal(b, trade["id"], tag, instrument, "time_stop")
        else:
            _log(f"{tag}: holding, no exit condition met")
        return

    _log(
        f"{tag}: flat. price={price:.5f} lower={lower:.5f} upper={upper:.5f} "
        f"trend_ema={t:.5f} atr={av:.5f}"
    )

    if stop_distance <= 0:
        _log(f"{tag}: ATR not ready, skipping")
        return

    risk_amount = sleeve_equity * RISK_PCT
    units = int(risk_amount / stop_distance)
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    if price < lower and price > t:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_distance
        _log(f"{tag}: BUY signal (dip in uptrend) -- {units} units, sl={sl:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl)
    elif price > upper and price < t:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_distance
        _log(f"{tag}: SELL signal (pop in downtrend) -- {units} units, sl={sl:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl)
    else:
        _log(f"{tag}: no signal")


def run_ema_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    Strat = EmaCrossoverAtr
    df = _load_candles(b, instrument)
    if len(df) < Strat.slow + 5:
        _log(f"{tag}: not enough history ({len(df)} bars), skipping")
        return

    ema_fast = ema(df["Close"], Strat.fast)
    ema_slow = ema(df["Close"], Strat.slow)
    a = atr(df["High"], df["Low"], df["Close"], Strat.atr_period)

    price = df["Close"].iloc[-1]
    av = a.iloc[-1]
    fast_now, slow_now = ema_fast.iloc[-1], ema_slow.iloc[-1]
    fast_prev, slow_prev = ema_fast.iloc[-2], ema_slow.iloc[-2]
    cross_up = fast_prev <= slow_prev and fast_now > slow_now
    cross_dn = fast_prev >= slow_prev and fast_now < slow_now

    trade = _tagged_trade(b, instrument, tag)

    if trade is not None:
        is_long = float(trade["currentUnits"]) > 0
        _log(f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']}, price={price:.5f}")
        flips = cross_dn if is_long else cross_up
        if not flips:
            _log(f"{tag}: holding, no opposite crossover -- broker-side SL/TP still governs exit")
            return
        _log(f"{tag}: opposite crossover -- closing to flip")
        _close_and_journal(b, trade["id"], tag, instrument, "opposite_crossover")
    else:
        _log(f"{tag}: flat. price={price:.5f} ema_fast={fast_now:.5f} ema_slow={slow_now:.5f} atr={av:.5f}")

    if av != av or av <= 0:
        _log(f"{tag}: ATR not ready, skipping")
        return
    if not (cross_up or cross_dn):
        _log(f"{tag}: no signal")
        return

    stop_distance = Strat.sl_atr_mult * av
    risk_amount = sleeve_equity * RISK_PCT
    units = int(risk_amount / stop_distance)
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    direction = "long" if cross_up else "short"
    if _opposite_direction_conflict(b, instrument, tag, direction):
        return

    if cross_up:
        sl, tp = price - stop_distance, price + Strat.tp_atr_mult * av
        _log(f"{tag}: BUY signal (EMA cross up) -- {units} units, sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)
    else:
        sl, tp = price + stop_distance, price - Strat.tp_atr_mult * av
        _log(f"{tag}: SELL signal (EMA cross down) -- {units} units, sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)


def run_bbmrt_m30_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    """Daily BBMRT bias + M30 RSI pullback entry. Checked once at 22:30 BST
    daily run -- M30 RSI crossover only evaluated at run time, not intraday."""
    Strat = BollingerMeanReversionTrendFilter
    df = _load_candles(b, instrument)
    if len(df) < Strat.trend_period + 5:
        _log(f"{tag}: not enough daily history ({len(df)} bars), skipping")
        return

    bb_mid = sma(df["Close"], Strat.bb_period)
    bb_std_ser = rolling_std(df["Close"], Strat.bb_period)
    trend = ema(df["Close"], Strat.trend_period)
    a = atr(df["High"], df["Low"], df["Close"], Strat.atr_period)

    price_d = df["Close"].iloc[-1]
    lower_d = (bb_mid - Strat.bb_k * bb_std_ser).iloc[-1]
    upper_d = (bb_mid + Strat.bb_k * bb_std_ser).iloc[-1]
    trend_d = trend.iloc[-1]
    mid_d   = bb_mid.iloc[-1]
    atr_d   = a.iloc[-1]

    if price_d < lower_d and price_d > trend_d:
        bias = 1
    elif price_d > upper_d and price_d < trend_d:
        bias = -1
    else:
        bias = 0

    _log(
        f"{tag}: daily price={price_d:.5f} lower={lower_d:.5f} upper={upper_d:.5f} "
        f"trend={trend_d:.5f} bias={'LONG' if bias == 1 else 'SHORT' if bias == -1 else 'NONE'}"
    )

    trade = _tagged_trade(b, instrument, tag)

    if trade is not None:
        is_long = float(trade["currentUnits"]) > 0
        m30 = _load_m30_bars(b, instrument)
        current_price = m30["Close"].iloc[-1] if not m30.empty else price_d
        _log(f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']}, price={current_price:.5f}")
        if (is_long and current_price >= mid_d) or (not is_long and current_price <= mid_d):
            _log(f"{tag}: price reverted to mean -- closing")
            _close_and_journal(b, trade["id"], tag, instrument, "mean_reversion")
        else:
            _log(f"{tag}: holding, no exit condition met")
        return

    if bias == 0:
        _log(f"{tag}: no daily bias signal")
        return

    m30 = _load_m30_bars(b, instrument)
    if len(m30) < 20:
        _log(f"{tag}: not enough M30 history ({len(m30)} bars), skipping")
        return

    m30_rsi_series = rsi(m30["Close"], 14)
    rsi_now  = m30_rsi_series.iloc[-1]
    rsi_prev = m30_rsi_series.iloc[-2]
    _log(f"{tag}: M30 RSI prev={rsi_prev:.1f} now={rsi_now:.1f}")

    if atr_d != atr_d or atr_d <= 0:
        _log(f"{tag}: daily ATR not ready, skipping")
        return

    stop_dist   = 2.0 * atr_d
    risk_amount = sleeve_equity * RISK_PCT
    units       = int(risk_amount / stop_dist)
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    price = m30["Close"].iloc[-1]

    if bias == 1 and rsi_prev < 40 and rsi_now >= 40:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_dist
        tp = price + 3.0 * atr_d
        _log(f"{tag}: BUY (BBMRT daily dip + M30 RSI recovery) -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)
    elif bias == -1 and rsi_prev > 60 and rsi_now <= 60:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_dist
        tp = price - 3.0 * atr_d
        _log(f"{tag}: SELL (BBMRT daily pop + M30 RSI fade) -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)
    else:
        _log(f"{tag}: no M30 entry signal (RSI crossover not triggered at this run time)")


def run_orb_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    """Opening Range Breakout on M30. Silently skips unless the second-to-last
    completed bar was a session open (08:00 or 13:00 UTC). Entry when the
    following bar breaks the opening range; SL = opposite end of range."""
    raw = b.get_candles(instrument, granularity="M30", count=10)
    completed = []
    for c in raw:
        if not c["complete"]:
            continue
        mid = c["mid"]
        t = pd.to_datetime(c["time"])
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        completed.append({"time": t, "High": float(mid["h"]), "Low": float(mid["l"]), "Close": float(mid["c"])})

    if len(completed) < 2:
        return

    or_bar     = completed[-2]
    signal_bar = completed[-1]
    if or_bar["time"].hour not in _ORB_SESSION_HOURS or or_bar["time"].minute != 0:
        return  # not a session-open bar; silent exit (fires every 30 min)

    or_high  = or_bar["High"]
    or_low   = or_bar["Low"]
    or_range = or_high - or_low
    if or_range <= 0:
        return

    session  = "London" if or_bar["time"].hour == 8 else "NY"
    price    = signal_bar["Close"]
    _log(f"{tag}: ORB {session} -- or_high={or_high:.5f} or_low={or_low:.5f} price={price:.5f}")

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        is_long = float(trade["currentUnits"]) > 0
        _log(f"{tag}: already in {'LONG' if is_long else 'SHORT'} -- ORB SL/TP governs exit")
        return

    stop_dist   = or_range
    risk_amount = sleeve_equity * ORB_RISK_PCT
    units       = int(risk_amount / stop_dist)
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    if price > or_high:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = or_low
        tp = price + ORB_TP_MULT * or_range
        _log(f"{tag}: BUY ORB breakout -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)
    elif price < or_low:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = or_high
        tp = price - ORB_TP_MULT * or_range
        _log(f"{tag}: SELL ORB breakout -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)
    else:
        _log(f"{tag}: no ORB breakout (price inside range)")


def run_h1_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    """H1 EMA(10/30) crossover. Trailing stop replaces fixed TP.
    Volatility filter skips entries when ATR is too small.
    Weekly EMA trend filter gates entry direction."""
    h1 = _load_h1_bars(b, instrument)
    if len(h1) < 35:
        _log(f"{tag}: not enough H1 history ({len(h1)} bars), skipping")
        return

    fast = ema(h1["Close"], 10)
    slow = ema(h1["Close"], 30)
    a    = atr(h1["High"], h1["Low"], h1["Close"], 14)

    price    = h1["Close"].iloc[-1]
    av       = a.iloc[-1]
    fast_now = fast.iloc[-1]; fast_prev = fast.iloc[-2]
    slow_now = slow.iloc[-1]; slow_prev = slow.iloc[-2]
    cross_up = fast_prev <= slow_prev and fast_now > slow_now
    cross_dn = fast_prev >= slow_prev and fast_now < slow_now

    trade = _tagged_trade(b, instrument, tag)

    if trade is not None:
        is_long = float(trade["currentUnits"]) > 0
        _log(f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']}, price={price:.5f}")
        # Trailing stop -- ratchet SL toward price each hour
        if av > 0:
            trail = price - TRAIL_MULT * av if is_long else price + TRAIL_MULT * av
            current_sl = float(trade.get("stopLossOrder", {}).get("price", 0))
            if is_long and trail > current_sl:
                b.update_trade_sl(trade["id"], trail)
                _log(f"{tag}: trailing stop raised to {trail:.5f}")
            elif not is_long and trail < current_sl:
                b.update_trade_sl(trade["id"], trail)
                _log(f"{tag}: trailing stop lowered to {trail:.5f}")
        # Flip on opposite crossover
        flips = cross_dn if is_long else cross_up
        if flips:
            _log(f"{tag}: opposite H1 crossover -- closing to flip")
            _close_and_journal(b, trade["id"], tag, instrument, "opposite_crossover")
        else:
            _log(f"{tag}: holding")
            return

    if av != av or av <= 0:
        _log(f"{tag}: ATR not ready, skipping")
        return
    if not (cross_up or cross_dn):
        _log(f"{tag}: no H1 crossover signal")
        return

    # Volatility filter
    if av < price * MIN_ATR_PCT:
        _log(f"{tag}: ATR too low ({av:.5f}) -- skipping low-volatility entry")
        return

    # Weekly trend filter
    try:
        wk = _load_weekly_bars(b, instrument)
        if len(wk) >= 2:
            w_ema = ema(wk["Close"], 10)
            w_trend_up = w_ema.iloc[-1] > w_ema.iloc[-2]
            if cross_up and not w_trend_up:
                _log(f"{tag}: weekly trend is DOWN -- skipping BUY signal")
                return
            if cross_dn and w_trend_up:
                _log(f"{tag}: weekly trend is UP -- skipping SELL signal")
                return
    except Exception:
        pass  # weekly filter optional -- proceed if data unavailable

    stop_dist   = 1.5 * av
    risk_amount = sleeve_equity * H1_RISK_PCT
    units       = int(risk_amount / stop_dist)
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    direction = "long" if cross_up else "short"
    if _opposite_direction_conflict(b, instrument, tag, direction):
        return

    if cross_up:
        sl = price - stop_dist
        _log(f"{tag}: BUY H1 EMA cross -- {units} units sl={sl:.5f} (trailing, no fixed TP)")
        _open_and_journal(b, tag, instrument, units, "long", sl)
    else:
        sl = price + stop_dist
        _log(f"{tag}: SELL H1 EMA cross -- {units} units sl={sl:.5f} (trailing, no fixed TP)")
        _open_and_journal(b, tag, instrument, -units, "short", sl)


def run_macd_h1_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    """H1 MACD(12,26,9) signal-line crossover. SL=1.5xATR, TP=2.5xATR."""
    h1 = _load_h1_bars(b, instrument)
    if len(h1) < 40:
        _log(f"{tag}: not enough H1 history ({len(h1)} bars), skipping")
        return

    ml, sig = macd(h1["Close"], 12, 26, 9)
    a        = atr(h1["High"], h1["Low"], h1["Close"], 14)

    price    = h1["Close"].iloc[-1]
    av       = a.iloc[-1]
    ml_now   = ml.iloc[-1];  ml_prev  = ml.iloc[-2]
    sig_now  = sig.iloc[-1]; sig_prev = sig.iloc[-2]
    cross_up = ml_prev <= sig_prev and ml_now > sig_now
    cross_dn = ml_prev >= sig_prev and ml_now < sig_now

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        is_long = float(trade["currentUnits"]) > 0
        _log(f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']}, price={price:.5f}")
        if av > 0:
            trail = price - TRAIL_MULT * av if is_long else price + TRAIL_MULT * av
            current_sl = float(trade.get("stopLossOrder", {}).get("price", 0))
            if is_long and trail > current_sl:
                b.update_trade_sl(trade["id"], trail)
                _log(f"{tag}: trailing stop raised to {trail:.5f}")
            elif not is_long and trail < current_sl:
                b.update_trade_sl(trade["id"], trail)
                _log(f"{tag}: trailing stop lowered to {trail:.5f}")
        flips = cross_dn if is_long else cross_up
        if flips:
            _log(f"{tag}: opposite MACD cross -- closing to flip")
            _close_and_journal(b, trade["id"], tag, instrument, "opposite_crossover")
        else:
            _log(f"{tag}: holding")
            return

    if av != av or av <= 0:
        _log(f"{tag}: ATR not ready, skipping")
        return
    if not (cross_up or cross_dn):
        _log(f"{tag}: no MACD crossover signal")
        return

    if av < price * MIN_ATR_PCT:
        _log(f"{tag}: ATR too low ({av:.5f}) -- skipping low-volatility entry")
        return

    stop_dist   = 1.5 * av
    risk_amount = sleeve_equity * H1_RISK_PCT
    units       = int(risk_amount / stop_dist)
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    direction = "long" if cross_up else "short"
    if _opposite_direction_conflict(b, instrument, tag, direction):
        return

    if cross_up:
        sl = price - stop_dist
        _log(f"{tag}: BUY MACD cross -- {units} units sl={sl:.5f} (trailing, no fixed TP)")
        _open_and_journal(b, tag, instrument, units, "long", sl)
    else:
        sl = price + stop_dist
        _log(f"{tag}: SELL MACD cross -- {units} units sl={sl:.5f} (trailing, no fixed TP)")
        _open_and_journal(b, tag, instrument, -units, "short", sl)


def run_pdhl_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    """Previous Day High/Low breakout. Checks hourly: enter when H1 close
    breaks above yesterday's high (BUY) or below yesterday's low (SELL).
    SL = midpoint of yesterday's range. TP = 1.5x range beyond entry."""
    d1_raw = b.get_candles(instrument, granularity="D", count=3)
    d1_bars = [c for c in d1_raw if c["complete"]]
    if not d1_bars:
        return

    yesterday  = d1_bars[-1]
    prev_high  = float(yesterday["mid"]["h"])
    prev_low   = float(yesterday["mid"]["l"])
    prev_mid   = (prev_high + prev_low) / 2
    prev_rng   = prev_high - prev_low
    if prev_rng <= 0:
        return

    h1_raw = b.get_candles(instrument, granularity="H1", count=3)
    h1_bars = [c for c in h1_raw if c["complete"]]
    if not h1_bars:
        return
    price = float(h1_bars[-1]["mid"]["c"])

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        is_long = float(trade["currentUnits"]) > 0
        _log(f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']}, price={price:.5f} -- PDH/PDL SL/TP governs exit")
        return

    _log(f"{tag}: PDH/PDL prev_high={prev_high:.5f} prev_low={prev_low:.5f} price={price:.5f}")

    stop_dist   = abs(price - prev_mid)
    if stop_dist <= 0:
        return
    risk_amount = sleeve_equity * H1_RISK_PCT
    units       = int(risk_amount / stop_dist)
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    if price > prev_high:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = prev_mid
        tp = price + 1.5 * prev_rng
        _log(f"{tag}: BUY PDH breakout -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)
    elif price < prev_low:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = prev_mid
        tp = price - 1.5 * prev_rng
        _log(f"{tag}: SELL PDL breakout -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)
    else:
        _log(f"{tag}: price inside yesterday's range -- no signal")


def main() -> None:
    b = broker.from_env()
    account_equity = float(b.account_summary()["account"]["balance"])
    sleeve_equity = account_equity * ALLOCATION_FRACTION
    total_sleeves = len(SLEEVES) + len(M30_SLEEVES) + len(H1_SLEEVES) + len(MACD_H1_SLEEVES) + len(ORB_SLEEVES) + len(PDHL_SLEEVES)
    _log(
        f"account equity={account_equity:.2f}, per-sleeve allocation={sleeve_equity:.2f} "
        f"({ALLOCATION_FRACTION:.0%} each across {total_sleeves} sleeves)"
    )
    for tag, instrument, kind in SLEEVES:
        try:
            if kind == "bbmrt":
                run_bbmrt_sleeve(b, tag, instrument, sleeve_equity)
            else:
                run_ema_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in M30_SLEEVES:
        try:
            run_bbmrt_m30_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in H1_SLEEVES:
        try:
            run_h1_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in MACD_H1_SLEEVES:
        try:
            run_macd_h1_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in ORB_SLEEVES:
        try:
            run_orb_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in PDHL_SLEEVES:
        try:
            run_pdhl_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")


if __name__ == "__main__":
    main()
