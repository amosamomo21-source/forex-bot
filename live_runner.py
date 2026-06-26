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
from strategies import atr, ema, rsi, rolling_std, sma  # noqa: E402

SLEEVES = [
    ("bbmrt_eurusd", "EUR_USD", "bbmrt"),
    ("bbmrt_gbpusd", "GBP_USD", "bbmrt"),
    ("ema_gbpusd", "GBP_USD", "ema"),
    ("ema_usdjpy", "USD_JPY", "ema"),
    ("ema_audusd", "AUD_USD", "ema"),
]
M30_SLEEVES = [
    ("bbmrt_m30_eurusd", "EUR_USD"),
    ("bbmrt_m30_gbpusd", "GBP_USD"),
    ("bbmrt_m30_xauusd", "XAU_USD"),
    ("bbmrt_m30_eurjpy", "EUR_JPY"),
    ("bbmrt_m30_chfjpy", "CHF_JPY"),
    ("bbmrt_m30_audchf", "AUD_CHF"),
]
ALLOCATION_FRACTION = 1 / (len(SLEEVES) + len(M30_SLEEVES))
RISK_PCT = 0.05       # 5% per trade on demo to accelerate forward testing; revert to 0.01 for funded account
WARMUP_CANDLES = 500  # comfortably more than either strategy's longest lookback (trend_period=100, slow=50)
M30_WARMUP = 100      # M30 bars needed for RSI(14) warmup


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


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


def main() -> None:
    b = broker.from_env()
    account_equity = float(b.account_summary()["account"]["balance"])
    sleeve_equity = account_equity * ALLOCATION_FRACTION
    _log(
        f"account equity={account_equity:.2f}, per-sleeve allocation={sleeve_equity:.2f} "
        f"({ALLOCATION_FRACTION:.0%} each across {len(SLEEVES) + len(M30_SLEEVES)} sleeves)"
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


if __name__ == "__main__":
    main()
