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

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
    # All removed 2026-07-06: win_rate_audit.py showed all 17 pairs with negative Sharpe
    # (avg -2.47) and WR 37–43%, below the ~50% break-even for 1:1 mean-reversion R:R.
    # Second consecutive audit confirming the same result (portfolio_audit.py: avg Sh -3.50).
    # Account drawdown from $100k to $91k consistent with these numbers.
    # No open positions at removal time.
]
H1_SLEEVES = [
    # All removed 2026-07-04: portfolio audit showed 32.6% aggregate WR across all 11 sleeves.
    # EMA(10/30) crossover is structurally low-WR (trend-following). Incompatible with
    # high-WR portfolio goal. Open positions (NAS100, JP225) will close via broker-side SL.
]
MACD_H1_SLEEVES = [
    # Removed 2026-07-04: 34.2% WR, Sharpe -3.49 in 10y backtest. Same structural issue
    # as H1 EMA — trend-following with inherently low WR.
]
ORB_SLEEVES = [
    # All removed 2026-07-05: 10y audit showed avg Sharpe -3.04 across all 10 pairs.
    # WR ~51% looks above the theoretical 40% break-even, but profit factor was 0.63–0.75
    # (losing money) because entry at breakout-bar close degrades actual R:R to ~0.8:1.
    # The strategy requires retracement entry at the ORB level to achieve true 1.5:1 R:R,
    # which cannot be implemented mechanically in the live runner.
]
PDHL_SLEEVES = [
    # Previous Day High/Low breakout -- H1 entry, fires hourly
    # FX pairs removed 2026-07-04: all had negative 10y Sharpe in static backtest,
    # confirmed by live losses week of Jun 30–Jul 3. JPY crosses (5 pairs) moved
    # together causing correlated multi-trade SL hits on same day (-$3,930 in one session).
    # pdhl_gbpusd removed: 10y Sharpe -0.20
    # pdhl_eurjpy removed: 10y Sharpe -0.29
    # pdhl_chfjpy removed: 10y Sharpe -0.60, live SL -$1,713 Jul 2
    # pdhl_audjpy removed: 10y Sharpe -0.31
    # pdhl_gbpjpy removed: 10y Sharpe +0.09 (borderline, not enough edge)
    # pdhl_nzdjpy removed: 10y Sharpe +0.21 (borderline, not enough edge)
    # pdhl_usdjpy removed: 10y Sharpe -0.28
    # pdhl_spx500 removed 2026-07-02: 10y Sharpe -0.75, live P/L -$1,328 (2 trades)
    # pdhl_us30   removed 2026-07-02: 10y Sharpe -0.59, live P/L -$1,850 (3 trades)
    # Commodity PDHL — only NATGAS passes 10y backtest (Sharpe 0.45, Ann% 1.4%)
    # pdhl_xagusd removed 2026-07-04: 10y Sharpe 0.06, effectively flat
    # pdhl_xauusd removed 2026-07-04: 10y Sharpe -0.15
    # pdhl_bcousd removed 2026-07-04: 10y Sharpe -0.16
    ("pdhl_natgas",   "NATGAS_USD"),
]
CONSEC_D1_SLEEVES = [
    # Consecutive-day mean reversion on WHEAT D1. SL=1.5 ATR, TP=1.0 ATR.
    # streak=3: Sharpe 1.06, WR 68.7%, N=316, max DD -2.4%
    # streak=4: Sharpe 1.05, WR 70.4%, N=162, max DD -1.2% — fires on stronger exhaustion
    # Both run together: streak=4 fires on day 4 of a streak, streak=3 already entered day 3.
    # Same direction, so no hedging conflict. Scales into high-conviction reversals.
    ("consec_d1_wheatusd_3", "WHEAT_USD", 3),
    ("consec_d1_wheatusd_4", "WHEAT_USD", 4),
]
FVG_M30_SLEEVES = [
    # Fair Value Gap fill — M30, EMA(200) trend filter, session open only (07-09 + 13-15 UTC).
    # Entry: price retraces into 3-candle FVG zone. SL beyond FVG edge. TP 3:1 R:R.
    # 10y backtest (fvg_scan.py): NATGAS Sh +0.38 Ann +1.8% WR 28% N=1405 PF=1.17
    #                              GBP/CHF Sh +0.38 Ann +0.6% WR 32% N=168  PF=1.33
    # Fires via fvg_session_runner.py cron (*/30 7-9,13-15 * * 1-5 UTC), not main cron.
    ("fvg_m30_natgas",  "NATGAS_USD"),
    ("fvg_m30_gbpchf",  "GBP_CHF"),
]
RSI_D1_SLEEVES = [
    # RSI extreme fade — D1 mean reversion at statistical exhaustion. SL=1.5 ATR.
    # (tag, instrument, oversold, overbought, tp_mult)
    # WHEAT RSI<15: Sh +1.01, WR 81.2%, N=16 (10y), PF=5.26 — TP=1.0×SL (tight, max WR)
    ("rsi_extreme_wheatusd", "WHEAT_USD", 15, 85, 1.0),
    # JP225 RSI<20: Sh +0.45, WR 56.2%, N=80 (10y), Ann +1.3% — most trades of ADD group
    ("rsi_extreme_jp225",    "JP225_USD", 20, 80, 0.8),
    # UK100 RSI<20: Sh +0.75, WR 70.9%, N=55 (10y), Ann +0.5% — best Sharpe of new candidates
    ("rsi_extreme_uk100",    "UK100_GBP", 20, 80, 0.8),
    # USD/CAD RSI<15: Sh +0.54, WR 73.9%, N=23 (10y), Ann +0.2% — near 75% WR target
    ("rsi_extreme_usdcad",   "USD_CAD",   15, 85, 0.8),
]
RSI_DIV_D1_SLEEVES = [
    # RSI divergence — D1. Price lower low + RSI higher low → BUY (and reverse for SELL).
    # (tag, instrument, pivot_n, rr, rsi_extreme_filter)
    # AUD/JPY N5 3:1 NoFlt:  Sh +1.26, WR 52.9%, Ann +1.2%, N=34 — best of full 34-instr scan
    ("rsi_div_audjpy", "AUD_JPY",  5, 3.0, False),
    # NZD/USD N5 3:1 Extreme: Sh +0.78, WR 40.5%, Ann +0.9%, N=42 — strong with RSI<45 filter
    ("rsi_div_nzdusd", "NZD_USD",  5, 3.0, True),
    # FR40    N5 2:1 NoFlt:  Sh +0.85, WR 54.1%, Ann +0.5%, N=37 — highest WR of all ADD
    ("rsi_div_fr40",   "FR40_EUR", 5, 2.0, False),
]
ENGULF_D1_SLEEVES = [
    # Bearish/Bullish engulfing candle — D1, 3:1 R:R, 1.5 ATR SL.
    # SPX500: Sh +1.04, OOS Sh +1.70, 9/10 profitable years, DD -1.8%
    ("engulf_spx500", "SPX500_USD", 3.0),
]
DONCHIAN_D1_SLEEVES = [
    # Donchian 20-day channel breakout — D1, 3:1 R:R, 1.5 ATR SL.
    # XAU/USD: Sh +0.99, OOS Sh +2.30, 8/10 profitable years, robust across periods
    ("donchian_xauusd", "XAU_USD", 20, 3.0),
]
VSF_D1_SLEEVES = [
    # Volatility Spike Fade — D1. Range > 2×ATR, close near extreme → mean reversion.
    # UK100: Sh +1.04, OOS Sh +1.26 (R3), 6/10 profitable years, DD -1.2%.
    # vol_mult=2.0 is the sweet spot; 1.5× goes negative, confirming real edge at 2×.
    ("vsf_uk100", "UK100_GBP", 2.0, 3.0),
]
ALLOCATION_FRACTION = 1 / (len(SLEEVES) + len(M30_SLEEVES) + len(H1_SLEEVES) + len(MACD_H1_SLEEVES) + len(ORB_SLEEVES) + len(PDHL_SLEEVES) + len(CONSEC_D1_SLEEVES) + len(FVG_M30_SLEEVES) + len(RSI_D1_SLEEVES) + len(RSI_DIV_D1_SLEEVES) + len(ENGULF_D1_SLEEVES) + len(DONCHIAN_D1_SLEEVES) + len(VSF_D1_SLEEVES))
# ── Risk mode ─────────────────────────────────────────────────────────────────
# Switch by changing RISK_MODE below.
#   "demo"       0.25% per sleeve — inflated so demo trades are visible
#   "challenge"  0.10% per sleeve — targets FTMO +10%/month within 5% daily DD
#   "funded"     0.01% per sleeve — conservative capital preservation
RISK_MODE = "demo"
_RISK_TABLE = {
    "demo":      (1.5, 1.5, 1.5),
    "challenge": (0.10, 0.10, 0.10),
    "funded":    (0.01, 0.01, 0.01),
}
RISK_PCT, H1_RISK_PCT, ORB_RISK_PCT = _RISK_TABLE[RISK_MODE]
ORB_TP_MULT  = 1.5    # TP = 1.5x the opening range width
_ORB_SESSION_HOURS = {8, 13}  # UTC: London open, NY open
TRAIL_MULT   = 2.0    # ATR multiplier for trailing stop on H1 EMA/MACD
MIN_ATR_PCT  = 0.0008 # volatility filter: skip if ATR < 0.08% of price
H1_SL_MULT   = 1.5    # initial SL distance for H1 EMA entries
ADX_MIN      = 20     # ADX filter: skip H1 EMA entry if trend too weak
BE_TRIGGER   = 1.0    # break-even: floor SL at entry once price moves 1R in profit
WARMUP_CANDLES = 500
M30_WARMUP = 100
H1_WARMUP  = 200     # H1 bars needed for EMA(30)/MACD(26) warmup
W_WARMUP   = 15      # weekly bars for trend filter


_RATE_CACHE: dict = {}           # rate_instrument -> (rate, timestamp)
_RATE_CACHE_TTL = 300            # seconds
PDHL_CLOSED_IDS  = Path(__file__).parent / "pdhl_closed_ids.json"
CONSEC_CLOSED_IDS = Path(__file__).parent / "consec_closed_ids.json"
FVG_CLOSED_IDS   = Path(__file__).parent / "fvg_closed_ids.json"
RSI_CLOSED_IDS   = Path(__file__).parent / "rsi_closed_ids.json"
RSI_DIV_CLOSED_IDS  = Path(__file__).parent / "rsi_div_closed_ids.json"
ENGULF_CLOSED_IDS   = Path(__file__).parent / "engulf_closed_ids.json"
DONCHIAN_CLOSED_IDS = Path(__file__).parent / "donchian_closed_ids.json"
VSF_CLOSED_IDS      = Path(__file__).parent / "vsf_closed_ids.json"
CONSEC_STREAK  = 3    # consecutive closes in same direction before entry
CONSEC_SL_ATR  = 1.5
CONSEC_TP_ATR  = 1.0  # TP < SL intentional — 68%+ win rate covers the asymmetry


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


def _quote_usd_rate(b: broker.OandaBroker, instrument: str) -> float:
    """USD value of 1 point move in the quote currency for 1 unit of this instrument.
    Corrects position sizing for non-USD-quoted pairs (JPY, SGD, AUD, etc.)."""
    if instrument.endswith("_USD"):
        return 1.0
    quote = instrument.split("_")[-1]
    meta = {
        "JPY": ("USD_JPY", True),
        "SGD": ("USD_SGD", True),
        "CAD": ("USD_CAD", True),
        "CHF": ("USD_CHF", True),
        "AUD": ("AUD_USD", False),
        "EUR": ("EUR_USD", False),
        "GBP": ("GBP_USD", False),
    }.get(quote)
    if meta is None:
        return 1.0
    rate_instr, invert = meta
    now = datetime.now(timezone.utc).timestamp()
    if rate_instr in _RATE_CACHE:
        cached_rate, cached_ts = _RATE_CACHE[rate_instr]
        if now - cached_ts < _RATE_CACHE_TTL:
            return 1.0 / cached_rate if invert else cached_rate
    try:
        bars = b.get_candles(rate_instr, granularity="M1", count=2)
        c = [x for x in bars if x.get("complete")] or bars
        rate = float(c[-1]["mid"]["c"])
        _RATE_CACHE[rate_instr] = (rate, now)
        return 1.0 / rate if invert else rate
    except Exception:
        return 1.0


def _journal_pdhl_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    """Detect trades closed by OANDA TP/SL (not by the bot) and journal them."""
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if PDHL_CLOSED_IDS.exists():
        journaled = set(json.loads(PDHL_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag:
            continue
        trade_id = t.get("id", "")
        if trade_id in journaled:
            continue
        units      = abs(float(t.get("initialUnits", 0)))
        close_px   = float(t.get("averageClosePrice", 0))
        realized   = float(t.get("realizedPL", 0))
        direction  = "long" if float(t.get("initialUnits", 0)) > 0 else "short"
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(trade_id)
        PDHL_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


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


_TRIPLE_FLAG = Path(__file__).parent / "triple_next.flag"
_MULT_FLAG   = Path(__file__).parent / "next_mult.flag"


def _open_and_journal(
    b: broker.OandaBroker,
    tag: str,
    instrument: str,
    units: int,
    direction: str,
    sl: float,
    tp: float | None = None,
) -> None:
    if _MULT_FLAG.exists() and instrument != "NATGAS_USD":
        try:
            mult = int(_MULT_FLAG.read_text().strip())
        except Exception:
            mult = 1
        _MULT_FLAG.unlink()
        units = units * mult
        _log(f"{tag}: NEXT_MULT flag active -- units x{mult} = {units}")
    elif _TRIPLE_FLAG.exists():
        units = units * 3
        _TRIPLE_FLAG.unlink()
        _log(f"{tag}: TRIPLE_NEXT flag active -- units multiplied to {units}")
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
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_distance * quote_rate)) if quote_rate > 0 else 0
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
    risk_amount   = sleeve_equity * RISK_PCT
    quote_rate    = _quote_usd_rate(b, instrument)
    units         = int(risk_amount / (stop_distance * quote_rate)) if quote_rate > 0 else 0
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
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
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

    # Minimum range filter: skip noise ranges (< 15 pips).
    # Tiny ranges produce absurdly large position sizes and are not tradeable signals.
    min_range = 0.15 if "JPY" in instrument else 0.0015
    if or_range < min_range:
        _log(f"{tag}: ORB range too narrow ({or_range:.5f} < {min_range}) -- skipping")
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
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return
    ORB_MAX_UNITS = 50_000
    if units > ORB_MAX_UNITS:
        _log(f"{tag}: capping units {units} -> {ORB_MAX_UNITS} (range too tight for risk formula)")
        units = ORB_MAX_UNITS

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
        is_long    = float(trade["currentUnits"]) > 0
        entry_p    = float(trade.get("price", 0))
        current_sl = float(trade.get("stopLossOrder", {}).get("price", 0))
        _log(f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']}, price={price:.5f}")
        if av > 0:
            trail = price - TRAIL_MULT * av if is_long else price + TRAIL_MULT * av
            # Break-even: once 1R profit reached, floor SL at entry
            if is_long:
                if price >= entry_p + BE_TRIGGER * H1_SL_MULT * av and current_sl < entry_p:
                    trail = max(trail, entry_p)
                    _log(f"{tag}: break-even triggered -- SL floored at entry {entry_p:.5f}")
            else:
                if price <= entry_p - BE_TRIGGER * H1_SL_MULT * av and current_sl > entry_p:
                    trail = min(trail, entry_p)
                    _log(f"{tag}: break-even triggered -- SL capped at entry {entry_p:.5f}")
            if is_long and trail > current_sl:
                b.update_trade_sl(trade["id"], trail, instrument)
                _log(f"{tag}: trailing stop raised to {trail:.5f}")
            elif not is_long and trail < current_sl:
                b.update_trade_sl(trade["id"], trail, instrument)
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

    # Weekly trend filter — relaxed to allow trades in flat/sideways markets.
    # Slope measured over 2 weeks; only blocks when clearly directional (>0.1%).
    try:
        wk = _load_weekly_bars(b, instrument)
        if len(wk) >= 3:
            w_ema = ema(wk["Close"], 10)
            w_slope = (w_ema.iloc[-1] - w_ema.iloc[-3]) / w_ema.iloc[-3]
            w_trend_up = w_slope >  0.001  # EMA rising >0.1% over 2 weeks
            w_trend_dn = w_slope < -0.001  # EMA falling >0.1% over 2 weeks
            if cross_up and w_trend_dn:
                _log(f"{tag}: weekly trend is DOWN -- skipping BUY signal")
                return
            if cross_dn and w_trend_up:
                _log(f"{tag}: weekly trend is UP -- skipping SELL signal")
                return
    except Exception:
        pass  # weekly filter optional -- proceed if data unavailable

    stop_dist   = H1_SL_MULT * av
    risk_amount = sleeve_equity * H1_RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
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
                b.update_trade_sl(trade["id"], trail, instrument)
                _log(f"{tag}: trailing stop raised to {trail:.5f}")
            elif not is_long and trail < current_sl:
                b.update_trade_sl(trade["id"], trail, instrument)
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
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
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
    SL = midpoint of yesterday's range. TP = 1.5x range beyond entry.
    Trailing stop + break-even active once in profit."""
    d1_raw = b.get_candles(instrument, granularity="D", count=3)
    d1_bars = [c for c in d1_raw if c["complete"]]
    if not d1_bars:
        return

    yesterday = d1_bars[-1]
    prev_high = float(yesterday["mid"]["h"])
    prev_low  = float(yesterday["mid"]["l"])
    prev_mid  = (prev_high + prev_low) / 2
    prev_rng  = prev_high - prev_low
    if prev_rng <= 0:
        return

    h1_raw = b.get_candles(instrument, granularity="H1", count=50)
    h1_bars = [c for c in h1_raw if c["complete"]]
    if not h1_bars:
        return
    h1_df = pd.DataFrame({
        "High":  [float(c["mid"]["h"]) for c in h1_bars],
        "Low":   [float(c["mid"]["l"]) for c in h1_bars],
        "Close": [float(c["mid"]["c"]) for c in h1_bars],
    })
    av    = atr(h1_df["High"], h1_df["Low"], h1_df["Close"]).iloc[-1]
    price = h1_df["Close"].iloc[-1]

    _PDHL_CLOSE_AT_USD  = 6_000   # close trade in profit when unrealizedPL hits this
    _PDHL_LOCK_AT_USD   = 1_500   # move SL to break-even when unrealizedPL hits this

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        is_long    = float(trade["currentUnits"]) > 0
        entry_p    = float(trade.get("price", 0))
        current_sl = float(trade.get("stopLossOrder", {}).get("price", 0))
        unreal     = float(trade.get("unrealizedPL", 0))
        _log(f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']}, "
             f"price={price:.5f} unrealizedPL={unreal:+.2f}")

        # ── Dollar profit close: take the money at $6k ────────────────────
        if unreal >= _PDHL_CLOSE_AT_USD:
            _log(f"{tag}: PROFIT TARGET ${_PDHL_CLOSE_AT_USD:,.0f} hit "
                 f"(unrealizedPL={unreal:+.2f}) -- closing in profit")
            _close_and_journal(b, trade["id"], tag, instrument, "profit_target")
            return

        if av > 0 and current_sl > 0:
            trail = price - TRAIL_MULT * av if is_long else price + TRAIL_MULT * av

            # ── Dollar break-even: lock trade at entry when up $1,500 ─────
            if unreal >= _PDHL_LOCK_AT_USD:
                if is_long and current_sl < entry_p:
                    trail = max(trail, entry_p)
                    _log(f"{tag}: $1,500 profit lock -- SL floored at entry {entry_p:.5f}")
                elif not is_long and current_sl > entry_p:
                    trail = min(trail, entry_p)
                    _log(f"{tag}: $1,500 profit lock -- SL capped at entry {entry_p:.5f}")
            # ── ATR break-even fallback (existing) ────────────────────────
            elif is_long:
                if price >= entry_p + BE_TRIGGER * H1_SL_MULT * av and current_sl < entry_p:
                    trail = max(trail, entry_p)
                    _log(f"{tag}: break-even triggered -- SL floored at entry {entry_p:.5f}")
            else:
                if price <= entry_p - BE_TRIGGER * H1_SL_MULT * av and current_sl > entry_p:
                    trail = min(trail, entry_p)
                    _log(f"{tag}: break-even triggered -- SL capped at entry {entry_p:.5f}")

            if is_long and trail > current_sl:
                b.update_trade_sl(trade["id"], trail, instrument)
                _log(f"{tag}: trailing stop raised to {trail:.5f}")
            elif not is_long and trail < current_sl:
                b.update_trade_sl(trade["id"], trail, instrument)
                _log(f"{tag}: trailing stop lowered to {trail:.5f}")
        return

    _journal_pdhl_close_if_needed(b, tag, instrument)

    _log(f"{tag}: PDH/PDL prev_high={prev_high:.5f} prev_low={prev_low:.5f} price={price:.5f}")

    stop_dist   = abs(price - prev_mid)
    if stop_dist <= 0:
        return
    risk_amount = sleeve_equity * H1_RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
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


def _journal_consec_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    """Detect WHEAT consec-reversion trades closed by OANDA TP/SL and journal them."""
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if CONSEC_CLOSED_IDS.exists():
        journaled = set(json.loads(CONSEC_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag:
            continue
        trade_id = t.get("id", "")
        if trade_id in journaled:
            continue
        realized = float(t.get("realizedPL", 0))
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(trade_id)
        CONSEC_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


def run_consec_d1_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float, streak: int = CONSEC_STREAK) -> None:
    """Consecutive-day mean reversion — D1 bars, WHEAT_USD.

    Exit: broker-side TP at 1.0 ATR and SL at 1.5 ATR.
    Break-even WR = 60%. streak=3 → WR 68.7%, streak=4 → WR 70.4%.
    """
    df = _load_candles(b, instrument)
    if len(df) < streak + 20:
        _log(f"{tag}: not enough history ({len(df)} bars), skipping")
        return

    a  = atr(df["High"], df["Low"], df["Close"], 14)
    av = a.iloc[-1]
    price = df["Close"].iloc[-1]

    if av <= 0 or np.isnan(av):
        _log(f"{tag}: ATR not ready, skipping")
        return

    _journal_consec_close_if_needed(b, tag, instrument)

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        is_long   = float(trade["currentUnits"]) > 0
        bars_held = _bars_held_since(df, trade["openTime"])
        _log(
            f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']} "
            f"({bars_held} bars held), price={price:.5f} -- TP/SL govern exit"
        )
        return

    closes = df["Close"].values
    n      = streak
    down   = all(closes[-(i + 1)] < closes[-(i + 2)] for i in range(n))
    up     = all(closes[-(i + 1)] > closes[-(i + 2)] for i in range(n))

    _log(
        f"{tag}: flat. price={price:.5f} atr={av:.5f} "
        f"consec_down={down} consec_up={up}"
    )

    if not (down or up):
        _log(f"{tag}: no signal (streak={n} not met)")
        return

    stop_dist   = CONSEC_SL_ATR * av
    risk_amount = sleeve_equity * RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    if down:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_dist
        tp = price + CONSEC_TP_ATR * av
        _log(f"{tag}: BUY ({n} consec down bars) -- {units} units, sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)
    else:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_dist
        tp = price - CONSEC_TP_ATR * av
        _log(f"{tag}: SELL ({n} consec up bars) -- {units} units, sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)


_RSI_D1_SL_ATR = 1.5


def _journal_rsi_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    """Detect RSI extreme trades closed by OANDA TP/SL and journal them."""
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if RSI_CLOSED_IDS.exists():
        journaled = set(json.loads(RSI_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag:
            continue
        trade_id = t.get("id", "")
        if trade_id in journaled:
            continue
        realized = float(t.get("realizedPL", 0))
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(trade_id)
        RSI_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


def run_rsi_d1_sleeve(
    b: broker.OandaBroker,
    tag: str,
    instrument: str,
    sleeve_equity: float,
    oversold: int,
    overbought: int,
    tp_mult: float,
) -> None:
    """RSI extreme fade on D1 bars. Mean reversion at statistical exhaustion.

    Entry: RSI crosses below oversold (BUY) or above overbought (SELL).
    SL: 1.5 ATR beyond entry. TP: tp_mult × SL distance (tight TP → high WR).
    WHEAT RSI<15 → WR 81%, Sh +1.01. JP225 RSI<20 → WR 56%, Sh +0.45.
    """
    df = _load_candles(b, instrument)
    if len(df) < 20:
        _log(f"{tag}: not enough history ({len(df)} bars), skipping")
        return

    _journal_rsi_close_if_needed(b, tag, instrument)

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        is_long   = float(trade["currentUnits"]) > 0
        bars_held = _bars_held_since(df, trade["openTime"])
        price     = df["Close"].iloc[-1]
        _log(
            f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']} "
            f"({bars_held} bars held), price={price:.5f} -- TP/SL govern exit"
        )
        return

    rsi_vals = rsi(df["Close"], 14)
    a        = atr(df["High"], df["Low"], df["Close"], 14)
    rsi_now  = rsi_vals.iloc[-1]
    av       = a.iloc[-1]
    price    = df["Close"].iloc[-1]

    if np.isnan(rsi_now) or np.isnan(av) or av <= 0:
        _log(f"{tag}: RSI/ATR not ready, skipping")
        return

    _log(f"{tag}: flat. price={price:.5f} rsi={rsi_now:.1f} atr={av:.5f}")

    stop_dist   = _RSI_D1_SL_ATR * av
    risk_amount = sleeve_equity * RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    if rsi_now < oversold:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_dist
        tp = price + tp_mult * stop_dist
        if not (sl < price < tp):
            _log(f"{tag}: buy geometry invalid (sl={sl:.5f} price={price:.5f} tp={tp:.5f})")
            return
        _log(f"{tag}: BUY (RSI {rsi_now:.1f} < {oversold}) -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)

    elif rsi_now > overbought:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_dist
        tp = price - tp_mult * stop_dist
        if not (tp < price < sl):
            _log(f"{tag}: sell geometry invalid (tp={tp:.5f} price={price:.5f} sl={sl:.5f})")
            return
        _log(f"{tag}: SELL (RSI {rsi_now:.1f} > {overbought}) -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)

    else:
        _log(f"{tag}: no signal (RSI {rsi_now:.1f} not extreme enough)")


def _journal_rsi_div_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    """Detect RSI divergence trades closed by OANDA TP/SL and journal them."""
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if RSI_DIV_CLOSED_IDS.exists():
        journaled = set(json.loads(RSI_DIV_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag:
            continue
        trade_id = t.get("id", "")
        if trade_id in journaled:
            continue
        realized = float(t.get("realizedPL", 0))
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(trade_id)
        RSI_DIV_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


def _detect_pivot_divergence(df: pd.DataFrame, rsi_vals: pd.Series, pivot_n: int, rsi_extreme: bool) -> str | None:
    """Check if today's closing bar newly confirms an RSI divergence signal.

    Replicates the backtesting framework exactly: a pivot is confirmed N bars
    after the candidate bar (i.e., candidate = bar at index -(pivot_n+1)).
    This means the signal fires once — on the first daily run after the pivot
    is confirmed — and is silent on all subsequent days unless a new pivot forms.

    Returns: 'bull', 'bear', or None.
    """
    lows  = df["Low"].values
    highs = df["High"].values
    rsi_a = rsi_vals.values
    n     = pivot_n
    total = len(lows)
    if total < 4 * n + 4:
        return None

    c = total - n - 1  # index of the newly-confirmed candidate bar

    # ── Bullish divergence check ──────────────────────────────────────────
    cl = lows[c]
    is_plow = all(lows[c + j] > cl for j in range(-n, n + 1) if j != 0)
    if is_plow:
        rsi_at_low = rsi_a[c]
        if not np.isnan(rsi_at_low):
            rsi_ok = (not rsi_extreme) or (rsi_at_low < 45)
            if rsi_ok:
                for i in range(c - n - 1, n - 1, -1):
                    prev_l = lows[i]
                    if all(lows[i + j] > prev_l for j in range(-n, n + 1) if j != 0):
                        prev_rsi = rsi_a[i]
                        if not np.isnan(prev_rsi) and cl < prev_l and rsi_at_low > prev_rsi:
                            return 'bull'
                        break

    # ── Bearish divergence check ──────────────────────────────────────────
    ch = highs[c]
    is_phigh = all(highs[c + j] < ch for j in range(-n, n + 1) if j != 0)
    if is_phigh:
        rsi_at_high = rsi_a[c]
        if not np.isnan(rsi_at_high):
            rsi_ok = (not rsi_extreme) or (rsi_at_high > 55)
            if rsi_ok:
                for i in range(c - n - 1, n - 1, -1):
                    prev_h = highs[i]
                    if all(highs[i + j] < prev_h for j in range(-n, n + 1) if j != 0):
                        prev_rsi = rsi_a[i]
                        if not np.isnan(prev_rsi) and ch > prev_h and rsi_at_high < prev_rsi:
                            return 'bear'
                        break

    return None


_RSI_DIV_SL_ATR = 1.5


def run_rsi_div_d1_sleeve(
    b: broker.OandaBroker,
    tag: str,
    instrument: str,
    sleeve_equity: float,
    pivot_n: int,
    rr: float,
    rsi_extreme: bool,
) -> None:
    """RSI divergence fade on D1 bars.

    Bull divergence: price makes a lower low while RSI makes a higher low → BUY.
    Bear divergence: price makes a higher high while RSI makes a lower high → SELL.
    Signal fires only on the day the divergence pivot is newly confirmed (N-bar window).
    SL: 1.5 ATR. TP: rr × SL distance.
    AUD/JPY Sh +1.26 WR 52.9% | NZD/USD Sh +0.78 WR 40.5% | FR40 Sh +0.85 WR 54.1%.
    """
    df = _load_candles(b, instrument)
    if len(df) < 4 * pivot_n + 4 + 15:
        _log(f"{tag}: not enough history ({len(df)} bars), skipping")
        return

    _journal_rsi_div_close_if_needed(b, tag, instrument)

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        is_long   = float(trade["currentUnits"]) > 0
        bars_held = _bars_held_since(df, trade["openTime"])
        price     = df["Close"].iloc[-1]
        _log(
            f"{tag}: in {'LONG' if is_long else 'SHORT'} since {trade['openTime']} "
            f"({bars_held} bars held), price={price:.5f} -- TP/SL govern exit"
        )
        return

    rsi_vals = rsi(df["Close"], 14)
    a        = atr(df["High"], df["Low"], df["Close"], 14)
    av       = a.iloc[-1]
    price    = df["Close"].iloc[-1]

    if np.isnan(av) or av <= 0:
        _log(f"{tag}: ATR not ready, skipping")
        return

    div = _detect_pivot_divergence(df, rsi_vals, pivot_n, rsi_extreme)
    _log(f"{tag}: flat. price={price:.5f} atr={av:.5f} divergence={div or 'none'}")

    if div is None:
        return

    stop_dist   = _RSI_DIV_SL_ATR * av
    risk_amount = sleeve_equity * RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    units       = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    if div == 'bull':
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_dist
        tp = price + rr * stop_dist
        if not (sl < price < tp):
            _log(f"{tag}: buy geometry invalid (sl={sl:.5f} price={price:.5f} tp={tp:.5f})")
            return
        _log(f"{tag}: BUY (RSI bull divergence) -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)

    else:  # bear
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_dist
        tp = price - rr * stop_dist
        if not (tp < price < sl):
            _log(f"{tag}: sell geometry invalid (tp={tp:.5f} price={price:.5f} sl={sl:.5f})")
            return
        _log(f"{tag}: SELL (RSI bear divergence) -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)


def _journal_fvg_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    """Detect FVG trades closed by OANDA TP/SL and journal them."""
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if FVG_CLOSED_IDS.exists():
        journaled = set(json.loads(FVG_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag:
            continue
        trade_id = t.get("id", "")
        if trade_id in journaled:
            continue
        realized = float(t.get("realizedPL", 0))
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(trade_id)
        FVG_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


_FVG_EMA_PERIOD  = 200
_FVG_MIN_GAP_PCT = 0.0015   # minimum gap size = 0.15% of price
_FVG_SL_BUFFER   = 0.0005   # 0.05% beyond FVG edge for SL
_FVG_RR          = 3.0
_FVG_EXPIRY_BARS = 8        # cancel stale FVG after 8 bars (4 hours)
_FVG_M30_COUNT   = 250      # bars to fetch; must be >= EMA period + expiry


def run_fvg_m30_sleeve(b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float) -> None:
    """Fair Value Gap fill strategy on M30 bars.

    FVG = 3-candle imbalance where price moved so fast it left a gap.
    Bullish FVG: bar[-3].high < bar[-1].low  → LONG when price retraces into gap.
    Bearish FVG: bar[-3].low  > bar[-1].high → SHORT when price retraces into gap.
    EMA(200) trend filter. Session open only: 07:00–09:00 or 13:00–15:00 UTC.
    SL: 0.05% beyond the FVG far edge. TP: 3:1 risk:reward.
    """
    _journal_fvg_close_if_needed(b, tag, instrument)

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        _log(f"{tag}: trade open since {trade['openTime']}, holding (FVG fixed TP/SL)")
        return

    now_hour = datetime.now(timezone.utc).hour
    in_session = (7 <= now_hour < 9) or (13 <= now_hour < 15)
    if not in_session:
        _log(f"{tag}: outside session window (07-09, 13-15 UTC), skipping entry")
        return

    raw = b.get_candles(instrument, granularity="M30", count=_FVG_M30_COUNT)
    bars = [c for c in raw if c["complete"]]
    if len(bars) < _FVG_EMA_PERIOD + _FVG_EXPIRY_BARS + 4:
        _log(f"{tag}: not enough M30 bars ({len(bars)}), skipping")
        return

    closes = [float(c["mid"]["c"]) for c in bars]
    highs  = [float(c["mid"]["h"]) for c in bars]
    lows   = [float(c["mid"]["l"]) for c in bars]

    ema_vals = pd.Series(closes).ewm(span=_FVG_EMA_PERIOD, adjust=False).mean()

    # Scan last _FVG_EXPIRY_BARS+2 bars for the most recent valid FVG.
    # Candle pattern: bars[i-2]=c1, bars[i-1]=c2, bars[i]=c3.
    # Entry is checked against bars[-1] (most recent completed bar).
    fvg_dir = fvg_top = fvg_bot = None
    search_start = len(bars) - _FVG_EXPIRY_BARS - 2
    for i in range(search_start, len(bars) - 1):
        h1 = highs[i - 2]
        l1 = lows[i - 2]
        h3 = highs[i]
        l3 = lows[i]
        price_at = closes[i]
        ema_at   = float(ema_vals.iloc[i])
        min_gap  = abs(price_at) * _FVG_MIN_GAP_PCT

        if h1 < l3 and (l3 - h1) >= min_gap and price_at > ema_at:
            fvg_dir = 'bull'; fvg_bot = h1; fvg_top = l3
        elif l1 > h3 and (l1 - h3) >= min_gap and price_at < ema_at:
            fvg_dir = 'bear'; fvg_bot = h3; fvg_top = l1

    if fvg_dir is None:
        _log(f"{tag}: no valid FVG in last {_FVG_EXPIRY_BARS} bars")
        return

    entry_high = highs[-1]
    entry_low  = lows[-1]
    price      = closes[-1]

    risk_amount = sleeve_equity * RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)

    _FVG_MAX_UNITS = 50_000   # NATGAS/FX liquidity cap — prevents INSUFFICIENT_LIQUIDITY errors

    if fvg_dir == 'bull' and entry_low <= fvg_top:
        sl      = fvg_bot * (1 - _FVG_SL_BUFFER)
        sl_dist = max(price - sl, 1e-9)
        units   = min(int(risk_amount / (sl_dist * quote_rate)) if quote_rate > 0 else 0, _FVG_MAX_UNITS)
        if units <= 0:
            _log(f"{tag}: computed size <= 0, skipping")
            return
        tp = price + _FVG_RR * sl_dist
        if not (sl < price < tp):
            _log(f"{tag}: FVG bull geometry invalid (sl={sl:.5f} price={price:.5f} tp={tp:.5f})")
            return
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        _log(f"{tag}: BUY FVG retrace -- gap [{fvg_bot:.5f},{fvg_top:.5f}] "
             f"{units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)

    elif fvg_dir == 'bear' and entry_high >= fvg_bot:
        sl      = fvg_top * (1 + _FVG_SL_BUFFER)
        sl_dist = max(sl - price, 1e-9)
        units   = min(int(risk_amount / (sl_dist * quote_rate)) if quote_rate > 0 else 0, _FVG_MAX_UNITS)
        if units <= 0:
            _log(f"{tag}: computed size <= 0, skipping")
            return
        tp = price - _FVG_RR * sl_dist
        if not (tp < price < sl):
            _log(f"{tag}: FVG bear geometry invalid (tp={tp:.5f} price={price:.5f} sl={sl:.5f})")
            return
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        _log(f"{tag}: SELL FVG retrace -- gap [{fvg_bot:.5f},{fvg_top:.5f}] "
             f"{units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)

    else:
        _log(f"{tag}: FVG {fvg_dir} detected gap=[{fvg_bot:.5f},{fvg_top:.5f}] "
             f"but price {price:.5f} not yet in zone")


def _journal_engulf_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if ENGULF_CLOSED_IDS.exists():
        journaled = set(json.loads(ENGULF_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag or t.get("id", "") in journaled:
            continue
        realized = float(t.get("realizedPL", 0))
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(t["id"])
        ENGULF_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


def run_engulf_d1_sleeve(
    b: broker.OandaBroker, tag: str, instrument: str, sleeve_equity: float, rr: float = 3.0
) -> None:
    """Bearish/Bullish engulfing candle — D1, 1.5 ATR SL.
    Bullish: prior bar bearish, yesterday's bar bullish and body engulfs prior body → BUY.
    Bearish: prior bar bullish, yesterday's bar bearish and body engulfs prior body → SELL.
    OOS Sh +1.70 on SPX500, 9/10 profitable years."""
    raw = b.get_candles(instrument, granularity="D", count=30)
    bars = [c for c in raw if c["complete"]]
    if len(bars) < 4:
        return

    df = pd.DataFrame({
        "Open":  [float(c["mid"]["o"]) for c in bars],
        "High":  [float(c["mid"]["h"]) for c in bars],
        "Low":   [float(c["mid"]["l"]) for c in bars],
        "Close": [float(c["mid"]["c"]) for c in bars],
    })
    av = atr(df["High"], df["Low"], df["Close"]).iloc[-1]
    price = df["Close"].iloc[-1]

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        _log(f"{tag}: in position since {trade['openTime']}, price={price:.5f}")
        return

    _journal_engulf_close_if_needed(b, tag, instrument)

    o1, c1 = df["Open"].iloc[-1], df["Close"].iloc[-1]   # yesterday (most recent complete)
    o2, c2 = df["Open"].iloc[-2], df["Close"].iloc[-2]   # day before
    body1, body2 = abs(c1 - o1), abs(c2 - o2)

    if body2 <= 0 or body1 < body2 * 1.1 or np.isnan(av) or av <= 0:
        _log(f"{tag}: flat. no engulfing pattern")
        return

    risk_amount = sleeve_equity * RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    stop_dist   = 1.5 * av
    units = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    # Bullish engulfing: prior bar bearish, yesterday bullish and engulfs
    if c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_dist
        tp = price + rr * stop_dist
        _log(f"{tag}: BUY engulfing -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)

    # Bearish engulfing: prior bar bullish, yesterday bearish and engulfs
    elif c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_dist
        tp = price - rr * stop_dist
        _log(f"{tag}: SELL engulfing -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)

    else:
        _log(f"{tag}: flat. no engulfing pattern")


def _journal_donchian_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if DONCHIAN_CLOSED_IDS.exists():
        journaled = set(json.loads(DONCHIAN_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag or t.get("id", "") in journaled:
            continue
        realized = float(t.get("realizedPL", 0))
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(t["id"])
        DONCHIAN_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


def run_donchian_d1_sleeve(
    b: broker.OandaBroker, tag: str, instrument: str,
    sleeve_equity: float, period: int = 20, rr: float = 3.0
) -> None:
    """Donchian channel breakout — D1. BUY when price breaks above N-day high,
    SELL when it breaks below N-day low. SL = 1.5 ATR, TP = rr × SL.
    XAU/USD period=20: OOS Sh +2.30, 8/10 profitable years."""
    raw = b.get_candles(instrument, granularity="D", count=period + 5)
    bars = [c for c in raw if c["complete"]]
    if len(bars) < period + 2:
        return

    df = pd.DataFrame({
        "High":  [float(c["mid"]["h"]) for c in bars],
        "Low":   [float(c["mid"]["l"]) for c in bars],
        "Close": [float(c["mid"]["c"]) for c in bars],
    })
    av    = atr(df["High"], df["Low"], df["Close"]).iloc[-1]
    price = df["Close"].iloc[-1]

    # Channel: prior N bars (exclude yesterday to avoid self-reference)
    chan_high = df["High"].iloc[-(period + 1):-1].max()
    chan_low  = df["Low"].iloc[-(period + 1):-1].min()

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        _log(f"{tag}: in position since {trade['openTime']}, price={price:.5f}")
        return

    _journal_donchian_close_if_needed(b, tag, instrument)

    _log(f"{tag}: flat. price={price:.5f} chan_high={chan_high:.5f} chan_low={chan_low:.5f}")

    if np.isnan(av) or av <= 0:
        return

    risk_amount = sleeve_equity * RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    stop_dist   = 1.5 * av
    units = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    if price > chan_high:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_dist
        tp = price + rr * stop_dist
        _log(f"{tag}: BUY Donchian breakout -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)

    elif price < chan_low:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_dist
        tp = price - rr * stop_dist
        _log(f"{tag}: SELL Donchian breakout -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)

    else:
        _log(f"{tag}: no breakout (price inside {period}-day channel)")


def _journal_vsf_close_if_needed(b: broker.OandaBroker, tag: str, instrument: str) -> None:
    try:
        closed = b.get_closed_trades(instrument, count=5)
    except Exception:
        return
    journaled: set = set()
    if VSF_CLOSED_IDS.exists():
        journaled = set(json.loads(VSF_CLOSED_IDS.read_text()).get("ids", []))
    for t in closed:
        trade_tag = (t.get("tradeClientExtensions") or {}).get("tag", "") or \
                    (t.get("clientExtensions") or {}).get("tag", "")
        if trade_tag != tag or t.get("id", "") in journaled:
            continue
        realized = float(t.get("realizedPL", 0))
        _log(f"{tag}: OANDA TP/SL close detected -- realized_pl={realized:+.2f}")
        journal.record_close(tag, instrument, "oanda_tp_sl", realized)
        journaled.add(t["id"])
        VSF_CLOSED_IDS.write_text(json.dumps({"ids": list(journaled)}))
        return


def run_vsf_d1_sleeve(
    b: broker.OandaBroker, tag: str, instrument: str,
    sleeve_equity: float, vol_mult: float = 2.0, rr: float = 3.0
) -> None:
    """Volatility Spike Fade — D1. Abnormally large candle (range > vol_mult×ATR)
    with close near the extreme → mean reversion entry next bar.
    Close in bottom 25% of range → BUY. Close in top 25% → SELL.
    SL = 1.5 ATR, TP = rr × SL. UK100: OOS Sh +1.26, 6/10 profitable years."""
    raw = b.get_candles(instrument, granularity="D", count=30)
    bars = [c for c in raw if c["complete"]]
    if len(bars) < 16:
        return

    df = pd.DataFrame({
        "High":  [float(c["mid"]["h"]) for c in bars],
        "Low":   [float(c["mid"]["l"]) for c in bars],
        "Close": [float(c["mid"]["c"]) for c in bars],
    })
    av    = atr(df["High"], df["Low"], df["Close"]).iloc[-1]
    price = df["Close"].iloc[-1]

    trade = _tagged_trade(b, instrument, tag)
    if trade is not None:
        _log(f"{tag}: in position since {trade['openTime']}, price={price:.5f}")
        return

    _journal_vsf_close_if_needed(b, tag, instrument)

    if np.isnan(av) or av <= 0:
        _log(f"{tag}: ATR not ready, skipping")
        return

    # Yesterday's candle (most recent completed bar)
    h1  = df["High"].iloc[-1]
    l1  = df["Low"].iloc[-1]
    c1  = df["Close"].iloc[-1]
    rng = h1 - l1

    if rng < vol_mult * av:
        _log(f"{tag}: flat. range={rng:.5f} < {vol_mult}×ATR={vol_mult*av:.5f} -- no vol spike")
        return

    pos_in_range = (c1 - l1) / rng
    _log(f"{tag}: vol spike! range={rng:.5f} vs {vol_mult}×ATR={vol_mult*av:.5f} "
         f"pos={pos_in_range:.2f} price={price:.5f}")

    risk_amount = sleeve_equity * RISK_PCT
    quote_rate  = _quote_usd_rate(b, instrument)
    stop_dist   = 1.5 * av
    units = int(risk_amount / (stop_dist * quote_rate)) if quote_rate > 0 else 0
    if units <= 0:
        _log(f"{tag}: computed size <= 0, skipping")
        return

    # Close in bottom 25% → spike down → fade with BUY
    if pos_in_range <= 0.25:
        if _opposite_direction_conflict(b, instrument, tag, "long"):
            return
        sl = price - stop_dist
        tp = price + rr * stop_dist
        if not (sl < price < tp):
            _log(f"{tag}: buy geometry invalid (sl={sl:.5f} price={price:.5f} tp={tp:.5f})")
            return
        _log(f"{tag}: BUY vol spike fade -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, units, "long", sl, tp)

    # Close in top 25% → spike up → fade with SELL
    elif pos_in_range >= 0.75:
        if _opposite_direction_conflict(b, instrument, tag, "short"):
            return
        sl = price + stop_dist
        tp = price - rr * stop_dist
        if not (tp < price < sl):
            _log(f"{tag}: sell geometry invalid (tp={tp:.5f} price={price:.5f} sl={sl:.5f})")
            return
        _log(f"{tag}: SELL vol spike fade -- {units} units sl={sl:.5f} tp={tp:.5f}")
        _open_and_journal(b, tag, instrument, -units, "short", sl, tp)

    else:
        _log(f"{tag}: vol spike but close not at extreme (pos={pos_in_range:.2f}) -- no signal")


def main() -> None:
    b = broker.from_env()
    account_equity = float(b.account_summary()["account"]["balance"])
    sleeve_equity = account_equity * ALLOCATION_FRACTION
    total_sleeves = len(SLEEVES) + len(M30_SLEEVES) + len(H1_SLEEVES) + len(MACD_H1_SLEEVES) + len(ORB_SLEEVES) + len(PDHL_SLEEVES) + len(CONSEC_D1_SLEEVES) + len(FVG_M30_SLEEVES) + len(RSI_D1_SLEEVES) + len(RSI_DIV_D1_SLEEVES) + len(ENGULF_D1_SLEEVES) + len(DONCHIAN_D1_SLEEVES) + len(VSF_D1_SLEEVES)
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
    for tag, instrument, streak in CONSEC_D1_SLEEVES:
        try:
            run_consec_d1_sleeve(b, tag, instrument, sleeve_equity, streak)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in FVG_M30_SLEEVES:
        try:
            run_fvg_m30_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument, oversold, overbought, tp_mult in RSI_D1_SLEEVES:
        try:
            run_rsi_d1_sleeve(b, tag, instrument, sleeve_equity, oversold, overbought, tp_mult)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument, pivot_n, rr, rsi_extreme in RSI_DIV_D1_SLEEVES:
        try:
            run_rsi_div_d1_sleeve(b, tag, instrument, sleeve_equity, pivot_n, rr, rsi_extreme)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument, rr in ENGULF_D1_SLEEVES:
        try:
            run_engulf_d1_sleeve(b, tag, instrument, sleeve_equity, rr)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument, period, rr in DONCHIAN_D1_SLEEVES:
        try:
            run_donchian_d1_sleeve(b, tag, instrument, sleeve_equity, period, rr)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument, vol_mult, rr in VSF_D1_SLEEVES:
        try:
            run_vsf_d1_sleeve(b, tag, instrument, sleeve_equity, vol_mult, rr)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")


if __name__ == "__main__":
    main()
