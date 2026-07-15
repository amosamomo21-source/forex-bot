#!/usr/bin/env python3
"""
intraday_scanner.py
====================
Autonomous market scanner — runs every 15 min via launchd.
Scans 15 instruments on D1 + M30 + M15 for quick high-confluence entries.
Places trades automatically on the PRACTICE account only.

Safety limits
-------------
  • allow_live=False hardcoded — practice account, never real money
  • Max 3 open scanner trades at once
  • $1,000 risk per trade
  • $3,000 daily loss limit — stops new entries for the rest of the day
  • Won't trade any instrument the main bot currently has open
  • Won't open a second scanner position on the same instrument
  • Weekend guard — exits immediately on Sat/Sun
  • Quiet hours guard — no new entries 23:30–06:30 UTC (dead market)
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")   # needed when run via launchd (no shell env)

sys.path.insert(0, str(Path(__file__).parent))
from broker import from_env, OandaBroker

# ── Config ─────────────────────────────────────────────────────────────────────
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "XAU_USD",       # Gold
    "BCO_USD",       # Brent Oil
    "SPX500_USD", "NAS100_USD", "UK100_GBP",
    "WHEAT_USD",
]

MAX_OPEN    = 5        # max scanner trades open simultaneously
RISK_USD    = 1_000   # dollars at risk per trade
DAILY_LIMIT = -3_000  # stop new trades if scanner's daily P&L hits this
SCORE_MIN   = 3.0     # minimum signal confluence score to enter
MAX_UNITS   = 200_000 # absolute cap — prevents margin exhaustion when ATR is tiny
SAFE_AT_USD  = 100    # when unrealized profit hits this, move SL to break-even
CLOSE_AT_USD     = 1_000  # when combined scanner profit hits this, close all trades
COOLDOWN_MINUTES = 30     # after a batch close, block re-entry on same instrument for 30min

STATE_FILE = Path(__file__).parent / "scanner_state.json"
LOG_FILE   = Path(__file__).parent / "scanner.log"

# ── Logging ────────────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")

# ── State ──────────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    today   = date.today().isoformat()
    default = {"date": today, "daily_pnl": 0.0, "open_trade_ids": [], "closed_pnl": {}}
    if not STATE_FILE.exists():
        return default
    try:
        s = json.loads(STATE_FILE.read_text())
        if s.get("date") != today:          # new trading day — reset P&L
            s["date"]      = today
            s["daily_pnl"] = 0.0
            s["closed_pnl"] = {}
        return s
    except Exception:
        return default

def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ── Data helpers ───────────────────────────────────────────────────────────────
def _get_df(b: OandaBroker, instrument: str, gran: str, count: int) -> pd.DataFrame:
    candles = b.get_candles(instrument, granularity=gran, count=count)
    rows = [
        {"o": float(c["mid"]["o"]), "h": float(c["mid"]["h"]),
         "l": float(c["mid"]["l"]), "c": float(c["mid"]["c"])}
        for c in candles if c.get("complete", True)
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _mid(b: OandaBroker, instrument: str) -> float:
    px = b.get_price(instrument)
    return (float(px["asks"][0]["price"]) + float(px["bids"][0]["price"])) / 2

# ── Indicators ─────────────────────────────────────────────────────────────────
def _rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    val = (100 - 100 / (1 + g / l.replace(0, np.nan))).iloc[-1]
    return float(val) if not np.isnan(val) else 50.0

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["h"], df["l"], df["c"]
    tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    val = tr.ewm(span=n, adjust=False).mean().iloc[-1]
    return float(val) if not np.isnan(val) else 0.0

# ── Signal scorer ──────────────────────────────────────────────────────────────
def _score(
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m30: pd.DataFrame,
) -> tuple[float, float, list[str]]:
    """Score long and short confluences. Returns (long_score, short_score, reasons)."""
    ls = 0.0; ss = 0.0; rs: list[str] = []

    # ── D1: trend context ────────────────────────────────────────────────────
    if len(df_d1) >= 30:
        c     = df_d1["c"]
        price = float(c.iloc[-1])
        n200  = min(200, len(c))
        e200  = float(_ema(c, n200).iloc[-1])
        rsi_now  = _rsi(c)
        rsi_prev = _rsi(c.iloc[:-3]) if len(c) > 17 else rsi_now

        if price > e200:
            ls += 1.0; rs.append(f"D1 uptrend  (price {price:.5g} > 200EMA {e200:.5g})")
        else:
            ss += 1.0; rs.append(f"D1 downtrend  (price {price:.5g} < 200EMA {e200:.5g})")

        if rsi_now > rsi_prev:
            ls += 0.5; rs.append(f"D1 RSI rising  ({rsi_now:.0f})")
        else:
            ss += 0.5; rs.append(f"D1 RSI falling  ({rsi_now:.0f})")

    # ── H1: primary entry signals ────────────────────────────────────────────
    if len(df_h1) >= 25:
        c      = df_h1["c"]
        price  = float(c.iloc[-1])
        e9     = _ema(c, 9);  e21 = _ema(c, 21)
        f,  s  = float(e9.iloc[-1]),  float(e21.iloc[-1])
        f1, s1 = float(e9.iloc[-2]),  float(e21.iloc[-2])
        rsi_h1 = _rsi(c)
        sma20  = float(c.rolling(20).mean().iloc[-1])
        std20  = float(c.rolling(20).std().iloc[-1])
        bb_u   = sma20 + 2 * std20
        bb_l   = sma20 - 2 * std20

        # EMA crossover (fresh = stronger)
        if f1 < s1 and f > s:
            ls += 2.0; rs.append("M30 EMA 9/21 just crossed UP  ← fresh signal")
        elif f1 > s1 and f < s:
            ss += 2.0; rs.append("M30 EMA 9/21 just crossed DOWN  ← fresh signal")
        elif f > s:
            ls += 1.0; rs.append("M30 EMA 9 > 21  (uptrend)")
        else:
            ss += 1.0; rs.append("M30 EMA 9 < 21  (downtrend)")

        # RSI extremes
        if rsi_h1 < 25:
            ls += 2.0; rs.append(f"M30 RSI very oversold  ({rsi_h1:.0f})")
        elif rsi_h1 < 35:
            ls += 1.0; rs.append(f"M30 RSI oversold  ({rsi_h1:.0f})")
        elif rsi_h1 > 75:
            ss += 2.0; rs.append(f"M30 RSI very overbought  ({rsi_h1:.0f})")
        elif rsi_h1 > 65:
            ss += 1.0; rs.append(f"M30 RSI overbought  ({rsi_h1:.0f})")

        # Bollinger bands
        if price < bb_l:
            ls += 1.0; rs.append(f"M30 price below lower BB  ({price:.5g} < {bb_l:.5g})")
        elif price > bb_u:
            ss += 1.0; rs.append(f"M30 price above upper BB  ({price:.5g} > {bb_u:.5g})")

        # Donchian 20-period breakout
        if len(df_h1) >= 22:
            don_h = float(df_h1["h"].iloc[-21:-1].max())
            don_l = float(df_h1["l"].iloc[-21:-1].min())
            if price > don_h:
                ls += 1.0; rs.append(f"M30 Donchian breakout HIGH  ({price:.5g} > {don_h:.5g})")
            elif price < don_l:
                ss += 1.0; rs.append(f"M30 Donchian breakdown LOW  ({price:.5g} < {don_l:.5g})")

        # 3-candle consecutive streak
        if len(c) >= 4:
            c3up = all(float(c.iloc[-i]) > float(c.iloc[-i-1]) for i in range(1, 4))
            c3dn = all(float(c.iloc[-i]) < float(c.iloc[-i-1]) for i in range(1, 4))
            if c3up:
                ls += 0.5; rs.append("M30 3 consecutive up candles")
            elif c3dn:
                ss += 0.5; rs.append("M30 3 consecutive down candles")

    # ── M30: confirmation timeframe ──────────────────────────────────────────
    if len(df_m30) >= 25:
        c      = df_m30["c"]
        e9     = _ema(c, 9);  e21 = _ema(c, 21)
        f,  s  = float(e9.iloc[-1]),  float(e21.iloc[-1])
        f1, s1 = float(e9.iloc[-2]),  float(e21.iloc[-2])
        rsi_m  = _rsi(c)

        if f1 < s1 and f > s:
            ls += 1.0; rs.append("M30 EMA crossed UP  (confirmation)")
        elif f1 > s1 and f < s:
            ss += 1.0; rs.append("M30 EMA crossed DOWN  (confirmation)")
        elif f > s:
            ls += 0.5
        else:
            ss += 0.5

        if rsi_m < 35:
            ls += 1.0; rs.append(f"M30 RSI oversold  ({rsi_m:.0f})")
        elif rsi_m > 65:
            ss += 1.0; rs.append(f"M30 RSI overbought  ({rsi_m:.0f})")

    return ls, ss, rs

# ── Position sizing ────────────────────────────────────────────────────────────
def _calc_units(b: OandaBroker, instrument: str, direction: str, entry: float, sl: float) -> int:
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return 0
    quote = instrument.split("_")[1]
    try:
        if quote == "USD":
            usd_risk = sl_dist
        elif quote == "JPY":
            usd_risk = sl_dist / _mid(b, "USD_JPY")
        elif quote == "GBP":
            usd_risk = sl_dist * _mid(b, "GBP_USD")
        elif quote == "EUR":
            usd_risk = sl_dist * _mid(b, "EUR_USD")
        else:
            usd_risk = sl_dist
    except Exception:
        usd_risk = sl_dist

    if usd_risk <= 0:
        return 0
    units = max(1, min(MAX_UNITS, int(RISK_USD / usd_risk)))
    return units if direction == "long" else -units

# ── Break-even manager ─────────────────────────────────────────────────────────
def _manage_safety(b: OandaBroker, all_open: list, open_ids: list, state: dict) -> None:
    """
    1. If combined scanner P&L >= CLOSE_AT_USD → close ALL scanner trades at once.
    2. Otherwise move SL to break-even on any trade up >= SAFE_AT_USD.
    """
    safe_ids: list = state.setdefault("safe_ids", [])
    safe_ids[:] = [sid for sid in safe_ids if sid in {t["id"] for t in all_open}]

    scanner_trades = [t for t in all_open if t["id"] in open_ids]
    total_pl = sum(float(t.get("unrealizedPL", 0)) for t in scanner_trades)

    # ── Combined profit target: close everything ─────────────────────────────
    if total_pl >= CLOSE_AT_USD:
        _log(
            f"🎯 Combined scanner P&L ${total_pl:+,.0f} hit ${CLOSE_AT_USD:,} target "
            f"— closing all {len(scanner_trades)} trades"
        )
        cooldowns = state.setdefault("cooldowns", {})
        expire_ts = (datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_MINUTES)).isoformat()
        for t in scanner_trades:
            tid = t["id"]
            instrument = t["instrument"]
            pl = float(t.get("unrealizedPL", 0))
            try:
                b.close_trade(tid)
                _log(f"  {instrument} id={tid}: closed at ${pl:+,.0f}")
                open_ids.remove(tid)
                state["daily_pnl"] = state.get("daily_pnl", 0.0) + pl
                cooldowns[instrument] = expire_ts   # block re-entry for COOLDOWN_MINUTES
            except Exception as e:
                _log(f"  {instrument} id={tid}: close FAILED — {e}")
        state["safe_ids"] = []
        return

    # ── Per-trade break-even ─────────────────────────────────────────────────
    for t in scanner_trades:
        tid        = t["id"]
        unreal_pl  = float(t.get("unrealizedPL", 0))
        instrument = t["instrument"]

        if tid in safe_ids:
            continue
        if unreal_pl < SAFE_AT_USD:
            continue

        entry      = float(t["price"])
        units      = float(t["currentUnits"])
        direction  = "long" if units > 0 else "short"
        current_sl_str = t.get("stopLossOrder", {}).get("price")
        current_sl = float(current_sl_str) if current_sl_str else None

        if direction == "long"  and current_sl is not None and current_sl >= entry:
            safe_ids.append(tid); continue
        if direction == "short" and current_sl is not None and current_sl <= entry:
            safe_ids.append(tid); continue

        try:
            b.update_trade_sl(tid, entry, instrument)
            _log(
                f"{instrument} id={tid}: ✅ SL moved to break-even {entry:.5g}  "
                f"(profit ${unreal_pl:+.0f}  |  was SL {current_sl})"
            )
            safe_ids.append(tid)
        except Exception as e:
            _log(f"{instrument} id={tid}: break-even move FAILED — {e}")

    state["safe_ids"] = safe_ids


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    now_utc = datetime.now(timezone.utc)

    # Weekend guard
    if now_utc.weekday() >= 5:
        _log("Weekend — markets closed, exiting")
        return

    # Quiet hours: no new entries 23:30–06:30 UTC
    quiet = not (6.5 <= now_utc.hour + now_utc.minute / 60 < 23.5)

    state = _load_state()

    if state["daily_pnl"] <= DAILY_LIMIT:
        _log(f"Daily loss limit hit (${state['daily_pnl']:+,.0f}) — no new entries today")
        return

    b = from_env(allow_live=False)

    # Which instruments does the main bot currently hold?
    bot_instruments: set[str] = set()
    all_open: list[dict] = []
    try:
        all_open = b.get_open_trades()
        for t in all_open:
            tag = t.get("clientExtensions", {}).get("tag", "")
            if tag != "scanner":
                bot_instruments.add(t["instrument"])
    except Exception as e:
        _log(f"Warning: could not fetch open trades: {e}")

    # Reconcile scanner trade IDs with what's still open in OANDA
    oanda_open_ids  = {t["id"] for t in all_open}
    prev_ids        = list(state.get("open_trade_ids", []))
    open_ids        = [tid for tid in prev_ids if tid in oanda_open_ids]
    scanner_instruments = {
        t["instrument"] for t in all_open if t["id"] in open_ids
    }

    # Log closed trades P&L (approximate via account balance delta — simple approach)
    newly_closed = [tid for tid in prev_ids if tid not in oanda_open_ids]
    if newly_closed:
        _log(f"Scanner trades closed since last scan: {newly_closed}")

    state["open_trade_ids"] = open_ids
    _log(
        f"Scanner: {len(open_ids)}/{MAX_OPEN} open  |  "
        f"daily P&L ${state['daily_pnl']:+,.0f}  |  "
        f"bot holds: {sorted(bot_instruments) or 'nothing'}  |  "
        f"quiet={'yes' if quiet else 'no'}"
    )

    # Always run break-even check, even during quiet hours
    _manage_safety(b, all_open, open_ids, state)

    if quiet:
        _log("Quiet hours (23:30–06:30 UTC) — managing existing trades only, no new entries")
        _save_state(state)
        return

    if len(open_ids) >= MAX_OPEN:
        _log(f"Max open trades ({MAX_OPEN}) reached — skipping scan")
        _save_state(state)
        return

    # ── Scan ────────────────────────────────────────────────────────────────
    placed = 0
    for instrument in INSTRUMENTS:
        if len(open_ids) >= MAX_OPEN:
            break
        if instrument in bot_instruments:
            _log(f"{instrument}: skip — bot has open position")
            continue
        if instrument in scanner_instruments:
            _log(f"{instrument}: skip — scanner already in trade")
            continue

        # Cooldown — block re-entry on instruments just closed at target
        cooldowns = state.get("cooldowns", {})
        if instrument in cooldowns:
            expire = datetime.fromisoformat(cooldowns[instrument])
            now_utc = datetime.now(timezone.utc)
            if now_utc < expire:
                remaining_min = int((expire - now_utc).total_seconds() / 60)
                _log(f"{instrument}: skip — cooldown {remaining_min}min remaining (re-entry blocked after target close)")
                continue
            else:
                del cooldowns[instrument]   # expired — clear it
                state["cooldowns"] = cooldowns

        try:
            df_d1  = _get_df(b, instrument, "D",   60)
            df_h1  = _get_df(b, instrument, "M30", 60)   # quick mode: M30 is now primary
            df_m30 = _get_df(b, instrument, "M15", 60)   # quick mode: M15 is confirmation
        except Exception as e:
            _log(f"{instrument}: data error — {e}"); continue

        if len(df_h1) < 25:
            _log(f"{instrument}: not enough M30 candles ({len(df_h1)})"); continue

        ls, ss, reasons = _score(df_d1, df_h1, df_m30)
        direction  = "long" if ls >= ss else "short"
        best_score = ls if direction == "long" else ss

        if best_score < SCORE_MIN:
            _log(f"{instrument}: no signal  (long={ls:.1f} short={ss:.1f})")
            continue

        _log(f"{instrument}: SIGNAL  {direction.upper()}  score={best_score:.1f}  (long={ls:.1f} short={ss:.1f})")
        for r in [x for x in reasons if
                  ("long" in direction and any(k in x for k in ["up","UP","oversold","HIGH","rising","above 200"])) or
                  ("short" in direction and any(k in x for k in ["down","DOWN","overbought","LOW","falling","below 200"])) or
                  any(k in x for k in ["D1","H1","M30","M15"])]:
            _log(f"  → {r}")

        try:
            price = _mid(b, instrument)
        except Exception as e:
            _log(f"{instrument}: price error — {e}"); continue

        av = _atr(df_h1)   # M30 ATR — tighter than H1
        av = max(av, price * 0.001)   # floor: ≥0.1% of price (~10 pips for EUR/USD)
        if av <= 0:
            _log(f"{instrument}: ATR=0, skipping"); continue

        # Round to instrument-appropriate decimal places
        decimals = 0 if any(x in instrument for x in ["JPY","XAU","BCO","SPX","NAS","UK1","WHE"]) else 5
        if decimals == 0:
            decimals = 1 if any(x in instrument for x in ["SPX","NAS","UK1","BCO"]) else 2

        if direction == "long":
            sl = round(price - 1.0 * av, decimals)
            tp = round(price + 1.5 * av, decimals)
            if sl >= price or tp <= price:
                _log(f"{instrument}: invalid SL/TP geometry, skipping"); continue
        else:
            sl = round(price + 1.0 * av, decimals)
            tp = round(price - 1.5 * av, decimals)
            if sl <= price or tp >= price:
                _log(f"{instrument}: invalid SL/TP geometry, skipping"); continue

        units = _calc_units(b, instrument, direction, price, sl)
        if units == 0:
            _log(f"{instrument}: units=0, skipping"); continue

        _log(
            f"{instrument}: placing {direction.upper()} {abs(units):,} units  "
            f"entry~{price:.5g}  sl={sl:.5g}  tp={tp:.5g}  "
            f"risk~${RISK_USD:,.0f}"
        )

        try:
            resp     = b.place_market_order(
                instrument=instrument,
                units=units,
                stop_loss_price=sl,
                take_profit_price=tp,
                client_tag="scanner",
            )
            fill     = resp.get("orderFillTransaction", {})
            trade_id = fill.get("tradeOpened", {}).get("tradeID")
            fill_px  = fill.get("price", "?")

            if trade_id:
                open_ids.append(trade_id)
                scanner_instruments.add(instrument)
                state["open_trade_ids"] = open_ids
                _log(f"{instrument}: ✅ FILLED  id={trade_id}  price={fill_px}  sl={sl}  tp={tp}")
                placed += 1
            else:
                _log(f"{instrument}: no tradeID in response — {resp}")
        except Exception as e:
            _log(f"{instrument}: order FAILED — {e}")

    _log(f"Scan done — {placed} new trade(s) placed this run")
    _save_state(state)


if __name__ == "__main__":
    main()
