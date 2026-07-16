#!/usr/bin/env python3
"""
scanner_backtest.py
====================
Backtests the intraday scanner signal logic on 5 months of M30 data
for all 15 instruments. Reports win rate, Sharpe, and quality verdict.

Signals replicated from intraday_scanner.py:
  - M30 EMA 9/21 crossover  (+2 fresh, +1 trend)
  - M30 RSI extremes         (+2 very, +1 moderate)
  - M30 Bollinger Band touch (+1)
  - M30 Donchian breakout   (+1)
  - D1 trend via 500-bar EMA on M30 (+1)
  - D1 RSI slope             (+0.5)
SL = 1×ATR (min 0.1% of price), TP = 1.5×ATR  →  1.5:1 R:R
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))
from broker import from_env
from backtesting import Backtest, Strategy

INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "XAU_USD", "BCO_USD",
    "SPX500_USD", "NAS100_USD", "UK100_GBP", "WHEAT_USD",
]

SCORE_MIN = 4.0
SL_MULT   = 1.0
TP_MULT   = 1.5
ATR_FLOOR = 0.001   # 0.1% of price — same floor as live scanner

# ── Indicator helpers (operate on numpy arrays) ────────────────────────────────
def _ema(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values

def _rsi(arr, n=14):
    s  = pd.Series(arr)
    d  = s.diff()
    g  = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    lx = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    rs = g / lx.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

def _atr(high, low, close, n=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values

def _bb_upper(arr, n=20):
    s = pd.Series(arr)
    return (s.rolling(n).mean() + 2 * s.rolling(n).std()).values

def _bb_lower(arr, n=20):
    s = pd.Series(arr)
    return (s.rolling(n).mean() - 2 * s.rolling(n).std()).values

def _donchian_high(high, n=20):
    return pd.Series(high).shift(1).rolling(n).max().values

def _donchian_low(low, n=20):
    return pd.Series(low).shift(1).rolling(n).min().values

# ── Strategy ───────────────────────────────────────────────────────────────────
class ScannerStrategy(Strategy):
    score_min = SCORE_MIN

    def init(self):
        c = self.data.Close
        h = self.data.High
        l = self.data.Low
        self.ema9       = self.I(_ema, c, 9,   name='EMA9')
        self.ema21      = self.I(_ema, c, 21,  name='EMA21')
        self.ema_trend  = self.I(_ema, c, 500, name='EMA500')  # D1 trend proxy
        self.rsi        = self.I(_rsi, c,      name='RSI')
        self.atr        = self.I(_atr, h, l, c, name='ATR')
        self.bb_up      = self.I(_bb_upper, c, name='BB_UP')
        self.bb_lo      = self.I(_bb_lower, c, name='BB_LO')
        self.don_hi     = self.I(_donchian_high, h, name='DON_HI')
        self.don_lo     = self.I(_donchian_low,  l, name='DON_LO')

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        if price <= 0 or np.isnan(price):
            return

        ls = 0.0; ss = 0.0

        # ── D1 trend (500-bar EMA proxy) ────────────────────────────────────
        trend_ema = self.ema_trend[-1]
        if not np.isnan(trend_ema):
            if price > trend_ema: ls += 1.0
            else:                 ss += 1.0

        # ── D1 RSI slope ────────────────────────────────────────────────────
        rsi_now  = self.rsi[-1]
        rsi_prev = self.rsi[-10] if len(self.rsi) > 10 else rsi_now
        if not (np.isnan(rsi_now) or np.isnan(rsi_prev)):
            if rsi_now > rsi_prev: ls += 0.5
            else:                  ss += 0.5

        # ── M30 EMA 9/21 cross ───────────────────────────────────────────────
        e9  = self.ema9[-1];  e21  = self.ema21[-1]
        e9p = self.ema9[-2];  e21p = self.ema21[-2]
        if not any(np.isnan(v) for v in [e9, e21, e9p, e21p]):
            if   e9p < e21p and e9 > e21: ls += 2.0
            elif e9p > e21p and e9 < e21: ss += 2.0
            elif e9 > e21:                ls += 1.0
            else:                         ss += 1.0

        # ── M30 RSI extremes ────────────────────────────────────────────────
        rsi = self.rsi[-1]
        if not np.isnan(rsi):
            if   rsi < 25: ls += 2.0
            elif rsi < 35: ls += 1.0
            elif rsi > 75: ss += 2.0
            elif rsi > 65: ss += 1.0

        # ── Bollinger bands ─────────────────────────────────────────────────
        bb_u = self.bb_up[-1]; bb_l = self.bb_lo[-1]
        if not (np.isnan(bb_u) or np.isnan(bb_l)):
            if   price < bb_l: ls += 1.0
            elif price > bb_u: ss += 1.0

        # ── Donchian breakout ───────────────────────────────────────────────
        don_h = self.don_hi[-1]; don_l = self.don_lo[-1]
        if not (np.isnan(don_h) or np.isnan(don_l)):
            if   price > don_h: ls += 1.0
            elif price < don_l: ss += 1.0

        direction = "long" if ls >= ss else "short"
        score     = ls if direction == "long" else ss

        if score < self.score_min:
            return

        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return
        atr = max(atr, price * ATR_FLOOR)

        if direction == "long":
            sl = price - SL_MULT * atr
            tp = price + TP_MULT * atr
            if sl < price < tp:
                self.buy(sl=sl, tp=tp)
        else:
            sl = price + SL_MULT * atr
            tp = price - TP_MULT * atr
            if tp < price < sl:
                self.sell(sl=sl, tp=tp)


# ── Data fetcher ───────────────────────────────────────────────────────────────
def fetch_df(b, instrument):
    candles = b.get_candles(instrument, granularity='M30', count=5000)
    rows = []
    for c in candles:
        if not c.get('complete', True):
            continue
        rows.append({
            'Open':  float(c['mid']['o']),
            'High':  float(c['mid']['h']),
            'Low':   float(c['mid']['l']),
            'Close': float(c['mid']['c']),
            'Volume': 1,
        })
        # backtesting.py needs a DatetimeIndex
    # Build a fake index (M30 spacing) — we only need it for ordering
    idx = pd.date_range(end='2026-07-16', periods=len(rows), freq='30min')
    df = pd.DataFrame(rows, index=idx)
    return df


# ── Run ────────────────────────────────────────────────────────────────────────
def verdict(sharpe):
    if sharpe >= 0.4:  return 'KEEP   ✅'
    if sharpe >= 0.1:  return 'WATCH  ⚠️'
    return                     'REVIEW ❌'

def main():
    b = from_env(allow_live=False)
    results = []

    print(f"\n{'Instrument':<14} {'Trades':>7} {'Win%':>6} {'Net%':>7} {'Sharpe':>7} {'MaxDD%':>7}  Verdict")
    print('─' * 72)

    for instrument in INSTRUMENTS:
        try:
            df = fetch_df(b, instrument)
            if len(df) < 200:
                print(f'{instrument:<14}  not enough data'); continue

            bt = Backtest(df, ScannerStrategy,
                          cash=100_000,
                          commission=0.00005,   # ~0.5 pip spread equivalent
                          exclusive_orders=True)
            stats = bt.run()

            trades  = int(stats['# Trades'])
            win_pct = float(stats['Win Rate [%]']) if trades > 0 else 0
            net_pct = float(stats['Return [%]'])
            sharpe  = float(stats['Sharpe Ratio']) if not np.isnan(float(stats['Sharpe Ratio'])) else 0
            max_dd  = float(stats['Max. Drawdown [%]'])

            results.append({
                'instrument': instrument,
                'trades': trades,
                'win_pct': win_pct,
                'net_pct': net_pct,
                'sharpe': sharpe,
                'max_dd': max_dd,
            })

            print(f'{instrument:<14} {trades:>7} {win_pct:>5.1f}% {net_pct:>+6.1f}% {sharpe:>7.2f} {max_dd:>6.1f}%  {verdict(sharpe)}')

        except Exception as e:
            print(f'{instrument:<14}  ERROR: {e}')

    # Summary
    if results:
        keeps  = [r for r in results if r['sharpe'] >= 0.4]
        watch  = [r for r in results if 0.1 <= r['sharpe'] < 0.4]
        review = [r for r in results if r['sharpe'] < 0.1]
        print('\n─' * 72)
        print(f"KEEP   ({len(keeps)}): {', '.join(r['instrument'] for r in keeps)}")
        print(f"WATCH  ({len(watch)}): {', '.join(r['instrument'] for r in watch)}")
        print(f"REVIEW ({len(review)}): {', '.join(r['instrument'] for r in review)}")

if __name__ == '__main__':
    main()
