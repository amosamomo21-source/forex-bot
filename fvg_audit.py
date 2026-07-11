"""
Fair Value Gap (FVG) day trading strategy — backtest audit.

What is an FVG?
  A 3-candle pattern where price moves so fast it leaves a gap:
  Bullish FVG : candle[-3].high < candle[-1].low  → gap below current price → LONG on retrace
  Bearish FVG : candle[-3].low  > candle[-1].high → gap above current price → SHORT on retrace

Rules:
  Trend filter : EMA(50) — only take bullish FVGs when price > EMA, bearish when price < EMA
  Session      : London (07:00–12:00 UTC) + NY (13:00–17:00 UTC) only
  Entry        : when current bar's price enters the FVG zone
  SL           : just beyond the far edge of the FVG (below gap bottom for longs)
  TP           : 2× or 3× risk distance from entry
  FVG expiry   : cancelled after 10 bars if not triggered
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

def _ema(s, n):
    return pd.Series(np.array(s, float)).ewm(span=n, adjust=False).mean().values

class FVG(Strategy):
    ema_period  = 50
    rr          = 2.0
    fvg_expiry  = 10     # bars before an untriggered FVG is cancelled
    risk_pct    = RISK_PCT
    sl_buffer   = 0.0005  # 0.05% buffer beyond FVG edge for SL

    def init(self):
        self.trend    = self.I(_ema, self.data.Close, self.ema_period)
        self._dir     = None   # 'bull' or 'bear'
        self._top     = np.nan
        self._bot     = np.nan
        self._fvg_idx = -999

    def next(self):
        idx = len(self.data) - 1
        if idx < 3: return

        # Session filter: London + NY only
        t = self.data.index[-1]
        if not (7 <= t.hour < 12 or 13 <= t.hour < 17):
            return

        p  = float(self.data.Close[-1])
        tr = float(self.trend[-1])
        if np.isnan(tr): return

        # ── Detect FVG from last 3 completed bars ([-3], [-2], [-1]) ──────────
        # These are all closed before the current bar, so no look-ahead.
        h1 = float(self.data.High[-3])
        l1 = float(self.data.Low[-3])
        h3 = float(self.data.High[-1])
        l3 = float(self.data.Low[-1])

        gap_size_min = abs(p) * 0.0002  # min gap = 0.02% of price

        if not self.position:
            # Bullish FVG: gap between candle[-3].high and candle[-1].low
            if h1 < l3 and (l3 - h1) >= gap_size_min and p > tr:
                self._dir     = 'bull'
                self._bot     = h1    # bottom of gap (FVG support)
                self._top     = l3    # top of gap (FVG resistance from below)
                self._fvg_idx = idx

            # Bearish FVG: gap between candle[-3].low and candle[-1].high
            elif l1 > h3 and (l1 - h3) >= gap_size_min and p < tr:
                self._dir     = 'bear'
                self._bot     = h3    # bottom of gap (FVG support from above)
                self._top     = l1    # top of gap (FVG resistance)
                self._fvg_idx = idx

        # Expire stale FVGs
        if self._dir and (idx - self._fvg_idx) > self.fvg_expiry:
            self._dir = None

        # ── Entry: price retraces into the FVG zone ───────────────────────────
        if not self._dir or self.position:
            return

        bar_low  = float(self.data.Low[-1])
        bar_high = float(self.data.High[-1])
        ra       = self.equity * self.risk_pct

        if self._dir == 'bull' and bar_low <= self._top:
            # Retrace into bullish FVG → LONG
            sl      = self._bot * (1 - self.sl_buffer)
            sl_dist = max(p - sl, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 50_000))
            tp      = p + self.rr * sl_dist
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)
            self._dir = None

        elif self._dir == 'bear' and bar_high >= self._bot:
            # Retrace into bearish FVG → SHORT
            sl      = self._top * (1 + self.sl_buffer)
            sl_dist = max(sl - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 50_000))
            tp      = p - self.rr * sl_dist
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)
            self._dir = None

class FVG3R(FVG):
    rr = 3.0

# ── Run ──────────────────────────────────────────────────────────────────────
INSTRUMENTS = [
    ('XAU/USD', 'XAU_USD'),
    ('GBP/USD', 'GBP_USD'),
    ('EUR/USD', 'EUR_USD'),
    ('GBP/JPY', 'GBP_JPY'),
    ('NAS100',  'NAS100_USD'),
]
INTERVALS = [('M30', '30m'), ('H1', '1h')]

print('FVG Day Trade Audit — 10y, London+NY sessions, EMA(50) trend filter')
print('Entry on retrace into FVG zone | SL beyond FVG edge | tested at 2:1 and 3:1 R:R')
print()
print(f"{'Instrument':<10} {'TF':<5} {'RR':<5} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  EV/R   Verdict")
print('─' * 90)

results = []
for lbl, instr in INSTRUMENTS:
    for tf_name, tf in INTERVALS:
        try:
            df = load_oanda_data(instr, period='10y', interval=tf)
        except Exception as e:
            print(f'{lbl:<10} {tf_name:<5}  ERROR loading: {e}')
            continue

        for rr_name, cls in [('2:1', FVG), ('3:1', FVG3R)]:
            try:
                b  = Backtest(df, cls, cash=100_000, commission=0.0002,
                              margin=1/30, finalize_trades=True)
                s  = b.run()
                sh  = float(s.get('Sharpe Ratio', 0) or 0)
                ann = float(s.get('Return (Ann.) [%]', 0) or 0)
                dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
                wr  = float(s.get('Win Rate [%]', 0) or 0)
                n   = int(s.get('# Trades', 0) or 0)
                pf  = float(s.get('Profit Factor', 0) or 0)
                rr  = cls.rr
                be  = 100 / (rr + 1)
                ev  = (wr/100) * rr - (1 - wr/100)
                ok  = sh >= 0.5 and ann > 0 and n >= 30
                verdict = 'ADD' if ok else ('WATCH' if sh >= 0.2 and ann > 0 else 'PASS')
                print(f'{lbl:<10} {tf_name:<5} {rr_name:<5} {sh:>+6.2f} {ann:>+6.1f}% '
                      f'{dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}  {ev:>+.3f}R  {verdict}')
                results.append({'lbl': lbl, 'tf': tf_name, 'rr': rr_name,
                                'sh': sh, 'ann': ann, 'wr': wr, 'n': n, 'verdict': verdict})
            except Exception as e:
                print(f'{lbl:<10} {tf_name:<5} {rr_name:<5}  ERROR: {e}')
        print()

# Summary
adds = [r for r in results if r['verdict'] == 'ADD']
print('─' * 90)
if adds:
    print(f'Candidates (Sharpe ≥ 0.5, positive Ann%, N ≥ 30):')
    for r in adds:
        print(f"  {r['lbl']} {r['tf']} {r['rr']} — Sh {r['sh']:+.2f}, WR {r['wr']:.0f}%, N={r['n']}")
else:
    print('No instruments passed the ADD threshold.')
print(f'Break-even WR: 33% at 2:1 R:R, 25% at 3:1 R:R')
