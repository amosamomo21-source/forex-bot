"""
Chart Champions – mechanical backtest.

Original rules (discretionary):
  1. Mark supply/demand zones (S/R as zones, not lines)
  2. Trade only in direction of market structure (HH/HL = bull, LH/LL = bear)
  3. Never enter on first touch of a level
  4. Enter on confirmation: rejection / reclaim after reaction at zone
  5. SL beyond zone invalidation point  |  TP = next opposing zone

Mechanical approximation:
  Structure  : pivot highs/lows (5-bar each side). HH+HL = bullish, LH+LL = bearish.
  Zone       : ±0.15% band around each confirmed swing (total 0.3% zone width).
  Touch rule : zone must have been visited at least once before (2nd+ approach only).
  Confirmation: bar that re-enters the zone closes back above zone bottom (demand)
               or below zone top (supply).
  SL         : 1 ATR beyond zone edge.
  TP         : 2:1 R:R (substitutes discretionary "next zone" target).
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

def _atr(high, low, close, n=14):
    h = pd.Series(np.array(high, float))
    l = pd.Series(np.array(low, float))
    c = pd.Series(np.array(close, float))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values


class ChartChampions(Strategy):
    pivot_n   = 5       # bars each side to confirm a swing
    zone_pct  = 0.003   # total zone width as fraction of price (0.3%)
    rr        = 2.0
    risk_pct  = RISK_PCT

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._sh = []    # confirmed swing high prices (chronological)
        self._sl = []    # confirmed swing low prices
        self._zones = [] # list of zone dicts

    # ── Pivot detection ───────────────────────────────────────────────────────
    def _is_pivot_high(self):
        n = self.pivot_n
        if len(self.data) < 2 * n + 2: return False
        c = -(n + 1)
        ch = float(self.data.High[c])
        return all(float(self.data.High[c + j]) < ch
                   for j in range(-n, n + 1) if j != 0)

    def _is_pivot_low(self):
        n = self.pivot_n
        if len(self.data) < 2 * n + 2: return False
        c = -(n + 1)
        cl = float(self.data.Low[c])
        return all(float(self.data.Low[c + j]) > cl
                   for j in range(-n, n + 1) if j != 0)

    # ── Structure ─────────────────────────────────────────────────────────────
    def _structure(self):
        if len(self._sh) < 2 or len(self._sl) < 2: return 0
        bull = self._sh[-1] > self._sh[-2] and self._sl[-1] > self._sl[-2]
        bear = self._sh[-1] < self._sh[-2] and self._sl[-1] < self._sl[-2]
        return 1 if bull else (-1 if bear else 0)

    # ── Main ──────────────────────────────────────────────────────────────────
    def next(self):
        idx = len(self.data) - 1
        p   = float(self.data.Close[-1])
        h   = float(self.data.High[-1])
        l   = float(self.data.Low[-1])
        av  = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        half = self.zone_pct / 2

        # ── 1. Detect new pivots (confirmed n+1 bars ago) ─────────────────
        if self._is_pivot_high():
            ph = float(self.data.High[-(self.pivot_n + 1)])
            self._sh.append(ph)
            if len(self._sh) > 12: self._sh.pop(0)
            self._zones.append({
                'dir': 'supply', 'price': ph,
                'top': ph * (1 + half), 'bot': ph * (1 - half),
                'visits': 0, 'last_bar': -999, 'active': True,
            })

        if self._is_pivot_low():
            pl = float(self.data.Low[-(self.pivot_n + 1)])
            self._sl.append(pl)
            if len(self._sl) > 12: self._sl.pop(0)
            self._zones.append({
                'dir': 'demand', 'price': pl,
                'top': pl * (1 + half), 'bot': pl * (1 - half),
                'visits': 0, 'last_bar': -999, 'active': True,
            })

        # Prune: keep only last 20 active zones
        self._zones = [z for z in self._zones if z['active']][-20:]

        # ── 2. Update zone visits and invalidations ───────────────────────
        for z in self._zones:
            in_zone = l <= z['top'] and h >= z['bot']
            if in_zone and (idx - z['last_bar']) > self.pivot_n:
                z['visits']   += 1
                z['last_bar']  = idx
            # Invalidate if price closes through the zone
            if z['dir'] == 'demand' and p < z['bot']:
                z['active'] = False
            elif z['dir'] == 'supply' and p > z['top']:
                z['active'] = False

        if self.position: return

        # ── 3. Entry signals ──────────────────────────────────────────────
        struct = self._structure()
        ra     = self.equity * self.risk_pct

        for z in self._zones:
            if not z['active'] or z['visits'] < 2: continue  # 2nd+ touch only

            if z['dir'] == 'demand' and struct == 1:
                # Bullish structure, price reacts at demand zone
                # Confirmation: bar touched the zone AND closed back above zone bottom
                if l <= z['top'] and p > z['bot']:
                    sl      = z['bot'] - av
                    sl_dist = max(p - sl, 1e-9)
                    units   = max(1, min(int(ra / sl_dist), 100_000))
                    tp      = p + self.rr * sl_dist
                    if sl < p < tp:
                        self.buy(size=units, sl=sl, tp=tp)
                    break

            elif z['dir'] == 'supply' and struct == -1:
                # Bearish structure, price reacts at supply zone
                if h >= z['bot'] and p < z['top']:
                    sl      = z['top'] + av
                    sl_dist = max(sl - p, 1e-9)
                    units   = max(1, min(int(ra / sl_dist), 100_000))
                    tp      = p - self.rr * sl_dist
                    if tp < p < sl:
                        self.sell(size=units, sl=sl, tp=tp)
                    break


class CC3R(ChartChampions):
    rr = 3.0


# ── Run ───────────────────────────────────────────────────────────────────────
INSTRUMENTS = [
    ('EUR/USD', 'EUR_USD'),
    ('GBP/USD', 'GBP_USD'),
    ('GBP/JPY', 'GBP_JPY'),
    ('XAU/USD', 'XAU_USD'),
    ('NAS100',  'NAS100_USD'),
]
TIMEFRAMES = [('H1', '1h'), ('M30', '30m')]

print('Chart Champions — mechanical backtest, 10y')
print('Pivot zones (5-bar) | structure filter | 2nd+ touch | 1-ATR SL | 2:1 and 3:1 R:R')
print()
print(f"{'Instrument':<10} {'TF':<5} {'RR':<5} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  EV/R   Verdict")
print('─' * 88)

results = []
for lbl, instr in INSTRUMENTS:
    for tf_name, tf in TIMEFRAMES:
        try:
            df = load_oanda_data(instr, period='10y', interval=tf)
        except Exception as e:
            print(f'{lbl:<10} {tf_name:<5}  load error: {e}')
            continue
        for rr_lbl, cls in [('2:1', ChartChampions), ('3:1', CC3R)]:
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
                ev  = (wr/100) * rr - (1 - wr/100)
                ok  = sh >= 0.4 and ann > 0 and n >= 20
                verdict = 'ADD' if ok else ('WATCH' if sh >= 0.1 and ann > 0 else 'PASS')
                print(f'{lbl:<10} {tf_name:<5} {rr_lbl:<5} {sh:>+6.2f} {ann:>+6.1f}% '
                      f'{dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}  {ev:>+.3f}R  {verdict}')
                results.append({'lbl': lbl, 'tf': tf_name, 'rr': rr_lbl,
                                'sh': sh, 'ann': ann, 'wr': wr, 'n': n, 'verdict': verdict})
            except Exception as e:
                print(f'{lbl:<10} {tf_name:<5} {rr_lbl:<5}  error: {e}')
        print()

print('─' * 88)
adds = [r for r in results if r['verdict'] == 'ADD']
watch = [r for r in results if r['verdict'] == 'WATCH']
print(f'ADD   (Sh ≥ 0.4, N ≥ 20): {len(adds)}')
print(f'WATCH (Sh ≥ 0.1):         {len(watch)}')
if adds:
    print('Candidates:')
    for r in sorted(adds, key=lambda x: x['sh'], reverse=True):
        print(f"  {r['lbl']} {r['tf']} {r['rr']} — Sh {r['sh']:+.2f}, Ann {r['ann']:+.1f}%, WR {r['wr']:.0f}%, N={r['n']}")
