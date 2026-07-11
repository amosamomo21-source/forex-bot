"""
D1 Strategy Audit — new short-signal strategies, all 34 instruments.

Strategies tested:
  1. Inside Bar Breakout  — bar inside prev range, enter on next break
  2. Donchian 20-day      — price breaks 20-day high/low
  3. Pin Bar              — long-wick rejection candle reversal
  4. Engulfing            — today's candle swallows yesterday's (reversal)

All strategies tested with 2:1 and 3:1 R:R.
SL = 1.5 ATR from entry.
D1, 10-year backtest.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

def _atr(high, low, close, n=14):
    h  = pd.Series(np.array(high, float))
    l  = pd.Series(np.array(low, float))
    c  = pd.Series(np.array(close, float))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values


# ── 1. Inside Bar Breakout ────────────────────────────────────────────────────
class InsideBar(Strategy):
    rr = 2.0

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._ib_high = None
        self._ib_low  = None

    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        h0, l0 = float(self.data.High[-1]), float(self.data.Low[-1])   # today
        h1, l1 = float(self.data.High[-2]), float(self.data.Low[-2])   # yesterday
        h2, l2 = float(self.data.High[-3]), float(self.data.Low[-3])   # day before

        # Detect inside bar (yesterday inside day before)
        if h1 < h2 and l1 > l2:
            self._ib_high = h1
            self._ib_low  = l1

        if self.position or self._ib_high is None:
            return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        # Breakout above inside bar high → BUY
        if h0 > self._ib_high:
            sl      = p - 1.5 * av
            sl_dist = max(p - sl, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p + self.rr * sl_dist
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)
            self._ib_high = self._ib_low = None

        # Breakout below inside bar low → SELL
        elif l0 < self._ib_low:
            sl      = p + 1.5 * av
            sl_dist = max(sl - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p - self.rr * sl_dist
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)
            self._ib_high = self._ib_low = None

class InsideBar_R2(InsideBar): rr = 2.0
class InsideBar_R3(InsideBar): rr = 3.0


# ── 2. Donchian 20-day Breakout ───────────────────────────────────────────────
class Donchian(Strategy):
    period = 20
    rr     = 2.0

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        n = self.period
        if len(self.data) < n + 2: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return
        if self.position: return

        # Use prior n bars (exclude today) for the channel
        highs = [float(self.data.High[-(i+2)]) for i in range(n)]
        lows  = [float(self.data.Low[-(i+2)]) for i in range(n)]
        chan_high = max(highs)
        chan_low  = min(lows)

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if p > chan_high:
            sl      = p - 1.5 * av
            sl_dist = max(p - sl, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p + self.rr * sl_dist
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)

        elif p < chan_low:
            sl      = p + 1.5 * av
            sl_dist = max(sl - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p - self.rr * sl_dist
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)

class Donchian_R2(Donchian): period=20; rr=2.0
class Donchian_R3(Donchian): period=20; rr=3.0


# ── 3. Pin Bar (Rejection Candle) ─────────────────────────────────────────────
class PinBar(Strategy):
    rr         = 2.0
    wick_ratio = 2.0   # wick must be ≥ 2× the body

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._want_long  = False
        self._want_short = False

    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        # Pin bar = yesterday's candle
        o = float(self.data.Open[-2])
        h = float(self.data.High[-2])
        l = float(self.data.Low[-2])
        c = float(self.data.Close[-2])

        body      = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_rng  = h - l
        if total_rng <= 0: return

        # Bullish pin bar: long lower wick, close in upper 1/3
        if (lower_wick >= self.wick_ratio * max(body, av * 0.1) and
                (c - l) / total_rng > 0.6):
            self._want_long = True

        # Bearish pin bar: long upper wick, close in lower 1/3
        if (upper_wick >= self.wick_ratio * max(body, av * 0.1) and
                (h - c) / total_rng > 0.6):
            self._want_short = True

        if self.position:
            self._want_long = self._want_short = False
            return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._want_long:
            sl      = p - 1.5 * av
            sl_dist = max(p - sl, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p + self.rr * sl_dist
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)
            self._want_long = False

        elif self._want_short:
            sl      = p + 1.5 * av
            sl_dist = max(sl - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p - self.rr * sl_dist
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)
            self._want_short = False

class PinBar_R2(PinBar): rr=2.0
class PinBar_R3(PinBar): rr=3.0


# ── 4. Engulfing Candle ───────────────────────────────────────────────────────
class Engulfing(Strategy):
    rr = 2.0

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._want_long  = False
        self._want_short = False

    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        o1, c1 = float(self.data.Open[-2]),  float(self.data.Close[-2])   # yesterday
        o2, c2 = float(self.data.Open[-3]),  float(self.data.Close[-3])   # day before

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        if body2 <= 0 or body1 < body2 * 1.1:  # must meaningfully engulf
            self._want_long = self._want_short = False

        # Bullish engulfing: day-before bearish, yesterday bullish & engulfs
        elif c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2:
            self._want_long  = True
            self._want_short = False

        # Bearish engulfing: day-before bullish, yesterday bearish & engulfs
        elif c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2:
            self._want_short = True
            self._want_long  = False

        else:
            self._want_long = self._want_short = False

        if self.position:
            self._want_long = self._want_short = False
            return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._want_long:
            sl      = p - 1.5 * av
            sl_dist = max(p - sl, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p + self.rr * sl_dist
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)
            self._want_long = False

        elif self._want_short:
            sl      = p + 1.5 * av
            sl_dist = max(sl - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p - self.rr * sl_dist
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)
            self._want_short = False

class Engulfing_R2(Engulfing): rr=2.0
class Engulfing_R3(Engulfing): rr=3.0


# ── Instruments & Variants ────────────────────────────────────────────────────
VARIANTS = [
    ('InsideBar 2:1', InsideBar_R2),
    ('InsideBar 3:1', InsideBar_R3),
    ('Donchian 2:1',  Donchian_R2),
    ('Donchian 3:1',  Donchian_R3),
    ('PinBar  2:1',   PinBar_R2),
    ('PinBar  3:1',   PinBar_R3),
    ('Engulf  2:1',   Engulfing_R2),
    ('Engulf  3:1',   Engulfing_R3),
]

INSTRUMENTS = [
    ('WHEAT',    'WHEAT_USD'),
    ('NATGAS',   'NATGAS_USD'),
    ('XAU/USD',  'XAU_USD'),
    ('XAG/USD',  'XAG_USD'),
    ('WTI Oil',  'WTICO_USD'),
    ('Brent',    'BCO_USD'),
    ('Corn',     'CORN_USD'),
    ('Soybeans', 'SOYBN_USD'),
    ('Sugar',    'SUGAR_USD'),
    ('Copper',   'XCUUSD'),
    ('JP225',    'JP225_USD'),
    ('UK100',    'UK100_GBP'),
    ('NAS100',   'NAS100_USD'),
    ('DE30',     'DE30_EUR'),
    ('SPX500',   'SPX500_USD'),
    ('HK33',     'HK33_HKD'),
    ('AU200',    'AU200_AUD'),
    ('FR40',     'FR40_EUR'),
    ('EU50',     'EU50_EUR'),
    ('EUR/USD',  'EUR_USD'),
    ('GBP/USD',  'GBP_USD'),
    ('USD/JPY',  'USD_JPY'),
    ('USD/CAD',  'USD_CAD'),
    ('AUD/USD',  'AUD_USD'),
    ('NZD/USD',  'NZD_USD'),
    ('USD/CHF',  'USD_CHF'),
    ('GBP/JPY',  'GBP_JPY'),
    ('EUR/JPY',  'EUR_JPY'),
    ('GBP/CHF',  'GBP_CHF'),
    ('EUR/GBP',  'EUR_GBP'),
    ('AUD/JPY',  'AUD_JPY'),
    ('GBP/AUD',  'GBP_AUD'),
    ('EUR/AUD',  'EUR_AUD'),
    ('CAD/JPY',  'CAD_JPY'),
]

print('D1 Strategy Audit — InsideBar / Donchian / PinBar / Engulfing')
print('10-year backtest, 34 instruments, 1.5 ATR SL, 2:1 & 3:1 R:R')
print()
print(f"{'Instrument':<10} {'Strategy':<14} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  Verdict")
print('─' * 80)

rows = []
for lbl, instr in INSTRUMENTS:
    try:
        df = load_oanda_data(instr, period='10y', interval='1d')
    except Exception as e:
        print(f'{lbl}: load error {e}')
        continue
    for var_lbl, cls in VARIANTS:
        try:
            bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                          margin=1/30, finalize_trades=True)
            s  = bt.run()
            sh  = float(s.get('Sharpe Ratio', 0) or 0)
            ann = float(s.get('Return (Ann.) [%]', 0) or 0)
            dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
            wr  = float(s.get('Win Rate [%]', 0) or 0)
            n   = int(s.get('# Trades', 0) or 0)
            pf  = float(s.get('Profit Factor', 0) or 0)
            ok  = sh >= 0.4 and ann > 0 and n >= 15
            verdict = 'ADD' if ok else ('WATCH' if sh >= 0.1 and ann > 0 else 'PASS')
            print(f'{lbl:<10} {var_lbl:<14} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% '
                  f'{wr:>5.1f}% {n:>5} {pf:>5.2f}  {verdict}')
            rows.append({'lbl': lbl, 'instr': instr, 'var': var_lbl,
                         'sh': sh, 'ann': ann, 'wr': wr, 'n': n,
                         'dd': dd, 'pf': pf, 'verdict': verdict})
        except Exception as e:
            print(f'{lbl:<10} {var_lbl:<14} error: {e}')
    print()

print('─' * 80)
adds  = [r for r in rows if r['verdict'] == 'ADD']
watch = [r for r in rows if r['verdict'] == 'WATCH']
print(f'ADD   (Sh ≥ 0.4, Ann > 0, N ≥ 15): {len(adds)}')
print(f'WATCH (Sh ≥ 0.1, Ann > 0):          {len(watch)}')
if adds or watch:
    print('\nTop candidates:')
    for r in sorted(adds + watch, key=lambda x: x['sh'], reverse=True)[:15]:
        print(f"  {r['lbl']:<10} {r['var']:<14} Sh {r['sh']:+.2f}  WR {r['wr']:.1f}%  "
              f"Ann {r['ann']:+.1f}%  DD {r['dd']:.1f}%  N={r['n']}  PF={r['pf']:.2f}")
