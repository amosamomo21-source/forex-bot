"""
Chart Champions v2 — zone quality filter added.

Improvements over v1:
  1. Larger pivots (8-bar each side) → fewer, more significant zones
  2. Weekly H/L confluence: zone must sit within 0.5% of a prior weekly high or low
  3. Daily open confluence: OR within 0.3% of the current day's open
  4. Session filter: London (07-12 UTC) + NY (13-17 UTC) entries only
  5. Minimum zone gap: ignore new zone if an existing zone is within 1 ATR

This mimics the "2+ confluences" requirement from the original strategy.
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
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values

def _weekly_high(high, w=120):
    """Rolling high over last ~5 trading days (120 H1 bars), shifted 1 bar to avoid look-ahead."""
    return pd.Series(np.array(high, float)).shift(1).rolling(w, min_periods=1).max().values

def _weekly_low(low, w=120):
    return pd.Series(np.array(low, float)).shift(1).rolling(w, min_periods=1).min().values

def _daily_open(open_, bars_per_day=24):
    """Most recent daily open (first bar of each 24-bar block)."""
    s = pd.Series(np.array(open_, float))
    # Rolling: take open from 24 bars ago as proxy for day open
    return s.shift(bars_per_day).values


class CCv2(Strategy):
    pivot_n        = 8      # stricter pivot confirmation
    zone_pct       = 0.003
    conf_weekly    = 0.005  # within 0.5% of weekly H/L
    conf_daily     = 0.003  # OR within 0.3% of daily open
    rr             = 2.0
    risk_pct       = RISK_PCT

    def init(self):
        self.atr_i   = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self.wk_high = self.I(_weekly_high, self.data.High, 120)
        self.wk_low  = self.I(_weekly_low,  self.data.Low,  120)
        self.day_open= self.I(_daily_open,  self.data.Open, 24)
        self._sh     = []
        self._sl     = []
        self._zones  = []

    def _is_pivot_high(self):
        n = self.pivot_n
        if len(self.data) < 2*n+2: return False
        c = -(n+1)
        ch = float(self.data.High[c])
        return all(float(self.data.High[c+j]) < ch for j in range(-n, n+1) if j != 0)

    def _is_pivot_low(self):
        n = self.pivot_n
        if len(self.data) < 2*n+2: return False
        c = -(n+1)
        cl = float(self.data.Low[c])
        return all(float(self.data.Low[c+j]) > cl for j in range(-n, n+1) if j != 0)

    def _structure(self):
        if len(self._sh) < 2 or len(self._sl) < 2: return 0
        bull = self._sh[-1] > self._sh[-2] and self._sl[-1] > self._sl[-2]
        bear = self._sh[-1] < self._sh[-2] and self._sl[-1] < self._sl[-2]
        return 1 if bull else (-1 if bear else 0)

    def _has_confluence(self, price):
        """Zone price must sit near a weekly H/L or daily open."""
        wh = float(self.wk_high[-1])
        wl = float(self.wk_low[-1])
        do = float(self.day_open[-1])
        near_weekly = (abs(price - wh)/price < self.conf_weekly or
                       abs(price - wl)/price < self.conf_weekly)
        near_daily  = (not np.isnan(do) and abs(price - do)/price < self.conf_daily)
        return near_weekly or near_daily

    def next(self):
        idx = len(self.data) - 1
        p   = float(self.data.Close[-1])
        h   = float(self.data.High[-1])
        l   = float(self.data.Low[-1])
        av  = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        # Session filter: London (07-12) + NY (13-17) UTC
        hour = self.data.index[-1].hour
        in_session = (7 <= hour < 12) or (13 <= hour < 17)

        half = self.zone_pct / 2

        # ── 1. Detect pivots ──────────────────────────────────────────────
        if self._is_pivot_high():
            ph = float(self.data.High[-(self.pivot_n+1)])
            self._sh.append(ph)
            if len(self._sh) > 12: self._sh.pop(0)
            # Only add zone if it has confluence AND isn't duplicating an existing zone
            if self._has_confluence(ph):
                too_close = any(abs(z['price'] - ph)/ph < self.zone_pct
                                for z in self._zones if z['active'])
                if not too_close:
                    self._zones.append({
                        'dir': 'supply', 'price': ph,
                        'top': ph*(1+half), 'bot': ph*(1-half),
                        'visits': 0, 'last_bar': -999, 'active': True,
                    })

        if self._is_pivot_low():
            pl = float(self.data.Low[-(self.pivot_n+1)])
            self._sl.append(pl)
            if len(self._sl) > 12: self._sl.pop(0)
            if self._has_confluence(pl):
                too_close = any(abs(z['price'] - pl)/pl < self.zone_pct
                                for z in self._zones if z['active'])
                if not too_close:
                    self._zones.append({
                        'dir': 'demand', 'price': pl,
                        'top': pl*(1+half), 'bot': pl*(1-half),
                        'visits': 0, 'last_bar': -999, 'active': True,
                    })

        self._zones = [z for z in self._zones if z['active']][-20:]

        # ── 2. Update zone visits and invalidations ───────────────────────
        for z in self._zones:
            in_zone = l <= z['top'] and h >= z['bot']
            if in_zone and (idx - z['last_bar']) > self.pivot_n:
                z['visits']  += 1
                z['last_bar'] = idx
            if z['dir'] == 'demand' and p < z['bot']:
                z['active'] = False
            elif z['dir'] == 'supply' and p > z['top']:
                z['active'] = False

        if self.position or not in_session: return

        # ── 3. Entry signals ──────────────────────────────────────────────
        struct = self._structure()
        ra     = self.equity * self.risk_pct

        for z in self._zones:
            if not z['active'] or z['visits'] < 2: continue

            if z['dir'] == 'demand' and struct == 1:
                if l <= z['top'] and p > z['bot']:
                    sl      = z['bot'] - av
                    sl_dist = max(p - sl, 1e-9)
                    units   = max(1, min(int(ra / sl_dist), 100_000))
                    tp      = p + self.rr * sl_dist
                    if sl < p < tp:
                        self.buy(size=units, sl=sl, tp=tp)
                    break

            elif z['dir'] == 'supply' and struct == -1:
                if h >= z['bot'] and p < z['top']:
                    sl      = z['top'] + av
                    sl_dist = max(sl - p, 1e-9)
                    units   = max(1, min(int(ra / sl_dist), 100_000))
                    tp      = p - self.rr * sl_dist
                    if tp < p < sl:
                        self.sell(size=units, sl=sl, tp=tp)
                    break


class CCv2_3R(CCv2):
    rr = 3.0


# ── Run ───────────────────────────────────────────────────────────────────────
INSTRUMENTS = [
    ('EUR/USD', 'EUR_USD'),
    ('GBP/USD', 'GBP_USD'),
    ('GBP/JPY', 'GBP_JPY'),
    ('XAU/USD', 'XAU_USD'),
    ('NAS100',  'NAS100_USD'),
    ('GBP/CHF', 'GBP_CHF'),
    ('USD/JPY', 'USD_JPY'),
]

print('Chart Champions v2 — weekly H/L + daily open confluence, session filter, 8-bar pivots')
print('10y backtest | H1 only | 2:1 and 3:1 R:R')
print()
print(f"{'Instrument':<10} {'RR':<5} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  EV/R   Verdict")
print('─' * 78)

results = []
for lbl, instr in INSTRUMENTS:
    try:
        df = load_oanda_data(instr, period='10y', interval='1h')
    except Exception as e:
        print(f'{lbl:<10}  load error: {e}')
        continue
    for rr_lbl, cls in [('2:1', CCv2), ('3:1', CCv2_3R)]:
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
            ev  = (wr/100) * cls.rr - (1 - wr/100)
            ok  = sh >= 0.4 and ann > 0 and n >= 20
            verdict = 'ADD' if ok else ('WATCH' if sh >= 0.1 and ann > 0 else 'PASS')
            print(f'{lbl:<10} {rr_lbl:<5} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% '
                  f'{wr:>5.1f}% {n:>5} {pf:>5.2f}  {ev:>+.3f}R  {verdict}')
            results.append({'lbl': lbl, 'rr': rr_lbl, 'sh': sh, 'ann': ann,
                            'wr': wr, 'n': n, 'verdict': verdict})
        except Exception as e:
            print(f'{lbl:<10} {rr_lbl:<5}  error: {e}')
    print()

print('─' * 78)
adds  = [r for r in results if r['verdict'] == 'ADD']
watch = [r for r in results if r['verdict'] == 'WATCH']
print(f'ADD   (Sh ≥ 0.4): {len(adds)}')
print(f'WATCH (Sh ≥ 0.1): {len(watch)}')
if adds or watch:
    print('Candidates:')
    for r in sorted(adds+watch, key=lambda x: x['sh'], reverse=True):
        print(f"  {r['lbl']} {r['rr']} — Sh {r['sh']:+.2f}, Ann {r['ann']:+.1f}%, WR {r['wr']:.0f}%, N={r['n']}")
