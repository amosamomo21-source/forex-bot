"""
FVG v2 wide instrument scan — M30, 3:1 R:R, 10y
Same tighter filters as v2: EMA(200) trend, min gap 0.15%, session open only.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

def _ema(s, n):
    return pd.Series(np.array(s, float)).ewm(span=n, adjust=False).mean().values

class FVGv2(Strategy):
    ema_period  = 200
    rr          = 3.0
    fvg_expiry  = 8
    risk_pct    = RISK_PCT
    sl_buffer   = 0.0005
    min_gap_pct = 0.0015

    def init(self):
        self.trend    = self.I(_ema, self.data.Close, self.ema_period)
        self._dir     = None
        self._top     = np.nan
        self._bot     = np.nan
        self._fvg_idx = -999

    def next(self):
        idx = len(self.data) - 1
        if idx < 3: return
        t = self.data.index[-1]
        if not (7 <= t.hour < 9 or 13 <= t.hour < 15): return

        p  = float(self.data.Close[-1])
        tr = float(self.trend[-1])
        if np.isnan(tr): return

        h1 = float(self.data.High[-3])
        l1 = float(self.data.Low[-3])
        h3 = float(self.data.High[-1])
        l3 = float(self.data.Low[-1])
        min_gap = abs(p) * self.min_gap_pct

        if not self.position:
            if h1 < l3 and (l3 - h1) >= min_gap and p > tr:
                self._dir='bull'; self._bot=h1; self._top=l3; self._fvg_idx=idx
            elif l1 > h3 and (l1 - h3) >= min_gap and p < tr:
                self._dir='bear'; self._bot=h3; self._top=l1; self._fvg_idx=idx

        if self._dir and (idx - self._fvg_idx) > self.fvg_expiry:
            self._dir = None
        if not self._dir or self.position: return

        ra = self.equity * self.risk_pct
        if self._dir == 'bull' and float(self.data.Low[-1]) <= self._top:
            sl = self._bot * (1 - self.sl_buffer)
            sl_dist = max(p - sl, 1e-9)
            units = max(1, min(int(ra / sl_dist), 50_000))
            tp = p + self.rr * sl_dist
            if sl < p < tp: self.buy(size=units, sl=sl, tp=tp)
            self._dir = None
        elif self._dir == 'bear' and float(self.data.High[-1]) >= self._bot:
            sl = self._top * (1 + self.sl_buffer)
            sl_dist = max(sl - p, 1e-9)
            units = max(1, min(int(ra / sl_dist), 50_000))
            tp = p - self.rr * sl_dist
            if tp < p < sl: self.sell(size=units, sl=sl, tp=tp)
            self._dir = None

INSTRUMENTS = [
    # FX Majors
    ('EUR/USD',  'EUR_USD'),
    ('GBP/USD',  'GBP_USD'),
    ('USD/JPY',  'USD_JPY'),
    ('AUD/USD',  'AUD_USD'),
    ('USD/CAD',  'USD_CAD'),
    ('NZD/USD',  'NZD_USD'),
    # FX Crosses
    ('EUR/JPY',  'EUR_JPY'),
    ('GBP/JPY',  'GBP_JPY'),
    ('CHF/JPY',  'CHF_JPY'),
    ('AUD/JPY',  'AUD_JPY'),
    ('EUR/GBP',  'EUR_GBP'),
    ('EUR/AUD',  'EUR_AUD'),
    ('EUR/CAD',  'EUR_CAD'),
    ('GBP/CAD',  'GBP_CAD'),
    ('GBP/CHF',  'GBP_CHF'),
    ('AUD/CHF',  'AUD_CHF'),
    ('CAD/JPY',  'CAD_JPY'),
    ('NZD/JPY',  'NZD_JPY'),
    # Commodities
    ('XAU/USD',  'XAU_USD'),
    ('XAG/USD',  'XAG_USD'),
    ('WTI Oil',  'WTICO_USD'),
    ('Brent',    'BCO_USD'),
    ('NATGAS',   'NATGAS_USD'),
    # Indices
    ('NAS100',   'NAS100_USD'),
    ('DE30',     'DE30_EUR'),
    ('JP225',    'JP225_USD'),
]

print('FVG v2 — wide instrument scan, M30, 3:1 R:R, 10y')
print('EMA(200) trend | min gap 0.15% | session open only (London 07-09, NY 13-15)')
print()
print(f"{'Instrument':<12} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  EV/R   Verdict")
print('─' * 78)

rows = []
for lbl, instr in INSTRUMENTS:
    try:
        df = load_oanda_data(instr, period='10y', interval='30m')
        b  = Backtest(df, FVGv2, cash=100_000, commission=0.0002,
                      margin=1/30, finalize_trades=True)
        s  = b.run()
        sh  = float(s.get('Sharpe Ratio', 0) or 0)
        ann = float(s.get('Return (Ann.) [%]', 0) or 0)
        dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
        wr  = float(s.get('Win Rate [%]', 0) or 0)
        n   = int(s.get('# Trades', 0) or 0)
        pf  = float(s.get('Profit Factor', 0) or 0)
        ev  = (wr/100) * 3.0 - (1 - wr/100)
        ok  = sh >= 0.4 and ann > 0 and n >= 20
        verdict = 'ADD' if ok else ('WATCH' if sh >= 0.1 and ann > 0 else 'PASS')
        print(f'{lbl:<12} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% '
              f'{n:>5} {pf:>5.2f}  {ev:>+.3f}R  {verdict}')
        rows.append({'lbl': lbl, 'instr': instr, 'sh': sh, 'ann': ann,
                     'wr': wr, 'n': n, 'pf': pf, 'ev': ev, 'verdict': verdict})
    except Exception as e:
        print(f'{lbl:<12} ERROR: {e}')

# Summary
print('─' * 78)
adds   = [r for r in rows if r['verdict'] == 'ADD']
watchs = [r for r in rows if r['verdict'] == 'WATCH']
pos    = [r for r in rows if r['ann'] > 0]

print(f'\nPositive Ann%: {len(pos)}/{len(rows)} instruments')
print(f'ADD  (Sh ≥ 0.4): {len(adds)}')
print(f'WATCH (Sh ≥ 0.1): {len(watchs)}')

if adds or watchs:
    print('\nCandidates sorted by Sharpe:')
    candidates = sorted(adds + watchs, key=lambda r: r['sh'], reverse=True)
    for r in candidates:
        tag = 'ADD' if r['verdict'] == 'ADD' else 'WATCH'
        print(f"  [{tag}] {r['lbl']:<12} Sh {r['sh']:+.2f} Ann {r['ann']:+.1f}% "
              f"WR {r['wr']:.0f}% N={r['n']} PF={r['pf']:.2f}")

    # Aggregate across all WATCH/ADD instruments
    total_n    = sum(r['n'] for r in candidates)
    total_wins = sum(int(r['n'] * r['wr'] / 100) for r in candidates)
    agg_wr     = total_wins / total_n * 100 if total_n else 0
    print(f'\nIf all candidates run together: {total_n} trades / 10y = '
          f'{total_n//10} per year, agg WR {agg_wr:.1f}%')
