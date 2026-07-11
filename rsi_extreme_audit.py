"""
RSI Extreme Fade — D1 mean reversion at statistical exhaustion.

Logic: when RSI reaches a genuine extreme (below threshold or above threshold),
price is statistically likely to snap back. Tight TP maximises win rate.

Entry  : RSI crosses below oversold_level (BUY) or above overbought_level (SELL)
SL     : 1.5 ATR beyond entry
TP     : 0.8× or 1.0× risk distance (tight — prioritises WR over R:R)
Filter : no trend filter (pure mean reversion — markets can't stay extreme)

Testing:
  RSI thresholds : 20/80, 15/85, 10/90
  TP multipliers : 0.8:1, 1.0:1
  Instruments    : commodities, indices, FX majors, 10y D1
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

def _rsi(close, n=14):
    c   = pd.Series(np.array(close, float))
    d   = c.diff()
    up  = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn  = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return (100 * up / (up + dn)).values

def _atr(high, low, close, n=14):
    h  = pd.Series(np.array(high, float))
    l  = pd.Series(np.array(low, float))
    c  = pd.Series(np.array(close, float))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values


class RSIFade(Strategy):
    oversold   = 20    # RSI below this → BUY
    overbought = 80    # RSI above this → SELL
    sl_atr     = 1.5
    tp_mult    = 0.8   # TP = 0.8 × risk → targets 75%+ WR
    risk_pct   = RISK_PCT

    def init(self):
        self.rsi_i = self.I(_rsi, self.data.Close, 14)
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        if self.position or len(self.data) < 20: return
        r  = float(self.rsi_i[-1])
        rp = float(self.rsi_i[-2])
        av = float(self.atr_i[-1])
        p  = float(self.data.Close[-1])
        if np.isnan(r) or np.isnan(av) or av <= 0: return

        ra    = self.equity * self.risk_pct
        units = max(1, min(int(ra / (self.sl_atr * av)), 100_000))

        # Cross INTO extreme — fire on the bar where RSI first enters the zone
        if r < self.oversold:
            sl = p - self.sl_atr * av
            tp = p + self.tp_mult * self.sl_atr * av
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)

        elif r > self.overbought:
            sl = p + self.sl_atr * av
            tp = p - self.tp_mult * self.sl_atr * av
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)


# ── Parameter variants ────────────────────────────────────────────────────────
class RSI20_08(RSIFade): oversold=20; overbought=80; tp_mult=0.8
class RSI20_10(RSIFade): oversold=20; overbought=80; tp_mult=1.0
class RSI15_08(RSIFade): oversold=15; overbought=85; tp_mult=0.8
class RSI15_10(RSIFade): oversold=15; overbought=85; tp_mult=1.0
class RSI10_08(RSIFade): oversold=10; overbought=90; tp_mult=0.8
class RSI10_10(RSIFade): oversold=10; overbought=90; tp_mult=1.0

VARIANTS = [
    ('RSI<20 TP0.8', RSI20_08),
    ('RSI<20 TP1.0', RSI20_10),
    ('RSI<15 TP0.8', RSI15_08),
    ('RSI<15 TP1.0', RSI15_10),
    ('RSI<10 TP0.8', RSI10_08),
    ('RSI<10 TP1.0', RSI10_10),
]

INSTRUMENTS = [
    # Commodities
    ('WHEAT',   'WHEAT_USD'),
    ('NATGAS',  'NATGAS_USD'),
    ('XAU/USD', 'XAU_USD'),
    ('XAG/USD', 'XAG_USD'),
    ('WTI Oil', 'WTICO_USD'),
    ('Brent',   'BCO_USD'),
    ('Corn',    'CORN_USD'),
    ('Soybeans','SOYBN_USD'),
    ('Sugar',   'SUGAR_USD'),
    ('Copper',  'XCUUSD'),
    # Indices
    ('NAS100',  'NAS100_USD'),
    ('SPX500',  'SPX500_USD'),
    ('DE30',    'DE30_EUR'),
    ('JP225',   'JP225_USD'),
    ('UK100',   'UK100_GBP'),
    ('HK33',    'HK33_HKD'),
    # FX majors + crosses
    ('EUR/USD', 'EUR_USD'),
    ('GBP/USD', 'GBP_USD'),
    ('GBP/JPY', 'GBP_JPY'),
    ('AUD/USD', 'AUD_USD'),
    ('USD/CAD', 'USD_CAD'),
    ('NZD/USD', 'NZD_USD'),
    ('EUR/JPY', 'EUR_JPY'),
    ('USD/CHF', 'USD_CHF'),
]

print('RSI Extreme Fade — D1 mean reversion, 10y backtest')
print('SL = 1.5 ATR | TP tested at 0.8× and 1.0× risk | three RSI thresholds')
print()
print(f"{'Instrument':<10} {'Params':<14} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  BE-WR  Verdict")
print('─' * 84)

rows = []
for lbl, instr in INSTRUMENTS:
    try:
        df = load_oanda_data(instr, period='10y', interval='1d')
    except Exception as e:
        print(f'{lbl}: load error {e}')
        continue

    best = None
    for var_lbl, cls in VARIANTS:
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
            tp  = cls.tp_mult
            be  = 100 / (1 + tp)          # break-even WR for this R:R
            ok  = sh >= 0.4 and ann > 0 and n >= 10
            verdict = 'ADD' if ok else ('WATCH' if sh >= 0.1 and ann > 0 else 'PASS')
            print(f'{lbl:<10} {var_lbl:<14} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% '
                  f'{wr:>5.1f}% {n:>5} {pf:>5.2f}  {be:.0f}%    {verdict}')
            rows.append({'lbl': lbl, 'instr': instr, 'var': var_lbl,
                         'sh': sh, 'ann': ann, 'wr': wr, 'n': n, 'pf': pf,
                         'be': be, 'verdict': verdict, 'cls': cls})
            if best is None or sh > best['sh']:
                best = rows[-1]
        except Exception as e:
            print(f'{lbl:<10} {var_lbl:<14} error: {e}')
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print('─' * 84)
adds  = [r for r in rows if r['verdict'] == 'ADD']
watch = [r for r in rows if r['verdict'] == 'WATCH']
above75 = [r for r in rows if r['wr'] >= 75 and r['ann'] > 0]

print(f'\nADD   (Sh ≥ 0.4, N ≥ 10): {len(adds)}')
print(f'WATCH (Sh ≥ 0.1):         {len(watch)}')
print(f'WR ≥ 75% AND positive Ann: {len(above75)}')

if above75:
    print('\nInstruments achieving 75%+ WR with positive returns:')
    for r in sorted(above75, key=lambda x: x['wr'], reverse=True):
        print(f"  {r['lbl']:<10} {r['var']:<14} WR {r['wr']:.1f}%  Sh {r['sh']:+.2f}  "
              f"Ann {r['ann']:+.1f}%  N={r['n']}  PF={r['pf']:.2f}")

candidates = sorted(adds + watch, key=lambda x: x['sh'], reverse=True)
if candidates:
    print('\nBest candidates by Sharpe:')
    for r in candidates[:10]:
        print(f"  {r['lbl']:<10} {r['var']:<14} Sh {r['sh']:+.2f}  WR {r['wr']:.1f}%  "
              f"Ann {r['ann']:+.1f}%  N={r['n']}")
