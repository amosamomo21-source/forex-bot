"""
RSI Divergence — D1 backtest.

Bullish divergence : price makes a LOWER low, RSI makes a HIGHER low → BUY
Bearish divergence : price makes a HIGHER high, RSI makes a LOWER high → SELL

This is the classic "catch the full reversal at the bottom" signal. Price and
momentum diverge because sellers (buyers) are exhausted — the move is running out
of steam even as price extends.

Mechanics:
  - Swing pivots: N-bar each side (bar is highest/lowest of 2N+1 window)
  - Divergence: compare RSI at the two most recent confirmed swing lows/highs
  - Entry: next bar after divergence confirmed
  - Extreme filter (optional): bull div only when RSI < 45, bear div only when RSI > 55
  - SL: 1.5 ATR from entry | TP: 2:1 and 3:1 R:R
  - D1, 10y backtest
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

def _rsi(close, n=14):
    c  = pd.Series(np.array(close, float))
    d  = c.diff()
    up = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return (100 * up / (up + dn)).values

def _atr(high, low, close, n=14):
    h  = pd.Series(np.array(high, float))
    l  = pd.Series(np.array(low, float))
    c  = pd.Series(np.array(close, float))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values


class RSIDiv(Strategy):
    pivot_n     = 5      # bars each side to confirm a swing
    rsi_n       = 14
    sl_atr      = 1.5
    rr          = 2.0
    extreme_flt = True   # only trade divergence when RSI is at an extreme

    def init(self):
        self.rsi_i = self.I(_rsi, self.data.Close, self.rsi_n)
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._s_lows  = []   # (price_low, rsi_at_low)
        self._s_highs = []   # (price_high, rsi_at_high)
        self._want_long  = False
        self._want_short = False

    def _is_pivot_low(self):
        n = self.pivot_n
        if len(self.data) < 2*n+2: return False
        c  = -(n+1)
        cl = float(self.data.Low[c])
        return all(float(self.data.Low[c+j]) > cl for j in range(-n, n+1) if j != 0)

    def _is_pivot_high(self):
        n = self.pivot_n
        if len(self.data) < 2*n+2: return False
        c  = -(n+1)
        ch = float(self.data.High[c])
        return all(float(self.data.High[c+j]) < ch for j in range(-n, n+1) if j != 0)

    def next(self):
        if len(self.data) < 30: return
        n  = self.pivot_n
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        # ── Detect pivot lows → bullish divergence ────────────────────────
        if self._is_pivot_low():
            pl_p = float(self.data.Low[-(n+1)])
            pl_r = float(self.rsi_i[-(n+1)])
            if not np.isnan(pl_r) and self._s_lows:
                prev_p, prev_r = self._s_lows[-1]
                if pl_p < prev_p and pl_r > prev_r:       # lower low, higher RSI
                    rsi_ok = (not self.extreme_flt) or pl_r < 45
                    if rsi_ok:
                        self._want_long = True
            if not np.isnan(pl_r):
                self._s_lows.append((pl_p, pl_r))
                if len(self._s_lows) > 6: self._s_lows.pop(0)

        # ── Detect pivot highs → bearish divergence ───────────────────────
        if self._is_pivot_high():
            ph_p = float(self.data.High[-(n+1)])
            ph_r = float(self.rsi_i[-(n+1)])
            if not np.isnan(ph_r) and self._s_highs:
                prev_p, prev_r = self._s_highs[-1]
                if ph_p > prev_p and ph_r < prev_r:       # higher high, lower RSI
                    rsi_ok = (not self.extreme_flt) or ph_r > 55
                    if rsi_ok:
                        self._want_short = True
            if not np.isnan(ph_r):
                self._s_highs.append((ph_p, ph_r))
                if len(self._s_highs) > 6: self._s_highs.pop(0)

        # ── Clear pending flags if already in a trade ─────────────────────
        if self.position:
            self._want_long = self._want_short = False
            return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._want_long:
            sl      = p - self.sl_atr * av
            sl_dist = max(p - sl, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p + self.rr * sl_dist
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)
            self._want_long = False

        elif self._want_short:
            sl      = p + self.sl_atr * av
            sl_dist = max(sl - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p - self.rr * sl_dist
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)
            self._want_short = False


# ── Variants ──────────────────────────────────────────────────────────────────
class Div_N5_R2_Flt(RSIDiv):  pivot_n=5;  rr=2.0; extreme_flt=True
class Div_N5_R3_Flt(RSIDiv):  pivot_n=5;  rr=3.0; extreme_flt=True
class Div_N8_R2_Flt(RSIDiv):  pivot_n=8;  rr=2.0; extreme_flt=True
class Div_N8_R3_Flt(RSIDiv):  pivot_n=8;  rr=3.0; extreme_flt=True
class Div_N5_R2_All(RSIDiv):  pivot_n=5;  rr=2.0; extreme_flt=False
class Div_N5_R3_All(RSIDiv):  pivot_n=5;  rr=3.0; extreme_flt=False

VARIANTS = [
    ('N5 2:1 Extreme', Div_N5_R2_Flt),
    ('N5 3:1 Extreme', Div_N5_R3_Flt),
    ('N8 2:1 Extreme', Div_N8_R2_Flt),
    ('N8 3:1 Extreme', Div_N8_R3_Flt),
    ('N5 2:1 NoFlt',   Div_N5_R2_All),
    ('N5 3:1 NoFlt',   Div_N5_R3_All),
]

INSTRUMENTS = [
    # Commodities
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
    # Indices
    ('JP225',    'JP225_USD'),
    ('UK100',    'UK100_GBP'),
    ('NAS100',   'NAS100_USD'),
    ('DE30',     'DE30_EUR'),
    ('SPX500',   'SPX500_USD'),
    ('HK33',     'HK33_HKD'),
    ('AU200',    'AU200_AUD'),
    ('FR40',     'FR40_EUR'),
    ('EU50',     'EU50_EUR'),
    # FX majors
    ('EUR/USD',  'EUR_USD'),
    ('GBP/USD',  'GBP_USD'),
    ('USD/JPY',  'USD_JPY'),
    ('USD/CAD',  'USD_CAD'),
    ('AUD/USD',  'AUD_USD'),
    ('NZD/USD',  'NZD_USD'),
    ('USD/CHF',  'USD_CHF'),
    # FX crosses
    ('GBP/JPY',  'GBP_JPY'),
    ('EUR/JPY',  'EUR_JPY'),
    ('GBP/CHF',  'GBP_CHF'),
    ('EUR/GBP',  'EUR_GBP'),
    ('AUD/JPY',  'AUD_JPY'),
    ('GBP/AUD',  'GBP_AUD'),
    ('EUR/AUD',  'EUR_AUD'),
    ('CAD/JPY',  'CAD_JPY'),
]

print('RSI Divergence — D1 10y backtest')
print('Bullish: lower price low + higher RSI low → BUY')
print('Bearish: higher price high + lower RSI high → SELL')
print()
print(f"{'Instrument':<10} {'Variant':<18} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  Verdict")
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
            b  = Backtest(df, cls, cash=100_000, commission=0.0002,
                          margin=1/30, finalize_trades=True)
            s  = b.run()
            sh  = float(s.get('Sharpe Ratio', 0) or 0)
            ann = float(s.get('Return (Ann.) [%]', 0) or 0)
            dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
            wr  = float(s.get('Win Rate [%]', 0) or 0)
            n   = int(s.get('# Trades', 0) or 0)
            pf  = float(s.get('Profit Factor', 0) or 0)
            ok  = sh >= 0.4 and ann > 0 and n >= 15
            verdict = 'ADD' if ok else ('WATCH' if sh >= 0.1 and ann > 0 else 'PASS')
            print(f'{lbl:<10} {var_lbl:<18} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% '
                  f'{wr:>5.1f}% {n:>5} {pf:>5.2f}  {verdict}')
            rows.append({'lbl': lbl, 'instr': instr, 'var': var_lbl,
                         'sh': sh, 'ann': ann, 'wr': wr, 'n': n, 'verdict': verdict})
        except Exception as e:
            print(f'{lbl:<10} {var_lbl:<18} error: {e}')
    print()

print('─' * 80)
adds  = [r for r in rows if r['verdict'] == 'ADD']
watch = [r for r in rows if r['verdict'] == 'WATCH']
print(f'ADD   (Sh ≥ 0.4, N ≥ 15): {len(adds)}')
print(f'WATCH (Sh ≥ 0.1):         {len(watch)}')
if adds or watch:
    print('\nBest candidates:')
    for r in sorted(adds+watch, key=lambda x: x['sh'], reverse=True)[:10]:
        print(f"  {r['lbl']:<10} {r['var']:<18} Sh {r['sh']:+.2f}  WR {r['wr']:.1f}%  "
              f"Ann {r['ann']:+.1f}%  N={r['n']}")
