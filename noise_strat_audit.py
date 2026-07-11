"""
Noise-to-Opportunity — D1 backtest across 34 instruments.

Strategy 1: False Breakout (FBO)
  - Price wicks above yesterday's high (or below yesterday's low) intraday
    but CLOSES back inside the previous day's range → trapped breakout traders
  - Variants: prev-day level and 20-day channel level
  - Entry next bar. SL = 1.5 ATR beyond the false-break wick. TP = 2:1 / 3:1

Strategy 2: Volatility Spike Fade (VSF)
  - Today's candle range > N × ATR(14) — an abnormally large move
  - If close is in the top 25% of the range → SELL (exhaustion)
  - If close is in the bottom 25% of the range → BUY (exhaustion)
  - Entry next bar. SL = 1.5 ATR. TP = 2:1 / 3:1

10-year D1 backtest, all 34 instruments.
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


# ── 1a. Previous-Day False Breakout ──────────────────────────────────────────
class PDFalseBreak(Strategy):
    """Wick pierces yesterday's H/L but close stays inside → fade."""
    rr = 2.0

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._want_long  = False
        self._want_short = False
        self._sl_ref     = 0.0

    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        # Yesterday's candle
        h1  = float(self.data.High[-2])
        l1  = float(self.data.Low[-2])
        c1  = float(self.data.Close[-2])
        # Day before (the reference range)
        h2  = float(self.data.High[-3])
        l2  = float(self.data.Low[-3])

        # Bullish FBO: wick below prior low, close back above prior low
        if l1 < l2 and c1 > l2:
            self._want_long  = True
            self._want_short = False
            self._sl_ref     = l1    # SL below the false-break wick

        # Bearish FBO: wick above prior high, close back below prior high
        elif h1 > h2 and c1 < h2:
            self._want_short = True
            self._want_long  = False
            self._sl_ref     = h1    # SL above the false-break wick

        else:
            self._want_long = self._want_short = False

        if self.position:
            self._want_long = self._want_short = False
            return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._want_long:
            sl_dist = max(p - (self._sl_ref - 0.5 * av), 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            sl = p - sl_dist
            tp = p + self.rr * sl_dist
            if sl < p < tp:
                self.buy(size=units, sl=sl, tp=tp)
            self._want_long = False

        elif self._want_short:
            sl_dist = max((self._sl_ref + 0.5 * av) - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            sl = p + sl_dist
            tp = p - self.rr * sl_dist
            if tp < p < sl:
                self.sell(size=units, sl=sl, tp=tp)
            self._want_short = False

class PDFBO_R2(PDFalseBreak): rr=2.0
class PDFBO_R3(PDFalseBreak): rr=3.0


# ── 1b. N-Day Channel False Breakout ─────────────────────────────────────────
class ChanFalseBreak(Strategy):
    """Price wicks beyond 20-day channel but closes back inside → fade."""
    period = 20
    rr     = 2.0

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._want_long  = False
        self._want_short = False
        self._sl_ref     = 0.0

    def next(self):
        n = self.period
        if len(self.data) < n + 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        # Channel from prior N bars (exclude yesterday)
        chan_high = max(float(self.data.High[-(i+3)]) for i in range(n))
        chan_low  = min(float(self.data.Low[-(i+3)])  for i in range(n))

        h1 = float(self.data.High[-2])
        l1 = float(self.data.Low[-2])
        c1 = float(self.data.Close[-2])

        if l1 < chan_low and c1 > chan_low:
            self._want_long  = True
            self._want_short = False
            self._sl_ref     = l1
        elif h1 > chan_high and c1 < chan_high:
            self._want_short = True
            self._want_long  = False
            self._sl_ref     = h1
        else:
            self._want_long = self._want_short = False

        if self.position:
            self._want_long = self._want_short = False
            return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._want_long:
            sl_dist = max(p - (self._sl_ref - 0.5 * av), 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            sl = p - sl_dist; tp = p + self.rr * sl_dist
            if sl < p < tp: self.buy(size=units, sl=sl, tp=tp)
            self._want_long = False

        elif self._want_short:
            sl_dist = max((self._sl_ref + 0.5 * av) - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            sl = p + sl_dist; tp = p - self.rr * sl_dist
            if tp < p < sl: self.sell(size=units, sl=sl, tp=tp)
            self._want_short = False

class ChanFBO_R2(ChanFalseBreak): period=20; rr=2.0
class ChanFBO_R3(ChanFalseBreak): period=20; rr=3.0


# ── 2. Volatility Spike Fade ─────────────────────────────────────────────────
class VolSpikeFade(Strategy):
    """Abnormally large candle with close near extreme → mean reversion."""
    vol_mult  = 2.0   # candle range must be > vol_mult × ATR
    close_pct = 0.25  # close must be in top/bottom % of candle range
    rr        = 2.0

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._want_long  = False
        self._want_short = False

    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        h1 = float(self.data.High[-2])
        l1 = float(self.data.Low[-2])
        c1 = float(self.data.Close[-2])
        rng = h1 - l1
        if rng < self.vol_mult * av:
            self._want_long = self._want_short = False
        else:
            pos_in_range = (c1 - l1) / rng
            # Close in bottom 25% → big down spike → fade with BUY
            if pos_in_range <= self.close_pct:
                self._want_long  = True
                self._want_short = False
            # Close in top 25% → big up spike → fade with SELL
            elif pos_in_range >= (1 - self.close_pct):
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
            if sl < p < tp: self.buy(size=units, sl=sl, tp=tp)
            self._want_long = False

        elif self._want_short:
            sl      = p + 1.5 * av
            sl_dist = max(sl - p, 1e-9)
            units   = max(1, min(int(ra / sl_dist), 100_000))
            tp      = p - self.rr * sl_dist
            if tp < p < sl: self.sell(size=units, sl=sl, tp=tp)
            self._want_short = False

# Vol×2 and Vol×1.5 variants, both R:R 2:1 and 3:1
class VSF_2x_R2(VolSpikeFade):  vol_mult=2.0; rr=2.0
class VSF_2x_R3(VolSpikeFade):  vol_mult=2.0; rr=3.0
class VSF_15x_R2(VolSpikeFade): vol_mult=1.5; rr=2.0
class VSF_15x_R3(VolSpikeFade): vol_mult=1.5; rr=3.0


# ── Instruments ───────────────────────────────────────────────────────────────
VARIANTS = [
    ('PD FBO 2:1',   PDFBO_R2),
    ('PD FBO 3:1',   PDFBO_R3),
    ('Ch FBO 2:1',   ChanFBO_R2),
    ('Ch FBO 3:1',   ChanFBO_R3),
    ('VSF 2x R2',    VSF_2x_R2),
    ('VSF 2x R3',    VSF_2x_R3),
    ('VSF 1.5x R2',  VSF_15x_R2),
    ('VSF 1.5x R3',  VSF_15x_R3),
]

INSTRUMENTS = [
    ('WHEAT',    'WHEAT_USD'),  ('NATGAS',   'NATGAS_USD'),
    ('XAU/USD',  'XAU_USD'),   ('XAG/USD',  'XAG_USD'),
    ('WTI Oil',  'WTICO_USD'), ('Brent',    'BCO_USD'),
    ('Corn',     'CORN_USD'),  ('Soybeans', 'SOYBN_USD'),
    ('Sugar',    'SUGAR_USD'), ('Copper',   'XCUUSD'),
    ('JP225',    'JP225_USD'), ('UK100',    'UK100_GBP'),
    ('NAS100',   'NAS100_USD'),('DE30',     'DE30_EUR'),
    ('SPX500',   'SPX500_USD'),('HK33',     'HK33_HKD'),
    ('AU200',    'AU200_AUD'), ('FR40',     'FR40_EUR'),
    ('EU50',     'EU50_EUR'),  ('EUR/USD',  'EUR_USD'),
    ('GBP/USD',  'GBP_USD'),   ('USD/JPY',  'USD_JPY'),
    ('USD/CAD',  'USD_CAD'),   ('AUD/USD',  'AUD_USD'),
    ('NZD/USD',  'NZD_USD'),   ('USD/CHF',  'USD_CHF'),
    ('GBP/JPY',  'GBP_JPY'),   ('EUR/JPY',  'EUR_JPY'),
    ('GBP/CHF',  'GBP_CHF'),   ('EUR/GBP',  'EUR_GBP'),
    ('AUD/JPY',  'AUD_JPY'),   ('GBP/AUD',  'GBP_AUD'),
    ('EUR/AUD',  'EUR_AUD'),   ('CAD/JPY',  'CAD_JPY'),
]

print('Noise Strategy Audit — False Breakout & Volatility Spike Fade')
print('D1 10-year backtest, 34 instruments, 1.5 ATR SL, 2:1 & 3:1 R:R')
print()
print(f"{'Instrument':<10} {'Strategy':<14} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  Verdict")
print('─' * 82)

rows = []
for lbl, instr in INSTRUMENTS:
    try:
        df = load_oanda_data(instr, period='10y', interval='1d')
    except Exception as e:
        print(f'{lbl}: load error {e}'); continue
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

print('─' * 82)
adds  = [r for r in rows if r['verdict'] == 'ADD']
watch = [r for r in rows if r['verdict'] == 'WATCH']
print(f'ADD   (Sh ≥ 0.4, N ≥ 15): {len(adds)}')
print(f'WATCH (Sh ≥ 0.1):         {len(watch)}')
if adds or watch:
    print('\nTop candidates:')
    for r in sorted(adds + watch, key=lambda x: x['sh'], reverse=True)[:15]:
        print(f"  {r['lbl']:<10} {r['var']:<14} Sh {r['sh']:+.2f}  WR {r['wr']:.1f}%"
              f"  Ann {r['ann']:+.1f}%  DD {r['dd']:.1f}%  N={r['n']}  PF={r['pf']:.2f}")
