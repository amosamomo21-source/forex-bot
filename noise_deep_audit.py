"""
Deep validation — Top noise strategy candidates from noise_strat_audit.py:
  1. UK100    — Volatility Spike Fade 2x, R2 & R3
  2. USD/CAD  — Previous-Day False Breakout 3:1
  3. JP225    — Channel False Breakout 3:1

Tests per candidate:
  A. Walk-forward: 70% in-sample / 30% out-of-sample
  B. Year-by-year P&L consistency
  C. Parameter sensitivity (vol multiplier, period, RR, SL variants)
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


# ── Strategy definitions ──────────────────────────────────────────────────────

class VolSpikeFade(Strategy):
    vol_mult  = 2.0
    close_pct = 0.25
    rr        = 2.0
    sl_atr    = 1.5

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._wl = self._ws = False

    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        h1, l1, c1 = float(self.data.High[-2]), float(self.data.Low[-2]), float(self.data.Close[-2])
        rng = h1 - l1
        if rng < self.vol_mult * av:
            self._wl = self._ws = False
        else:
            pos = (c1 - l1) / rng
            if pos <= self.close_pct:
                self._wl, self._ws = True, False
            elif pos >= (1 - self.close_pct):
                self._ws, self._wl = True, False
            else:
                self._wl = self._ws = False

        if self.position:
            self._wl = self._ws = False; return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._wl:
            sl = p - self.sl_atr * av; sd = max(p - sl, 1e-9)
            u  = max(1, min(int(ra / sd), 100_000)); tp = p + self.rr * sd
            if sl < p < tp: self.buy(size=u, sl=sl, tp=tp)
            self._wl = False

        elif self._ws:
            sl = p + self.sl_atr * av; sd = max(sl - p, 1e-9)
            u  = max(1, min(int(ra / sd), 100_000)); tp = p - self.rr * sd
            if tp < p < sl: self.sell(size=u, sl=sl, tp=tp)
            self._ws = False

# VSF variants: vol multiplier × R:R
class VSF_2x_R2(VolSpikeFade):   vol_mult=2.0; rr=2.0; sl_atr=1.5
class VSF_2x_R3(VolSpikeFade):   vol_mult=2.0; rr=3.0; sl_atr=1.5
class VSF_2x_R4(VolSpikeFade):   vol_mult=2.0; rr=4.0; sl_atr=1.5
class VSF_15x_R2(VolSpikeFade):  vol_mult=1.5; rr=2.0; sl_atr=1.5
class VSF_15x_R3(VolSpikeFade):  vol_mult=1.5; rr=3.0; sl_atr=1.5
class VSF_25x_R2(VolSpikeFade):  vol_mult=2.5; rr=2.0; sl_atr=1.5
class VSF_25x_R3(VolSpikeFade):  vol_mult=2.5; rr=3.0; sl_atr=1.5
class VSF_2x_SL1(VolSpikeFade):  vol_mult=2.0; rr=2.0; sl_atr=1.0
class VSF_2x_SL2(VolSpikeFade):  vol_mult=2.0; rr=2.0; sl_atr=2.0


class PDFalseBreak(Strategy):
    rr     = 3.0
    sl_atr = 0.5   # SL = wick + 0.5 ATR buffer

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._wl = self._ws = False
        self._sl_ref = 0.0

    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        h1, l1, c1 = float(self.data.High[-2]), float(self.data.Low[-2]), float(self.data.Close[-2])
        h2, l2     = float(self.data.High[-3]), float(self.data.Low[-3])

        if l1 < l2 and c1 > l2:
            self._wl, self._ws, self._sl_ref = True, False, l1
        elif h1 > h2 and c1 < h2:
            self._ws, self._wl, self._sl_ref = True, False, h1
        else:
            self._wl = self._ws = False

        if self.position:
            self._wl = self._ws = False; return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._wl:
            sl_dist = max(p - (self._sl_ref - self.sl_atr * av), 1e-9)
            u = max(1, min(int(ra / sl_dist), 100_000))
            sl, tp = p - sl_dist, p + self.rr * sl_dist
            if sl < p < tp: self.buy(size=u, sl=sl, tp=tp)
            self._wl = False

        elif self._ws:
            sl_dist = max((self._sl_ref + self.sl_atr * av) - p, 1e-9)
            u = max(1, min(int(ra / sl_dist), 100_000))
            sl, tp = p + sl_dist, p - self.rr * sl_dist
            if tp < p < sl: self.sell(size=u, sl=sl, tp=tp)
            self._ws = False

class PDFBO_R2(PDFalseBreak):     rr=2.0; sl_atr=0.5
class PDFBO_R25(PDFalseBreak):    rr=2.5; sl_atr=0.5
class PDFBO_R3(PDFalseBreak):     rr=3.0; sl_atr=0.5
class PDFBO_R4(PDFalseBreak):     rr=4.0; sl_atr=0.5
class PDFBO_SL025(PDFalseBreak):  rr=3.0; sl_atr=0.25
class PDFBO_SL075(PDFalseBreak):  rr=3.0; sl_atr=0.75


class ChanFalseBreak(Strategy):
    period = 20
    rr     = 3.0
    sl_atr = 0.5

    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._wl = self._ws = False
        self._sl_ref = 0.0

    def next(self):
        n = self.period
        if len(self.data) < n + 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return

        chan_high = max(float(self.data.High[-(i+3)]) for i in range(n))
        chan_low  = min(float(self.data.Low[-(i+3)])  for i in range(n))

        h1, l1, c1 = float(self.data.High[-2]), float(self.data.Low[-2]), float(self.data.Close[-2])

        if l1 < chan_low and c1 > chan_low:
            self._wl, self._ws, self._sl_ref = True, False, l1
        elif h1 > chan_high and c1 < chan_high:
            self._ws, self._wl, self._sl_ref = True, False, h1
        else:
            self._wl = self._ws = False

        if self.position:
            self._wl = self._ws = False; return

        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT

        if self._wl:
            sl_dist = max(p - (self._sl_ref - self.sl_atr * av), 1e-9)
            u = max(1, min(int(ra / sl_dist), 100_000))
            sl, tp = p - sl_dist, p + self.rr * sl_dist
            if sl < p < tp: self.buy(size=u, sl=sl, tp=tp)
            self._wl = False

        elif self._ws:
            sl_dist = max((self._sl_ref + self.sl_atr * av) - p, 1e-9)
            u = max(1, min(int(ra / sl_dist), 100_000))
            sl, tp = p + sl_dist, p - self.rr * sl_dist
            if tp < p < sl: self.sell(size=u, sl=sl, tp=tp)
            self._ws = False

class ChanFBO_P15_R3(ChanFalseBreak): period=15; rr=3.0; sl_atr=0.5
class ChanFBO_P20_R3(ChanFalseBreak): period=20; rr=3.0; sl_atr=0.5
class ChanFBO_P25_R3(ChanFalseBreak): period=25; rr=3.0; sl_atr=0.5
class ChanFBO_P20_R2(ChanFalseBreak): period=20; rr=2.0; sl_atr=0.5
class ChanFBO_P20_R4(ChanFalseBreak): period=20; rr=4.0; sl_atr=0.5
class ChanFBO_SL025(ChanFalseBreak):  period=20; rr=3.0; sl_atr=0.25
class ChanFBO_SL075(ChanFalseBreak):  period=20; rr=3.0; sl_atr=0.75


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(df, cls):
    bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                  margin=1/30, finalize_trades=True)
    s = bt.run()
    return (float(s.get('Sharpe Ratio', 0) or 0),
            float(s.get('Return (Ann.) [%]', 0) or 0),
            float(s.get('Max. Drawdown [%]', 0) or 0),
            float(s.get('Win Rate [%]', 0) or 0),
            int(s.get('# Trades', 0) or 0),
            float(s.get('Profit Factor', 0) or 0), s)


def year_by_year(df, cls):
    results = []
    for y in sorted(df.index.year.unique()):
        sl = df[df.index.year == y]
        if len(sl) < 30: continue
        try:
            bt = Backtest(sl, cls, cash=100_000, commission=0.0002,
                          margin=1/30, finalize_trades=True)
            s  = bt.run()
            results.append((y,
                             float(s.get('Return (Ann.) [%]', 0) or 0),
                             int(s.get('# Trades', 0) or 0),
                             float(s.get('Win Rate [%]', 0) or 0)))
        except Exception:
            results.append((y, 0.0, 0, 0.0))
    return results


def sep(): print('─' * 72)


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 1: UK100 — Volatility Spike Fade 2× ATR')
print('═'*72)

df_uk = load_oanda_data('UK100_GBP', period='10y', interval='1d')
split = int(len(df_uk) * 0.70)
df_train, df_oos = df_uk.iloc[:split], df_uk.iloc[split:]

print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}  [R2 variant]")
sh,ann,dd,wr,n,pf,_ = run(df_train, VSF_2x_R2)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_oos, VSF_2x_R2)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_uk, VSF_2x_R2)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print(f"\n  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}  [R3 variant]")
sh,ann,dd,wr,n,pf,_ = run(df_train, VSF_2x_R3)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_oos, VSF_2x_R3)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_uk, VSF_2x_R3)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print('\nB. Year-by-Year P&L  (R2 variant — best WR)')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_uk, VSF_2x_R2):
    bar = '█' * max(0, int(abs(ann) * 10)) if ann > 0 else '░' * max(0, int(abs(ann) * 10))
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

print('\nC. Parameter Sensitivity (vol multiplier & RR variants)')
sep()
print(f"  {'Variant':<18} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [
    ('VM 1.5x R2',   VSF_15x_R2), ('VM 2.0x R2 ★', VSF_2x_R2),
    ('VM 2.5x R2',   VSF_25x_R2), ('VM 1.5x R3',   VSF_15x_R3),
    ('VM 2.0x R3 ★', VSF_2x_R3),  ('VM 2.5x R3',   VSF_25x_R3),
    ('VM 2.0x R4',   VSF_2x_R4),  ('SL 1.0xATR',   VSF_2x_SL1),
    ('SL 2.0xATR',   VSF_2x_SL2),
]:
    try:
        sh,ann,dd,wr,n,pf,_ = run(df_uk, cls)
        print(f"  {lbl:<18} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")
    except Exception as e:
        print(f"  {lbl:<18} error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 2: USD/CAD — Previous-Day False Breakout 3:1')
print('═'*72)

df_cad = load_oanda_data('USD_CAD', period='10y', interval='1d')
split  = int(len(df_cad) * 0.70)
df_train, df_oos = df_cad.iloc[:split], df_cad.iloc[split:]

print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}")
sh,ann,dd,wr,n,pf,_ = run(df_train, PDFBO_R3)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_oos, PDFBO_R3)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_cad, PDFBO_R3)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print('\nB. Year-by-Year P&L')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_cad, PDFBO_R3):
    bar = '█' * max(0, int(abs(ann) * 10)) if ann > 0 else '░' * max(0, int(abs(ann) * 10))
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

print('\nC. Parameter Sensitivity (RR & SL buffer variants)')
sep()
print(f"  {'Variant':<18} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [
    ('RR 2.0:1',    PDFBO_R2),   ('RR 2.5:1',    PDFBO_R25),
    ('RR 3.0:1 ★',  PDFBO_R3),   ('RR 4.0:1',    PDFBO_R4),
    ('SL buf 0.25x', PDFBO_SL025),('SL buf 0.75x', PDFBO_SL075),
]:
    try:
        sh,ann,dd,wr,n,pf,_ = run(df_cad, cls)
        print(f"  {lbl:<18} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")
    except Exception as e:
        print(f"  {lbl:<18} error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 3: JP225 — Channel False Breakout 3:1')
print('═'*72)

df_jp = load_oanda_data('JP225_USD', period='10y', interval='1d')
split  = int(len(df_jp) * 0.70)
df_train, df_oos = df_jp.iloc[:split], df_jp.iloc[split:]

print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}")
sh,ann,dd,wr,n,pf,_ = run(df_train, ChanFBO_P20_R3)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_oos, ChanFBO_P20_R3)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_jp, ChanFBO_P20_R3)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print('\nB. Year-by-Year P&L')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_jp, ChanFBO_P20_R3):
    bar = '█' * max(0, int(abs(ann) * 10)) if ann > 0 else '░' * max(0, int(abs(ann) * 10))
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

print('\nC. Parameter Sensitivity (channel period, RR & SL buffer)')
sep()
print(f"  {'Variant':<18} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [
    ('Period 15',    ChanFBO_P15_R3), ('Period 20 ★',   ChanFBO_P20_R3),
    ('Period 25',    ChanFBO_P25_R3), ('RR 2.0:1',      ChanFBO_P20_R2),
    ('RR 4.0:1',    ChanFBO_P20_R4), ('SL buf 0.25x',   ChanFBO_SL025),
    ('SL buf 0.75x', ChanFBO_SL075),
]:
    try:
        sh,ann,dd,wr,n,pf,_ = run(df_jp, cls)
        print(f"  {lbl:<18} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")
    except Exception as e:
        print(f"  {lbl:<18} error: {e}")


print('\n' + '═'*72)
print('SUMMARY — Add or Reject?')
print('═'*72)
print("""
Criteria to ADD:
  ✓ OOS Sharpe ≥ 0.3  (holds up out-of-sample)
  ✓ Profitable in ≥ 6/10 years  (consistent, not lucky)
  ✓ Parameters: Sh stays positive across RR & vol-mult variants
  ✗ Reject if OOS negative or only 3-4 profitable years

Note on trade count (N):
  UK100 VSF has low N (~22) — small sample, OOS verdict is most important.
  USD/CAD FBO has high N (~130) — more robust statistics.
""")
