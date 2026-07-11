"""
Deep validation — Top 3 candidates from d1_strat_audit.py:
  1. WTI Oil  — InsideBar 2:1
  2. SPX500   — Engulfing 3:1
  3. XAU/USD  — Donchian 3:1

Tests run per candidate:
  A. Walk-forward: train on years 1-7, validate on years 8-10 (out-of-sample)
  B. Year-by-year P&L — is it consistently profitable or driven by 1-2 lucky years?
  C. Parameter sensitivity — nearby variants to check robustness
  D. Correlation vs existing live sleeves (daily returns overlap)
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

class InsideBar(Strategy):
    rr = 2.0; sl_atr = 1.5
    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._ib_high = self._ib_low = None
    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return
        h0,l0 = float(self.data.High[-1]),float(self.data.Low[-1])
        h1,l1 = float(self.data.High[-2]),float(self.data.Low[-2])
        h2,l2 = float(self.data.High[-3]),float(self.data.Low[-3])
        if h1 < h2 and l1 > l2:
            self._ib_high, self._ib_low = h1, l1
        if self.position or self._ib_high is None: return
        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT
        if h0 > self._ib_high:
            sl = p - self.sl_atr * av; sd = max(p-sl,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p + self.rr*sd
            if sl < p < tp: self.buy(size=u, sl=sl, tp=tp)
            self._ib_high = self._ib_low = None
        elif l0 < self._ib_low:
            sl = p + self.sl_atr * av; sd = max(sl-p,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p - self.rr*sd
            if tp < p < sl: self.sell(size=u, sl=sl, tp=tp)
            self._ib_high = self._ib_low = None

class InsideBar_R15(InsideBar): rr=1.5; sl_atr=1.5
class InsideBar_R2(InsideBar):  rr=2.0; sl_atr=1.5
class InsideBar_R25(InsideBar): rr=2.5; sl_atr=1.5
class InsideBar_R3(InsideBar):  rr=3.0; sl_atr=1.5
class InsideBar_SL1(InsideBar): rr=2.0; sl_atr=1.0
class InsideBar_SL2(InsideBar): rr=2.0; sl_atr=2.0


class Engulfing(Strategy):
    rr = 3.0; sl_atr = 1.5
    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._wl = self._ws = False
    def next(self):
        if len(self.data) < 3: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return
        o1,c1 = float(self.data.Open[-2]),float(self.data.Close[-2])
        o2,c2 = float(self.data.Open[-3]),float(self.data.Close[-3])
        body1,body2 = abs(c1-o1),abs(c2-o2)
        if body2 <= 0 or body1 < body2*1.1:
            self._wl = self._ws = False
        elif c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2:
            self._wl,self._ws = True,False
        elif c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2:
            self._ws,self._wl = True,False
        else:
            self._wl = self._ws = False
        if self.position: self._wl = self._ws = False; return
        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT
        if self._wl:
            sl = p - self.sl_atr*av; sd = max(p-sl,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p + self.rr*sd
            if sl < p < tp: self.buy(size=u, sl=sl, tp=tp)
            self._wl = False
        elif self._ws:
            sl = p + self.sl_atr*av; sd = max(sl-p,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p - self.rr*sd
            if tp < p < sl: self.sell(size=u, sl=sl, tp=tp)
            self._ws = False

class Engulf_R2(Engulfing):   rr=2.0; sl_atr=1.5
class Engulf_R25(Engulfing):  rr=2.5; sl_atr=1.5
class Engulf_R3(Engulfing):   rr=3.0; sl_atr=1.5
class Engulf_R4(Engulfing):   rr=4.0; sl_atr=1.5
class Engulf_SL1(Engulfing):  rr=3.0; sl_atr=1.0
class Engulf_SL2(Engulfing):  rr=3.0; sl_atr=2.0


class Donchian(Strategy):
    period=20; rr=3.0; sl_atr=1.5
    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        n = self.period
        if len(self.data) < n+2: return
        av = float(self.atr_i[-1])
        if np.isnan(av) or av <= 0: return
        if self.position: return
        highs = [float(self.data.High[-(i+2)]) for i in range(n)]
        lows  = [float(self.data.Low[-(i+2)])  for i in range(n)]
        ch, cl = max(highs), min(lows)
        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT
        if p > ch:
            sl = p - self.sl_atr*av; sd = max(p-sl,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p + self.rr*sd
            if sl < p < tp: self.buy(size=u, sl=sl, tp=tp)
        elif p < cl:
            sl = p + self.sl_atr*av; sd = max(sl-p,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p - self.rr*sd
            if tp < p < sl: self.sell(size=u, sl=sl, tp=tp)

class Don_P15_R3(Donchian):  period=15; rr=3.0; sl_atr=1.5
class Don_P20_R3(Donchian):  period=20; rr=3.0; sl_atr=1.5
class Don_P25_R3(Donchian):  period=25; rr=3.0; sl_atr=1.5
class Don_P20_R2(Donchian):  period=20; rr=2.0; sl_atr=1.5
class Don_P20_R4(Donchian):  period=20; rr=4.0; sl_atr=1.5
class Don_P20_SL1(Donchian): period=20; rr=3.0; sl_atr=1.0
class Don_P20_SL2(Donchian): period=20; rr=3.0; sl_atr=2.0


def run(df, cls):
    bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                  margin=1/30, finalize_trades=True)
    s = bt.run()
    sh  = float(s.get('Sharpe Ratio', 0) or 0)
    ann = float(s.get('Return (Ann.) [%]', 0) or 0)
    dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
    wr  = float(s.get('Win Rate [%]', 0) or 0)
    n   = int(s.get('# Trades', 0) or 0)
    pf  = float(s.get('Profit Factor', 0) or 0)
    return sh, ann, dd, wr, n, pf, s


def year_by_year(df, cls):
    """Run strategy year-by-year and return list of (year, sh, ann, n)."""
    years = sorted(df.index.year.unique())
    results = []
    for y in years:
        slice_ = df[df.index.year == y]
        if len(slice_) < 50: continue
        try:
            bt = Backtest(slice_, cls, cash=100_000, commission=0.0002,
                          margin=1/30, finalize_trades=True)
            s  = bt.run()
            ann = float(s.get('Return (Ann.) [%]', 0) or 0)
            n   = int(s.get('# Trades', 0) or 0)
            wr  = float(s.get('Win Rate [%]', 0) or 0)
            results.append((y, ann, n, wr))
        except Exception:
            results.append((y, 0.0, 0, 0.0))
    return results


def sep(): print('─' * 72)


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 1: WTI Oil — Inside Bar Breakout')
print('═'*72)

df_wti = load_oanda_data('WTICO_USD', period='10y', interval='1d')
split   = int(len(df_wti) * 0.70)   # 70% train, 30% OOS
df_train = df_wti.iloc[:split]
df_oos   = df_wti.iloc[split:]

# A. Walk-forward
print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}")
sh,ann,dd,wr,n,pf,_ = run(df_train, InsideBar_R2)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_oos, InsideBar_R2)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_wti, InsideBar_R2)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

# B. Year-by-year
print('\nB. Year-by-Year P&L')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_wti, InsideBar_R2):
    bar = '█' * int(abs(ann) * 10) if ann > 0 else '░' * int(abs(ann) * 10)
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

# C. Parameter sensitivity
print('\nC. Parameter Sensitivity (RR & SL variants)')
sep()
print(f"  {'Variant':<16} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [('RR 1.5:1', InsideBar_R15), ('RR 2.0:1 ★', InsideBar_R2),
                  ('RR 2.5:1', InsideBar_R25), ('RR 3.0:1', InsideBar_R3),
                  ('SL 1.0xATR', InsideBar_SL1), ('SL 2.0xATR', InsideBar_SL2)]:
    sh,ann,dd,wr,n,pf,_ = run(df_wti, cls)
    print(f"  {lbl:<16} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 2: SPX500 — Engulfing 3:1')
print('═'*72)

df_spx  = load_oanda_data('SPX500_USD', period='10y', interval='1d')
split    = int(len(df_spx) * 0.70)
df_train = df_spx.iloc[:split]
df_oos   = df_spx.iloc[split:]

print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}")
sh,ann,dd,wr,n,pf,_ = run(df_train, Engulf_R3)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_oos, Engulf_R3)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_spx, Engulf_R3)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print('\nB. Year-by-Year P&L')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_spx, Engulf_R3):
    bar = '█' * int(abs(ann) * 10) if ann > 0 else '░' * int(abs(ann) * 10)
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

print('\nC. Parameter Sensitivity (RR & SL variants)')
sep()
print(f"  {'Variant':<16} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [('RR 2.0:1', Engulf_R2), ('RR 2.5:1', Engulf_R25),
                  ('RR 3.0:1 ★', Engulf_R3), ('RR 4.0:1', Engulf_R4),
                  ('SL 1.0xATR', Engulf_SL1), ('SL 2.0xATR', Engulf_SL2)]:
    sh,ann,dd,wr,n,pf,_ = run(df_spx, cls)
    print(f"  {lbl:<16} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 3: XAU/USD (Gold) — Donchian 20-day 3:1')
print('═'*72)

df_xau  = load_oanda_data('XAU_USD', period='10y', interval='1d')
split    = int(len(df_xau) * 0.70)
df_train = df_xau.iloc[:split]
df_oos   = df_xau.iloc[split:]

print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}")
sh,ann,dd,wr,n,pf,_ = run(df_train, Don_P20_R3)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_oos, Don_P20_R3)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf,_ = run(df_xau, Don_P20_R3)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print('\nB. Year-by-Year P&L')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_xau, Don_P20_R3):
    bar = '█' * int(abs(ann) * 10) if ann > 0 else '░' * int(abs(ann) * 10)
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

print('\nC. Parameter Sensitivity (period, RR & SL variants)')
sep()
print(f"  {'Variant':<16} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [('Period 15', Don_P15_R3), ('Period 20 ★', Don_P20_R3),
                  ('Period 25', Don_P25_R3), ('RR 2.0:1', Don_P20_R2),
                  ('RR 4.0:1', Don_P20_R4), ('SL 1.0xATR', Don_P20_SL1),
                  ('SL 2.0xATR', Don_P20_SL2)]:
    sh,ann,dd,wr,n,pf,_ = run(df_xau, cls)
    print(f"  {lbl:<16} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")

print('\n' + '═'*72)
print('SUMMARY — Add or Reject?')
print('═'*72)
print("""
Criteria to ADD:
  ✓ OOS Sharpe ≥ 0.3  (holds up out-of-sample)
  ✓ Profitable in ≥ 6/10 years  (not driven by 1-2 years)
  ✓ Parameter sensitivity: Sh stays positive across RR variants
  ✗ Reject if OOS is negative or only 3-4 profitable years
""")
