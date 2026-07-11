"""
Deep validation — VSF 1×ATR candidates from vsf_atr_audit.py:
  1. FR40  — Vol Spike Fade 1×ATR, R2 & R3
  2. EU50  — Vol Spike Fade 1×ATR, R2 & R3

Tests:
  A. Walk-forward: 70% in-sample / 30% out-of-sample
  B. Year-by-year P&L consistency
  C. Parameter sensitivity (vol_mult, RR, SL ATR variants)
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
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values

class VolSpikeFade(Strategy):
    vol_mult  = 1.0
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
        h1,l1,c1 = float(self.data.High[-2]),float(self.data.Low[-2]),float(self.data.Close[-2])
        rng = h1 - l1
        if rng < self.vol_mult * av:
            self._wl = self._ws = False
        else:
            pos = (c1 - l1) / rng
            if pos <= self.close_pct:     self._wl,self._ws = True,False
            elif pos >= 1-self.close_pct: self._ws,self._wl = True,False
            else:                          self._wl = self._ws = False
        if self.position: self._wl = self._ws = False; return
        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT
        if self._wl:
            sl = p - self.sl_atr*av; sd = max(p-sl,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p + self.rr*sd
            if sl < p < tp: self.buy(size=u,sl=sl,tp=tp)
            self._wl = False
        elif self._ws:
            sl = p + self.sl_atr*av; sd = max(sl-p,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p - self.rr*sd
            if tp < p < sl: self.sell(size=u,sl=sl,tp=tp)
            self._ws = False

# Variants
class VSF_1x_R2(VolSpikeFade):   vol_mult=1.0; rr=2.0; sl_atr=1.5
class VSF_1x_R3(VolSpikeFade):   vol_mult=1.0; rr=3.0; sl_atr=1.5
class VSF_15x_R2(VolSpikeFade):  vol_mult=1.5; rr=2.0; sl_atr=1.5
class VSF_15x_R3(VolSpikeFade):  vol_mult=1.5; rr=3.0; sl_atr=1.5
class VSF_2x_R2(VolSpikeFade):   vol_mult=2.0; rr=2.0; sl_atr=1.5
class VSF_2x_R3(VolSpikeFade):   vol_mult=2.0; rr=3.0; sl_atr=1.5
class VSF_1x_R25(VolSpikeFade):  vol_mult=1.0; rr=2.5; sl_atr=1.5
class VSF_1x_R4(VolSpikeFade):   vol_mult=1.0; rr=4.0; sl_atr=1.5
class VSF_1x_SL1(VolSpikeFade):  vol_mult=1.0; rr=2.0; sl_atr=1.0
class VSF_1x_SL2(VolSpikeFade):  vol_mult=1.0; rr=2.0; sl_atr=2.0


def run(df, cls):
    bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                  margin=1/30, finalize_trades=True)
    s = bt.run()
    return (float(s.get('Sharpe Ratio',0) or 0),
            float(s.get('Return (Ann.) [%]',0) or 0),
            float(s.get('Max. Drawdown [%]',0) or 0),
            float(s.get('Win Rate [%]',0) or 0),
            int(s.get('# Trades',0) or 0),
            float(s.get('Profit Factor',0) or 0))

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
                float(s.get('Return (Ann.) [%]',0) or 0),
                int(s.get('# Trades',0) or 0),
                float(s.get('Win Rate [%]',0) or 0)))
        except Exception:
            results.append((y, 0.0, 0, 0.0))
    return results

def sep(): print('─'*72)


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 1: FR40 — Vol Spike Fade 1×ATR')
print('═'*72)

df_fr = load_oanda_data('FR40_EUR', period='10y', interval='1d')
split = int(len(df_fr) * 0.70)
df_train, df_oos = df_fr.iloc[:split], df_fr.iloc[split:]

print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}  [R2]")
sh,ann,dd,wr,n,pf = run(df_train, VSF_1x_R2)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_oos, VSF_1x_R2)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_fr, VSF_1x_R2)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print(f"\n  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}  [R3]")
sh,ann,dd,wr,n,pf = run(df_train, VSF_1x_R3)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_oos, VSF_1x_R3)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_fr, VSF_1x_R3)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print('\nB. Year-by-Year P&L  (R2)')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_fr, VSF_1x_R2):
    bar = '█' * max(0, int(abs(ann)*10)) if ann > 0 else '░' * max(0, int(abs(ann)*10))
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

print('\nC. Parameter Sensitivity')
sep()
print(f"  {'Variant':<18} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [
    ('1x R2 ★',    VSF_1x_R2),  ('1x R2.5',    VSF_1x_R25),
    ('1x R3',      VSF_1x_R3),  ('1x R4',       VSF_1x_R4),
    ('1.5x R2',    VSF_15x_R2), ('2x R2',       VSF_2x_R2),
    ('SL 1.0×ATR', VSF_1x_SL1), ('SL 2.0×ATR',  VSF_1x_SL2),
]:
    try:
        sh,ann,dd,wr,n,pf = run(df_fr, cls)
        print(f"  {lbl:<18} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")
    except Exception as e:
        print(f"  {lbl:<18} error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '═'*72)
print('CANDIDATE 2: EU50 — Vol Spike Fade 1×ATR')
print('═'*72)

df_eu = load_oanda_data('EU50_EUR', period='10y', interval='1d')
split = int(len(df_eu) * 0.70)
df_train, df_oos = df_eu.iloc[:split], df_eu.iloc[split:]

print('\nA. Walk-Forward (70% in-sample / 30% out-of-sample)')
sep()
print(f"  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}  [R2]")
sh,ann,dd,wr,n,pf = run(df_train, VSF_1x_R2)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_oos, VSF_1x_R2)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_eu, VSF_1x_R2)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print(f"\n  {'Period':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5}  [R3]")
sh,ann,dd,wr,n,pf = run(df_train, VSF_1x_R3)
print(f"  {'In-sample (7y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_oos, VSF_1x_R3)
print(f"  {'Out-of-sample (3y)':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")
sh,ann,dd,wr,n,pf = run(df_eu, VSF_1x_R3)
print(f"  {'Full 10y':<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}")

print('\nB. Year-by-Year P&L  (R2)')
sep()
print(f"  {'Year':<6} {'Ann%':>7} {'N':>5} {'WR%':>6}  Bar")
profitable = 0
for y, ann, n, wr in year_by_year(df_eu, VSF_1x_R2):
    bar = '█' * max(0, int(abs(ann)*10)) if ann > 0 else '░' * max(0, int(abs(ann)*10))
    sign = '+' if ann >= 0 else ''
    if ann > 0: profitable += 1
    print(f"  {y:<6} {sign}{ann:>5.1f}%  {n:>4}  {wr:>5.1f}%  {bar}")
print(f"  Profitable years: {profitable}/10")

print('\nC. Parameter Sensitivity')
sep()
print(f"  {'Variant':<18} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}")
for lbl, cls in [
    ('1x R2 ★',    VSF_1x_R2),  ('1x R2.5',    VSF_1x_R25),
    ('1x R3',      VSF_1x_R3),  ('1x R4',       VSF_1x_R4),
    ('1.5x R2',    VSF_15x_R2), ('2x R2',       VSF_2x_R2),
    ('SL 1.0×ATR', VSF_1x_SL1), ('SL 2.0×ATR',  VSF_1x_SL2),
]:
    try:
        sh,ann,dd,wr,n,pf = run(df_eu, cls)
        print(f"  {lbl:<18} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}")
    except Exception as e:
        print(f"  {lbl:<18} error: {e}")


print('\n' + '═'*72)
print('SUMMARY — Add or Reject?')
print('═'*72)
print("""
Criteria to ADD:
  ✓ OOS Sharpe ≥ 0.3
  ✓ Profitable in ≥ 6/10 years
  ✓ Parameters: Sh stays positive across vol_mult & RR variants
  ✗ Reject if OOS negative or only 3-4 profitable years
""")
