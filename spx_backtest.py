from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

def _ema(arr, n):
    return pd.Series(np.array(arr, dtype=float)).ewm(span=n, adjust=False).mean().values

def _atr(high, low, close, period=14):
    h,l,c = np.array(high),np.array(low),np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return pd.Series(np.concatenate([[tr[0]], tr])).ewm(span=period, adjust=False).mean().values

# ── PDHL H1 ──────────────────────────────────────────────────────────────────
class PDHL(Strategy):
    risk_pct=0.0025; rr=1.5
    def init(self):
        self.prev_high=np.nan; self.prev_low=np.nan
        self.last_date=None; self.day_high=-np.inf; self.day_low=np.inf
    def next(self):
        t=self.data.index[-1]; date=t.date()
        if date!=self.last_date:
            if self.last_date is not None:
                self.prev_high=self.day_high; self.prev_low=self.day_low
            self.day_high=float(self.data.High[-1]); self.day_low=float(self.data.Low[-1])
            self.last_date=date
        else:
            self.day_high=max(self.day_high,float(self.data.High[-1]))
            self.day_low=min(self.day_low,float(self.data.Low[-1]))
        if np.isnan(self.prev_high) or self.position: return
        price=float(self.data.Close[-1])
        prev_mid=(self.prev_high+self.prev_low)/2; prev_rng=self.prev_high-self.prev_low
        if prev_rng<=0: return
        ra=self.equity*self.risk_pct
        if price>self.prev_high:
            sd=price-prev_mid
            if sd<=0: return
            self.buy(size=max(1,int(ra/sd)),sl=prev_mid,tp=price+self.rr*prev_rng)
        elif price<self.prev_low:
            sd=prev_mid-price
            if sd<=0: return
            self.sell(size=max(1,int(ra/sd)),sl=prev_mid,tp=price-self.rr*prev_rng)

# ── ORB M30 ───────────────────────────────────────────────────────────────────
class ORB(Strategy):
    risk_pct=0.0025; rr=1.5; session_hour=8
    def init(self):
        self.or_high=np.nan; self.or_low=np.nan; self.or_set=False; self.last_date=None
    def next(self):
        t=self.data.index[-1]; hour=t.hour; date=t.date()
        if date!=self.last_date:
            self.or_high=np.nan; self.or_low=np.nan; self.or_set=False; self.last_date=date
        if hour==self.session_hour and not self.or_set:
            self.or_high=float(self.data.High[-1]); self.or_low=float(self.data.Low[-1])
            self.or_set=True; return
        if not self.or_set or np.isnan(self.or_high) or self.position: return
        session_end=(self.session_hour+7)%24
        in_window=(self.session_hour<hour<=session_end) if session_end>self.session_hour else (hour>self.session_hour or hour<=session_end)
        if not in_window: return
        price=float(self.data.Close[-1]); or_rng=self.or_high-self.or_low
        if or_rng<=0: return
        ra=self.equity*self.risk_pct
        if price>self.or_high:
            sd=price-self.or_low
            if sd<=0: return
            self.buy(size=max(1,int(ra/sd)),sl=self.or_low,tp=price+self.rr*or_rng)
        elif price<self.or_low:
            sd=self.or_high-price
            if sd<=0: return
            self.sell(size=max(1,int(ra/sd)),sl=self.or_high,tp=price-self.rr*or_rng)

# ── Consec D1 Mean Reversion ─────────────────────────────────────────────────
class ConsecReversion(Strategy):
    streak=3; sl_atr=1.5; tp_atr=1.0; risk_pct=0.0025
    def init(self):
        self.atr=self.I(_atr,self.data.High,self.data.Low,self.data.Close,14)
    def next(self):
        if self.position or len(self.data.Close)<self.streak+1: return
        closes=[float(self.data.Close[-(i+1)]) for i in range(self.streak+1)]
        downs=all(closes[i]<closes[i+1] for i in range(self.streak))
        ups=all(closes[i]>closes[i+1] for i in range(self.streak))
        price=float(self.data.Close[-1]); atr=float(self.atr[-1])
        if atr<=0 or np.isnan(atr): return
        ra=self.equity*self.risk_pct; sd=self.sl_atr*atr
        size=max(1,int(ra/sd))
        if downs:
            self.buy(size=size,sl=price-sd,tp=price+self.tp_atr*atr)
        elif ups:
            self.sell(size=size,sl=price+sd,tp=price-self.tp_atr*atr)

# ── EMA H1 crossover ─────────────────────────────────────────────────────────
class EMAcross(Strategy):
    fast=9; slow=21; sl_atr=2.0; risk_pct=0.0025; trail_mult=2.0
    def init(self):
        self.fast_e=self.I(_ema,self.data.Close,self.fast)
        self.slow_e=self.I(_ema,self.data.Close,self.slow)
        self.atr=self.I(_atr,self.data.High,self.data.Low,self.data.Close,14)
        self.peak=np.nan; self.trough=np.nan
    def next(self):
        atr=float(self.atr[-1]); price=float(self.data.Close[-1])
        if atr<=0 or np.isnan(atr): return
        if self.position.is_long:
            self.peak=max(self.peak,price)
            if price<self.peak-self.trail_mult*atr: self.position.close(); self.peak=np.nan
            return
        if self.position.is_short:
            self.trough=min(self.trough,price)
            if price>self.trough+self.trail_mult*atr: self.position.close(); self.trough=np.nan
            return
        f,fp=float(self.fast_e[-1]),float(self.fast_e[-2])
        s,sp=float(self.slow_e[-1]),float(self.slow_e[-2])
        ra=self.equity*self.risk_pct; sd=self.sl_atr*atr; sz=max(1,int(ra/sd))
        if f>s and fp<=sp: self.peak=price; self.buy(size=sz,sl=price-sd)
        elif f<s and fp>=sp: self.trough=price; self.sell(size=sz,sl=price+sd)

def run(df, cls, **kw):
    bt=Backtest(df,cls,cash=100_000,commission=0.0002,margin=1/30,finalize_trades=True)
    s=bt.run(**kw)
    sh=float(s.get('Sharpe Ratio',0) or 0)
    ar=float(s.get('Return (Ann.) [%]',0) or 0)
    dd=float(s.get('Max. Drawdown [%]',0) or 0)
    wr=float(s.get('Win Rate [%]',0) or 0)
    n=int(s.get('# Trades',0) or 0)
    return sh,ar,dd,wr,n

print('\nSPX500 Strategy Backtest — 10y, 0.25% risk/trade')
print('='*62)
print(f"{'Strategy':<22} {'TF':<5} {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>5} {'N':>5}  Verdict")
print('-'*62)

periods = ['2y','3y','5y','7y','10y']

tests = [
    ('PDHL H1',         PDHL,           '1h',  {}),
    ('ORB London 08:00',ORB,            '30m', {'session_hour':8}),
    ('ORB NY 13:00',    ORB,            '30m', {'session_hour':13}),
    ('ConsecRev D1',    ConsecReversion,'1d',  {}),
    ('EMA cross H1',    EMAcross,       '1h',  {}),
]

for label, cls, tf, kw in tests:
    df = load_oanda_data('SPX500_USD', period='10y', interval=tf)
    sh,ar,dd,wr,n = run(df,cls,**kw)
    flag='YES' if sh>=1.0 and ar>0 else ('WATCH' if sh>=0.5 and ar>0 else '---')
    print(f'{label:<22} {tf:<5} {sh:>5.2f} {ar:>5.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}  {flag}')

# Cross-period check for PDHL (already live)
print()
print('PDHL H1 cross-period validation:')
for p in periods:
    df=load_oanda_data('SPX500_USD',period=p,interval='1h')
    sh,ar,dd,wr,n=run(df,PDHL)
    flag='OK' if sh>=0.5 and ar>0 else 'WEAK'
    print(f'  {p:>4}: Sh={sh:.2f}  Ann={ar:+.1f}%  DD={dd:.1f}%  WR={wr:.1f}%  N={n}  {flag}')
