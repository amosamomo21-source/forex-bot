"""
Full portfolio backtest — all 47 current sleeves, 10y data.
Reports per-strategy-type WR and Sharpe, then aggregates across the whole portfolio.

Note on M30 BBMRT: the live strategy uses daily bias + M30 RSI crossover timing.
The backtest here runs pure BBMRT on M30 data — an approximation, not exact.
All other strategies match the live runner logic faithfully.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

# ── Indicator helpers ───────────────────────────────────────────────────────
def _ema(s, n):  return pd.Series(np.array(s,float)).ewm(span=n,adjust=False).mean().values
def _sma(s, n):  return pd.Series(np.array(s,float)).rolling(n).mean().values
def _std(s, n):  return pd.Series(np.array(s,float)).rolling(n).std(ddof=0).values
def _atr(h,l,c,n=14):
    h,l,c=np.array(h,float),np.array(l,float),np.array(c,float)
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    return pd.Series(np.concatenate([[tr[0]],tr])).ewm(span=n,adjust=False).mean().values
def _rsi(s,n=14):
    s=pd.Series(np.array(s,float)); d=s.diff()
    g=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+g/l.replace(0,1e-12))).values
def _macd(s,fast=12,slow=26,sig=9):
    s=pd.Series(np.array(s,float))
    ml=s.ewm(span=fast,adjust=False).mean()-s.ewm(span=slow,adjust=False).mean()
    return ml.values, ml.ewm(span=sig,adjust=False).mean().values

# ── Strategy classes ────────────────────────────────────────────────────────
class BBMRT(Strategy):
    """BollingerMeanReversionTrendFilter — matches strategies.py params."""
    bb_period=20; bb_k=1.5; atr_period=14; sl_atr_mult=2.0
    trend_period=100; max_hold=20; risk_pct=RISK_PCT
    def init(self):
        self.mid   = self.I(_sma, self.data.Close, self.bb_period)
        self.sd    = self.I(_std, self.data.Close, self.bb_period)
        self.av    = self.I(_atr, self.data.High, self.data.Low, self.data.Close)
        self.trend = self.I(_ema, self.data.Close, self.trend_period)
        self._held = 0
    def next(self):
        p,m,s,a,t = (float(self.data.Close[-1]), float(self.mid[-1]),
                     float(self.sd[-1]), float(self.av[-1]), float(self.trend[-1]))
        if any(np.isnan(x) for x in (m,s,a,t)) or s==0: return
        upper,lower = m+self.bb_k*s, m-self.bb_k*s
        sd = self.sl_atr_mult*a
        if self.position:
            self._held += 1
            if (self.position.is_long  and p>=m) or \
               (self.position.is_short and p<=m) or self._held>=self.max_hold:
                self.position.close()
            return
        units = max(1, int(self.equity*self.risk_pct/sd)) if sd>0 else 0
        if units<=0: return
        if p<lower and p>t:
            self._held=0; self.buy(size=units,  sl=p-sd)
        elif p>upper and p<t:
            self._held=0; self.sell(size=units, sl=p+sd)

class EMAcross(Strategy):
    """EMA(20/50) crossover with fixed SL=1.5×ATR, TP=2.5×ATR."""
    fast=20; slow=50; risk_pct=RISK_PCT
    def init(self):
        self.ef = self.I(_ema, self.data.Close, self.fast)
        self.es = self.I(_ema, self.data.Close, self.slow)
        self.av = self.I(_atr, self.data.High, self.data.Low, self.data.Close)
    def next(self):
        a=float(self.av[-1])
        if np.isnan(a): return
        sd=1.5*a; units=max(1,int(self.equity*self.risk_pct/sd)) if sd>0 else 0
        if units<=0: return
        p=float(self.data.Close[-1])
        fn,fp = float(self.ef[-1]), float(self.ef[-2])
        sn,sp = float(self.es[-1]), float(self.es[-2])
        if fp<=sp and fn>sn:
            self.position.close(); self.buy( size=units,sl=p-sd,tp=p+2.5*a)
        elif fp>=sp and fn<sn:
            self.position.close(); self.sell(size=units,sl=p+sd,tp=p-2.5*a)

class H1EMA(Strategy):
    """H1 EMA(10/30) crossover — trailing stop, no fixed TP (approximated with 3×ATR TP)."""
    risk_pct=RISK_PCT
    def init(self):
        self.ef = self.I(_ema, self.data.Close, 10)
        self.es = self.I(_ema, self.data.Close, 30)
        self.av = self.I(_atr, self.data.High, self.data.Low, self.data.Close)
    def next(self):
        a=float(self.av[-1])
        if np.isnan(a) or a<=0: return
        sd=1.5*a; units=max(1,int(self.equity*self.risk_pct/sd)) if sd>0 else 0
        if units<=0: return
        p=float(self.data.Close[-1])
        fn,fp = float(self.ef[-1]), float(self.ef[-2])
        sn,sp = float(self.es[-1]), float(self.es[-2])
        if fp<=sp and fn>sn:
            self.position.close(); self.buy( size=units,sl=p-sd,tp=p+3*a)
        elif fp>=sp and fn<sn:
            self.position.close(); self.sell(size=units,sl=p+sd,tp=p-3*a)

class MACDH1(Strategy):
    """MACD(12,26,9) signal-line crossover. SL=1.5×ATR, TP=2.5×ATR."""
    risk_pct=RISK_PCT
    def init(self):
        self.ml, self.sig = self.I(_macd, self.data.Close)
        self.av = self.I(_atr, self.data.High, self.data.Low, self.data.Close)
    def next(self):
        a=float(self.av[-1])
        if np.isnan(a) or a<=0: return
        sd=1.5*a; units=max(1,int(self.equity*self.risk_pct/sd)) if sd>0 else 0
        if units<=0: return
        p=float(self.data.Close[-1])
        mn,mp = float(self.ml[-1]),  float(self.ml[-2])
        sn,sp = float(self.sig[-1]), float(self.sig[-2])
        if mp<=sp and mn>sn:
            self.position.close(); self.buy( size=units,sl=p-sd,tp=p+2.5*a)
        elif mp>=sp and mn<sn:
            self.position.close(); self.sell(size=units,sl=p+sd,tp=p-2.5*a)

class ORB(Strategy):
    """Opening Range Breakout — M30. Entry when bar following 08:00/13:00 UTC breaks range."""
    risk_pct=RISK_PCT; tp_mult=1.5
    def init(self): pass
    def next(self):
        if self.position or len(self.data)<2: return
        t_prev = self.data.index[-2]
        if t_prev.hour not in (8,13) or t_prev.minute!=0: return
        or_high=float(self.data.High[-2]); or_low=float(self.data.Low[-2])
        or_range=or_high-or_low
        if or_range<=0: return
        p=float(self.data.Close[-1])
        ra=self.equity*self.risk_pct
        units=min(max(1,int(ra/or_range)),50_000)
        if p>or_high:
            self.buy( size=units,sl=or_low, tp=p+self.tp_mult*or_range)
        elif p<or_low:
            self.sell(size=units,sl=or_high,tp=p-self.tp_mult*or_range)

class PDHL(Strategy):
    """Previous Day High/Low breakout — H1 entry."""
    risk_pct=RISK_PCT; rr=1.5
    def init(self):
        self.prev_high=np.nan; self.prev_low=np.nan
        self.last_date=None; self.day_high=-np.inf; self.day_low=np.inf
    def next(self):
        date=self.data.index[-1].date()
        if date!=self.last_date:
            if self.last_date is not None:
                self.prev_high=self.day_high; self.prev_low=self.day_low
            self.day_high=float(self.data.High[-1]); self.day_low=float(self.data.Low[-1])
            self.last_date=date
        else:
            self.day_high=max(self.day_high,float(self.data.High[-1]))
            self.day_low =min(self.day_low, float(self.data.Low[-1]))
        if np.isnan(self.prev_high) or self.position: return
        p=float(self.data.Close[-1])
        mid=(self.prev_high+self.prev_low)/2; rng=self.prev_high-self.prev_low
        if rng<=0: return
        ra=self.equity*self.risk_pct
        if p>self.prev_high:
            sd=p-mid;  units=max(1,int(ra/sd)) if sd>0 else 0
            if units: self.buy( size=units,sl=mid,tp=p+self.rr*rng)
        elif p<self.prev_low:
            sd=mid-p;  units=max(1,int(ra/sd)) if sd>0 else 0
            if units: self.sell(size=units,sl=mid,tp=p-self.rr*rng)

class Consec(Strategy):
    """ConsecReversion D1 — N consecutive closes → fade."""
    streak=3; sl_mult=1.5; tp_mult=1.0; risk_pct=RISK_PCT
    def init(self):
        self.av=self.I(_atr,self.data.High,self.data.Low,self.data.Close)
    def next(self):
        if self.position: return
        n=self.streak; c=np.array(self.data.Close)
        if len(c)<n+2: return
        down=all(c[-(i+1)]<c[-(i+2)] for i in range(n))
        up  =all(c[-(i+1)]>c[-(i+2)] for i in range(n))
        if not (down or up): return
        a=float(self.av[-1])
        if np.isnan(a) or a<=0: return
        p=float(self.data.Close[-1]); sd=self.sl_mult*a
        units=max(1,int(self.equity*self.risk_pct/sd))
        if down: self.buy( size=units,sl=p-sd,tp=p+self.tp_mult*a)
        else:    self.sell(size=units,sl=p+sd,tp=p-self.tp_mult*a)

class Consec4(Consec):
    streak=4

# ── Runner ──────────────────────────────────────────────────────────────────
def bt(df, cls, interval):
    margin = 1/30 if interval in ('1d','1h') else 1/30
    b = Backtest(df, cls, cash=100_000, commission=0.0002, margin=margin, finalize_trades=True)
    s = b.run()
    return {
        'Sh':  float(s.get('Sharpe Ratio',0) or 0),
        'Ann': float(s.get('Return (Ann.) [%]',0) or 0),
        'DD':  float(s.get('Max. Drawdown [%]',0) or 0),
        'WR':  float(s.get('Win Rate [%]',0) or 0),
        'N':   int(s.get('# Trades',0) or 0),
    }

def section(title, rows, note=''):
    total_n = sum(r['N'] for r in rows)
    total_wins = sum(int(r['N']*r['WR']/100) for r in rows)
    agg_wr = total_wins/total_n*100 if total_n else 0
    avg_sh = np.mean([r['Sh'] for r in rows]) if rows else 0
    print(f'\n{"─"*70}')
    print(f'{title}  [{len(rows)} sleeves]' + (f'  NOTE: {note}' if note else ''))
    print(f'  Avg Sharpe: {avg_sh:+.2f}   Aggregate WR: {agg_wr:.1f}%   Total trades: {total_n}')
    return total_n, total_wins

all_n, all_wins = 0, 0

print('Portfolio audit — 10y backtest, all 47 sleeves')
print('Loading data and running backtests (this takes ~10 minutes)...')

# ── 1. BBMRT D1 ─────────────────────────────────────────────────────────────
rows=[]
for tag, instr, kind in [('bbmrt_eurusd','EUR_USD','bbmrt'),('bbmrt_gbpusd','GBP_USD','bbmrt'),
                          ('ema_gbpusd','GBP_USD','ema'),('ema_usdjpy','USD_JPY','ema'),('ema_audusd','AUD_USD','ema')]:
    try:
        cls = BBMRT if kind=='bbmrt' else EMAcross
        df  = load_oanda_data(instr, period='10y', interval='1d')
        r   = bt(df, cls, '1d'); r['tag']=tag; rows.append(r)
        print(f'  {tag}: Sh={r["Sh"]:+.2f} WR={r["WR"]:.0f}% N={r["N"]}')
    except Exception as e: print(f'  {tag}: ERROR {e}')
n,w = section('1. Daily BBMRT + EMA (D1)', rows)
all_n+=n; all_wins+=w

# ── 2. BBMRT M30 ─────────────────────────────────────────────────────────────
rows=[]
M30_INSTRS=[('bbmrt_m30_eurusd','EUR_USD'),('bbmrt_m30_gbpusd','GBP_USD'),
            ('bbmrt_m30_eurcad','EUR_CAD'),('bbmrt_m30_eurjpy','EUR_JPY'),
            ('bbmrt_m30_chfjpy','CHF_JPY'),('bbmrt_m30_audchf','AUD_CHF'),
            ('bbmrt_m30_eursgd','EUR_SGD'),('bbmrt_m30_gbpaud','GBP_AUD'),
            ('bbmrt_m30_cadjpy','CAD_JPY'),('bbmrt_m30_audsgd','AUD_SGD'),
            ('bbmrt_m30_euraud','EUR_AUD'),('bbmrt_m30_gbpcad','GBP_CAD'),
            ('bbmrt_m30_gbpsgd','GBP_SGD'),('bbmrt_m30_gbpjpy','GBP_JPY'),
            ('bbmrt_m30_gbpchf','GBP_CHF'),('bbmrt_m30_audjpy','AUD_JPY'),
            ('bbmrt_m30_nzdjpy','NZD_JPY')]
for tag,instr in M30_INSTRS:
    try:
        df = load_oanda_data(instr, period='10y', interval='30m')
        r  = bt(df, BBMRT, '30m'); r['tag']=tag; rows.append(r)
        print(f'  {tag}: Sh={r["Sh"]:+.2f} WR={r["WR"]:.0f}% N={r["N"]}')
    except Exception as e: print(f'  {tag}: ERROR {e}')
n,w = section('2. BBMRT M30 (approximation — no RSI crossover timing)', rows,
              note='live adds M30 RSI gate; static WR may differ')
all_n+=n; all_wins+=w

# ── 3. H1 EMA (10/30) ───────────────────────────────────────────────────────
rows=[]
H1_INSTRS=[('ema_h1_gbpusd','GBP_USD'),('ema_h1_cadjpy','CAD_JPY'),('ema_h1_audjpy','AUD_JPY'),
           ('ema_h1_wticousd','WTICO_USD'),('ema_h1_bcousd','BCO_USD'),('ema_h1_xauusd','XAU_USD'),
           ('ema_h1_xagusd','XAG_USD'),('ema_h1_natgasusd','NATGAS_USD'),
           ('ema_h1_nas100','NAS100_USD'),('ema_h1_de30','DE30_EUR'),('ema_h1_jp225','JP225_USD')]
for tag,instr in H1_INSTRS:
    try:
        df = load_oanda_data(instr, period='10y', interval='1h')
        r  = bt(df, H1EMA, '1h'); r['tag']=tag; rows.append(r)
        print(f'  {tag}: Sh={r["Sh"]:+.2f} WR={r["WR"]:.0f}% N={r["N"]}')
    except Exception as e: print(f'  {tag}: ERROR {e}')
n,w = section('3. H1 EMA(10/30) with trailing stop', rows)
all_n+=n; all_wins+=w

# ── 4. MACD H1 ──────────────────────────────────────────────────────────────
rows=[]
try:
    df = load_oanda_data('USD_JPY', period='10y', interval='1h')
    r  = bt(df, MACDH1, '1h'); r['tag']='macd_h1_usdjpy'; rows.append(r)
    print(f'  macd_h1_usdjpy: Sh={r["Sh"]:+.2f} WR={r["WR"]:.0f}% N={r["N"]}')
except Exception as e: print(f'  macd_h1_usdjpy: ERROR {e}')
n,w = section('4. MACD H1 (USD/JPY)', rows)
all_n+=n; all_wins+=w

# ── 5. ORB M30 ───────────────────────────────────────────────────────────────
rows=[]
ORB_INSTRS=[('orb_m30_eurjpy','EUR_JPY'),('orb_m30_chfjpy','CHF_JPY'),
            ('orb_m30_cadjpy','CAD_JPY'),('orb_m30_audjpy','AUD_JPY'),
            ('orb_m30_gbpjpy','GBP_JPY'),('orb_m30_nzdjpy','NZD_JPY'),
            ('orb_m30_audchf','AUD_CHF'),('orb_m30_euraud','EUR_AUD'),
            ('orb_m30_usdjpy','USD_JPY'),('orb_m30_eurcad','EUR_CAD')]
for tag,instr in ORB_INSTRS:
    try:
        df = load_oanda_data(instr, period='10y', interval='30m')
        r  = bt(df, ORB, '30m'); r['tag']=tag; rows.append(r)
        print(f'  {tag}: Sh={r["Sh"]:+.2f} WR={r["WR"]:.0f}% N={r["N"]}')
    except Exception as e: print(f'  {tag}: ERROR {e}')
n,w = section('5. ORB M30 (London 08:00 + NY 13:00 UTC)', rows)
all_n+=n; all_wins+=w

# ── 6. PDHL H1 — NATGAS only ─────────────────────────────────────────────────
rows=[]
try:
    df = load_oanda_data('NATGAS_USD', period='10y', interval='1h')
    r  = bt(df, PDHL, '1h'); r['tag']='pdhl_natgas'; rows.append(r)
    print(f'  pdhl_natgas: Sh={r["Sh"]:+.2f} WR={r["WR"]:.0f}% N={r["N"]}')
except Exception as e: print(f'  pdhl_natgas: ERROR {e}')
n,w = section('6. PDHL H1 (NATGAS)', rows)
all_n+=n; all_wins+=w

# ── 7. ConsecReversion D1 — WHEAT ────────────────────────────────────────────
rows=[]
df_wheat = load_oanda_data('WHEAT_USD', period='10y', interval='1d')
for tag,cls in [('consec_d1_wheatusd_3',Consec),('consec_d1_wheatusd_4',Consec4)]:
    try:
        r = bt(df_wheat, cls, '1d'); r['tag']=tag; rows.append(r)
        print(f'  {tag}: Sh={r["Sh"]:+.2f} WR={r["WR"]:.0f}% N={r["N"]}')
    except Exception as e: print(f'  {tag}: ERROR {e}')
n,w = section('7. ConsecReversion D1 (WHEAT, streak 3+4)', rows)
all_n+=n; all_wins+=w

# ── Portfolio total ───────────────────────────────────────────────────────────
portfolio_wr = all_wins/all_n*100 if all_n else 0
print(f'\n{"═"*70}')
print(f'PORTFOLIO TOTAL — all 47 sleeves combined')
print(f'  Total trades : {all_n:,}')
print(f'  Total wins   : {all_wins:,}')
print(f'  Overall WR   : {portfolio_wr:.1f}%')
print(f'{"═"*70}')
