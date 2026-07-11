from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd, sys

def _ema(arr, n):
    return pd.Series(np.array(arr, dtype=float)).ewm(span=n, adjust=False).mean().values
def _atr(high, low, close, period=14):
    h,l,c = np.array(high),np.array(low),np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

class ConsecReversion(Strategy):
    streak=3; sl_atr=1.5; tp_atr=1.5; risk_pct=0.0025; trend_ema=0
    def init(self):
        cl,hi,lo=self.data.Close,self.data.High,self.data.Low
        self.atr=self.I(_atr,hi,lo,cl,14,name='ATR')
        self.ema_v=self.I(_ema,cl,self.trend_ema,name='EMA') if self.trend_ema>0 else None
    def next(self):
        if self.position: return
        closes=self.data.Close
        if len(closes)<self.streak+1: return
        atr=float(self.atr[-1])
        if atr<=0 or np.isnan(atr): return
        down=all(float(closes[-(i+1)])<float(closes[-(i+2)]) for i in range(self.streak))
        up=all(float(closes[-(i+1)])>float(closes[-(i+2)]) for i in range(self.streak))
        price=float(closes[-1])
        tu=td=True
        if self.trend_ema>0 and self.ema_v is not None:
            e=float(self.ema_v[-1]); tu=price>e; td=price<e
        ra=self.equity*self.risk_pct; sd=self.sl_atr*atr; sz=max(1,int(ra/sd))
        if down and tu: self.buy(size=sz,sl=price-sd,tp=price+self.tp_atr*atr)
        elif up and td: self.sell(size=sz,sl=price+sd,tp=price-self.tp_atr*atr)

class BBReversion(Strategy):
    bb_per=20; bb_std=2.0; sl_atr=1.5; tp_mult=1.0; risk_pct=0.0025; trend_ema=0
    def init(self):
        cl,hi,lo=self.data.Close,self.data.High,self.data.Low
        self.atr=self.I(_atr,hi,lo,cl,14,name='ATR')
        sma=self.I(lambda c:pd.Series(np.array(c,dtype=float)).rolling(self.bb_per).mean().values,cl,name='SMA')
        std=self.I(lambda c:pd.Series(np.array(c,dtype=float)).rolling(self.bb_per).std().values,cl,name='STD')
        self.upper=self.I(lambda s,d:np.array(s,dtype=float)+self.bb_std*np.array(d,dtype=float),sma,std,name='BBU')
        self.lower=self.I(lambda s,d:np.array(s,dtype=float)-self.bb_std*np.array(d,dtype=float),sma,std,name='BBL')
        self.sma=sma
        self.ema_v=self.I(_ema,cl,self.trend_ema,name='EMA') if self.trend_ema>0 else None
    def next(self):
        if self.position: return
        price=float(self.data.Close[-1]); atr=float(self.atr[-1])
        upper=float(self.upper[-1]); lower=float(self.lower[-1]); mid=float(self.sma[-1])
        if atr<=0 or np.isnan(atr) or np.isnan(upper) or np.isnan(lower): return
        tu=td=True
        if self.trend_ema>0 and self.ema_v is not None:
            e=float(self.ema_v[-1]); tu=price>e; td=price<e
        ra=self.equity*self.risk_pct; sd=self.sl_atr*atr; sz=max(1,int(ra/sd))
        if price<lower and tu:
            tp=price+self.tp_mult*(mid-lower)
            self.buy(size=sz,sl=price-sd,tp=max(tp,price+1e-5))
        elif price>upper and td:
            tp=price-self.tp_mult*(upper-mid)
            self.sell(size=sz,sl=price+sd,tp=min(tp,price-1e-5))

def run(df, cls, **kw):
    bt=Backtest(df,cls,cash=100_000,commission=0.0002,margin=1/30,finalize_trades=True)
    s=bt.run(**kw)
    return (float(s.get('Sharpe Ratio',0) or 0), float(s.get('Return (Ann.) [%]',0) or 0),
            float(s.get('Max. Drawdown [%]',0) or 0), float(s.get('Win Rate [%]',0) or 0),
            int(s.get('# Trades',0) or 0))

# PART 1: WHEAT sweep
print('PART 1: WHEAT consecutive-day reversion parameter sweep')
print(f"{'Variant':<28} {'Sh':>5} {'Ann':>7} {'DD':>7} {'WR':>5} {'N':>5}  Verdict")
print('-'*62)
wheat = load_oanda_data('WHEAT_USD', period='10y', interval='1d')
for streak in [2,3,4,5]:
    for tp in [1.0,1.5,2.0,2.5]:
        for ema in [0,200]:
            lbl = 'S={} TP={}x {}'.format(streak, tp, 'EMA200' if ema else 'no-flt')
            try:
                sh,ar,dd,wr,n = run(wheat, ConsecReversion, streak=streak, tp_atr=tp, trend_ema=ema)
                if sh < 0.5: continue
                flag='✅' if sh>=1.0 and ar>0 else '⚠'
                print('{:<28} {:>5.2f} {:>6.1f}% {:>6.1f}% {:>5.1f}% {:>5}  {}'.format(lbl,sh,ar,dd,wr,n,flag))
            except Exception as e:
                print('ERR {}: {}'.format(lbl, e))
                import traceback; traceback.print_exc()

# PART 2: Agri commodities consec-3
print()
print('PART 2: Consec-3 reversion — commodities')
print(f"{'Instr':<10} {'Sh':>5} {'Ann':>7} {'DD':>7} {'WR':>5} {'N':>5}  Verdict")
print('-'*50)
for lbl,instr in [('WHEAT','WHEAT_USD'),('CORN','CORN_USD'),('SOYBN','SOYBN_USD'),
                   ('SUGAR','SUGAR_USD'),('NATGAS','NATGAS_USD'),('BCO','BCO_USD'),('WTICO','WTICO_USD')]:
    try:
        df=load_oanda_data(instr,period='10y',interval='1d')
        sh,ar,dd,wr,n=run(df,ConsecReversion,streak=3,tp_atr=1.5,trend_ema=0)
        flag='✅ ADD' if sh>=1.0 and ar>0 else ('⚠  WATCH' if sh>=0.5 and ar>0 else '❌')
        print('{:<10} {:>5.2f} {:>6.1f}% {:>6.1f}% {:>5.1f}% {:>5}  {}'.format(lbl,sh,ar,dd,wr,n,flag))
    except Exception as e:
        print('{:<10} ERR: {}'.format(lbl, e))

# PART 3: BB reversion EMA200 on FX
print()
print('PART 3: BB Reversion + EMA200 — FX pairs')
print(f"{'Instr+BB':<16} {'Sh':>5} {'Ann':>7} {'DD':>7} {'WR':>5} {'N':>5}  Verdict")
print('-'*56)
fx=[('EUR/USD','EUR_USD'),('GBP/USD','GBP_USD'),('AUD/USD','AUD_USD'),
    ('NZD/USD','NZD_USD'),('USD/CHF','USD_CHF'),
    ('EUR/JPY','EUR_JPY'),('GBP/JPY','GBP_JPY'),('AUD/JPY','AUD_JPY'),
    ('EUR/GBP','EUR_GBP'),('EUR/AUD','EUR_AUD'),('GBP/AUD','GBP_AUD')]
for lbl,instr in fx:
    try:
        df=load_oanda_data(instr,period='10y',interval='1d')
        for bbs in [1.5,2.0]:
            sh,ar,dd,wr,n=run(df,BBReversion,bb_std=bbs,trend_ema=200,tp_mult=1.0)
            if sh<0.5: continue
            flag='✅ ADD' if sh>=1.0 and ar>0 else '⚠  WATCH'
            name='{} BB{}'.format(lbl,bbs)
            print('{:<16} {:>5.2f} {:>6.1f}% {:>6.1f}% {:>5.1f}% {:>5}  {}'.format(name,sh,ar,dd,wr,n,flag))
    except Exception as e:
        print('{} ERR: {}'.format(lbl, e))
