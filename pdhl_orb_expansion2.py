from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

def _atr(high, low, close, period=14):
    h,l,c = np.array(high),np.array(low),np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

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
        if session_end>self.session_hour:
            in_window = self.session_hour < hour <= session_end
        else:
            in_window = hour > self.session_hour or hour <= session_end
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

def run(df, cls, **kw):
    bt=Backtest(df,cls,cash=100_000,commission=0.0002,margin=1/30,finalize_trades=True)
    s=bt.run(**kw)
    return (float(s.get('Sharpe Ratio',0) or 0),float(s.get('Return (Ann.) [%]',0) or 0),
            float(s.get('Max. Drawdown [%]',0) or 0),float(s.get('Win Rate [%]',0) or 0),
            int(s.get('# Trades',0) or 0))

PDHL_UNTESTED = [
    ('EUR/USD','EUR_USD'),('AUD/USD','AUD_USD'),('USD/CHF','USD_CHF'),('NZD/USD','NZD_USD'),
    ('EUR/GBP','EUR_GBP'),('EUR/AUD','EUR_AUD'),('EUR/CAD','EUR_CAD'),
    ('GBP/AUD','GBP_AUD'),('GBP/CAD','GBP_CAD'),('GBP/CHF','GBP_CHF'),
    ('CAD/JPY','CAD_JPY'),('AUD/CHF','AUD_CHF'),
    ('NAS100','NAS100_USD'),('DE30','DE30_EUR'),('JP225','JP225_USD'),
    ('UK100','UK100_GBP'),('AU200','AU200_AUD'),('EU50','EU50_EUR'),('FR40','FR40_EUR'),
    ('WTICO','WTICO_USD'),('CORN','CORN_USD'),('WHEAT','WHEAT_USD'),
    ('SOYBN','SOYBN_USD'),('SUGAR','SUGAR_USD'),
]
ORB_LONDON_UNTESTED = [
    ('EUR/USD','EUR_USD'),('GBP/USD','GBP_USD'),('AUD/USD','AUD_USD'),
    ('USD/CHF','USD_CHF'),('NZD/USD','NZD_USD'),
    ('EUR/GBP','EUR_GBP'),('GBP/AUD','GBP_AUD'),('GBP/CAD','GBP_CAD'),('GBP/CHF','GBP_CHF'),
    ('NAS100','NAS100_USD'),('SPX500','SPX500_USD'),('US30','US30_USD'),
    ('DE30','DE30_EUR'),('UK100','UK100_GBP'),
    ('XAU/USD','XAU_USD'),('XAG/USD','XAG_USD'),
    ('BCO','BCO_USD'),('WTICO','WTICO_USD'),('NATGAS','NATGAS_USD'),
]
ORB_TOKYO_JPY = [
    ('EUR/JPY','EUR_JPY'),('GBP/JPY','GBP_JPY'),('USD/JPY','USD_JPY'),
    ('AUD/JPY','AUD_JPY'),('CAD/JPY','CAD_JPY'),('NZD/JPY','NZD_JPY'),('CHF/JPY','CHF_JPY'),
]

def section(title, tests, cls, tf, session_hour=None):
    print('\n' + '='*62)
    print(title)
    print('='*62)
    print('{:<13} {:>5} {:>6} {:>6} {:>5} {:>5}  Verdict'.format('Instrument','Sh','Ann%','DD%','WR%','N'))
    print('-'*52)
    results=[]
    for lbl,instr in tests:
        try:
            df=load_oanda_data(instr,period='10y',interval=tf)
            kw = {'session_hour':session_hour} if session_hour is not None else {}
            sh,ar,dd,wr,n=run(df,cls,**kw)
            results.append((sh,ar,dd,wr,n,lbl))
        except Exception as e:
            pass
    for sh,ar,dd,wr,n,lbl in sorted(results,reverse=True):
        flag='YES ADD' if sh>=1.0 and ar>0 else ('WATCH' if sh>=0.5 and ar>0 else '---')
        print('{:<13} {:>5.2f} {:>5.1f}% {:>5.1f}% {:>5.1f}% {:>5}  {}'.format(lbl,sh,ar,dd,wr,n,flag))
    return results

section('PDHL H1 -- untested instruments (10y)', PDHL_UNTESTED, PDHL, '1h')
section('ORB London 08:00 UTC -- untested (10y)', ORB_LONDON_UNTESTED, ORB, '30m', 8)
section('ORB Tokyo  00:00 UTC -- JPY pairs (10y)', ORB_TOKYO_JPY, ORB, '30m', 0)
section('ORB NY     13:00 UTC -- untested (10y)', ORB_LONDON_UNTESTED, ORB, '30m', 13)
