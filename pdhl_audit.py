from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

def _atr(high, low, close, period=14):
    h,l,c = np.array(high),np.array(low),np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return pd.Series(np.concatenate([[tr[0]], tr])).ewm(span=period, adjust=False).mean().values

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

def run(df):
    bt=Backtest(df,PDHL,cash=100_000,commission=0.0002,margin=1/30,finalize_trades=True)
    s=bt.run()
    return (float(s.get('Sharpe Ratio',0) or 0), float(s.get('Return (Ann.) [%]',0) or 0),
            float(s.get('Max. Drawdown [%]',0) or 0), float(s.get('Win Rate [%]',0) or 0),
            int(s.get('# Trades',0) or 0))

LIVE_NON_FX = [
    ('SPX500', 'SPX500_USD'),
    ('US30',   'US30_USD'),
    ('NAS100', 'NAS100_USD'),
    ('XAU',    'XAU_USD'),
    ('XAG',    'XAG_USD'),
    ('BCO',    'BCO_USD'),
    ('NATGAS', 'NATGAS_USD'),
]

print('PDHL H1 audit -- live non-FX sleeves (10y)')
print('{:<12} {:>5} {:>6} {:>6} {:>5} {:>5}  Verdict'.format('Instrument','Sh','Ann%','DD%','WR%','N'))
print('-'*55)
for lbl, instr in LIVE_NON_FX:
    try:
        df = load_oanda_data(instr, period='10y', interval='1h')
        sh,ar,dd,wr,n = run(df)
        flag = 'KEEP' if sh>=0.5 and ar>0 else ('BORDERLINE' if sh>=0 else 'REMOVE?')
        print('{:<12} {:>5.2f} {:>5.1f}% {:>5.1f}% {:>5.1f}% {:>5}  {}'.format(lbl,sh,ar,dd,wr,n,flag))
    except Exception as e:
        print('{:<12} ERROR: {}'.format(lbl, e))
