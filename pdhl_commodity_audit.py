from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

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

COMMODITIES = [
    ('XAG/USD', 'XAG_USD'),
    ('XAU/USD', 'XAU_USD'),
    ('BCO',     'BCO_USD'),
    ('NATGAS',  'NATGAS_USD'),
]

print('PDHL H1 commodity audit — 10y static backtest')
print(f"{'Instrument':<12} {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>5} {'N':>5}  Verdict")
print('-'*58)
for lbl, instr in COMMODITIES:
    df = load_oanda_data(instr, period='10y', interval='1h')
    bt = Backtest(df, PDHL, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
    s  = bt.run()
    sh  = float(s.get('Sharpe Ratio',0) or 0)
    ann = float(s.get('Return (Ann.) [%]',0) or 0)
    dd  = float(s.get('Max. Drawdown [%]',0) or 0)
    wr  = float(s.get('Win Rate [%]',0) or 0)
    n   = int(s.get('# Trades',0) or 0)
    verdict = 'KEEP' if sh>=0.3 and ann>0 else 'REMOVE'
    print(f'{lbl:<12} {sh:>5.2f} {ann:>5.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}  {verdict}')
