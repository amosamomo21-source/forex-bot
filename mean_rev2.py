from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

def _ema(arr, n):
    return pd.Series(np.array(arr, dtype=float)).ewm(span=n, adjust=False).mean().values
def _atr(high, low, close, period=14):
    h,l,c = np.array(high),np.array(low),np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

class ConsecReversion(Strategy):
    streak=3; sl_atr=1.5; tp_atr=1.0; risk_pct=0.0025; trend_ema=0; longs=True; shorts=True
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
        if down and tu and self.longs:
            self.buy(size=sz,sl=price-sd,tp=price+self.tp_atr*atr)
        elif up and td and self.shorts:
            self.sell(size=sz,sl=price+sd,tp=price-self.tp_atr*atr)

def run(df, cls=ConsecReversion, **kw):
    bt=Backtest(df,cls,cash=100_000,commission=0.0002,margin=1/30,finalize_trades=True)
    s=bt.run(**kw)
    return (float(s.get('Sharpe Ratio',0) or 0), float(s.get('Return (Ann.) [%]',0) or 0),
            float(s.get('Max. Drawdown [%]',0) or 0), float(s.get('Win Rate [%]',0) or 0),
            int(s.get('# Trades',0) or 0))

print('WHEAT Consec-Day Reversion -- cross-period validation')
print('S=3, SL=1.5 ATR, TP=1.0 ATR  (need >60% WR to be +EV at this R:R)')
print()
print('{:<8} {:>5} {:>6} {:>7} {:>5} {:>5}  Verdict'.format('Period','Sh','Ann%','MaxDD','WR%','N'))
print('-'*48)
for period in ['2y','3y','5y','7y','10y']:
    df = load_oanda_data('WHEAT_USD', period=period, interval='1d')
    sh,ar,dd,wr,n = run(df, streak=3, sl_atr=1.5, tp_atr=1.0, trend_ema=0)
    flag = 'PASS' if sh>=1.0 and ar>0 else ('WATCH' if sh>=0.5 and ar>0 else 'FAIL')
    print('{:<8} {:>5.2f} {:>5.1f}% {:>6.1f}% {:>5.1f}% {:>5}  {}'.format(period,sh,ar,dd,wr,n,flag))

print()
print('Streak sensitivity on 10y:')
df10 = load_oanda_data('WHEAT_USD', period='10y', interval='1d')
for streak in [2,3,4,5]:
    sh,ar,dd,wr,n = run(df10, streak=streak, sl_atr=1.5, tp_atr=1.0, trend_ema=0)
    flag = 'PASS' if sh>=1.0 and ar>0 else ('WATCH' if sh>=0.5 and ar>0 else 'FAIL')
    print('  S={} Sh={:.2f} Ann={:+.1f}% DD={:.1f}% WR={:.0f}% N={}  {}'.format(streak,sh,ar,dd,wr,n,flag))

print()
print('Long only vs Short only vs Both (S=3, TP=1.0x, 10y):')
for lbl,lo,sh_ in [('Both  ',True,True),('Longs ',True,False),('Shorts',False,True)]:
    sh,ar,dd,wr,n = run(df10, streak=3, sl_atr=1.5, tp_atr=1.0, trend_ema=0, longs=lo, shorts=sh_)
    print('  {} Sh={:.2f} Ann={:+.1f}% WR={:.0f}% N={}'.format(lbl,sh,ar,wr,n))
