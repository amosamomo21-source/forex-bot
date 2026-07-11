"""
Two improvement tests:
1. PDHL + D1 200 EMA trend filter (only long above EMA, only short below)
2. Compare live commodity PDHL sleeves with/without trend filter
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

def _atr(high, low, close, period=14):
    h,l,c = np.array(high),np.array(low),np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return pd.Series(np.concatenate([[tr[0]], tr])).ewm(span=period, adjust=False).mean().values

class PDHLBase(Strategy):
    risk_pct=0.0025; rr=1.5; use_trend=False
    def init(self):
        self.prev_high=np.nan; self.prev_low=np.nan
        self.last_date=None; self.day_high=-np.inf; self.day_low=np.inf
        self.trend_ema = self.data.TrendEMA  # precomputed D1 200 EMA
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
        trend=float(self.trend_ema[-1])

        if price>self.prev_high:
            if self.use_trend and price < trend: return  # skip: long in downtrend
            sd=price-prev_mid
            if sd<=0: return
            self.buy(size=max(1,int(ra/sd)),sl=prev_mid,tp=price+self.rr*prev_rng)
        elif price<self.prev_low:
            if self.use_trend and price > trend: return  # skip: short in uptrend
            sd=prev_mid-price
            if sd<=0: return
            self.sell(size=max(1,int(ra/sd)),sl=prev_mid,tp=price-self.rr*prev_rng)

class PDHLNoFilter(PDHLBase):
    use_trend = False

class PDHLTrendFilter(PDHLBase):
    use_trend = True

def prep(instr):
    df = load_oanda_data(instr, period='10y', interval='1h')
    daily = df['Close'].resample('D').last().dropna()
    ema200 = daily.ewm(span=200, adjust=False).mean()
    df['TrendEMA'] = ema200.reindex(df.index, method='ffill').bfill()
    return df

def run(df, cls):
    bt=Backtest(df,cls,cash=100_000,commission=0.0002,margin=1/30,finalize_trades=True)
    s=bt.run()
    return (float(s.get('Sharpe Ratio',0) or 0),
            float(s.get('Return (Ann.) [%]',0) or 0),
            float(s.get('Max. Drawdown [%]',0) or 0),
            float(s.get('Win Rate [%]',0) or 0),
            int(s.get('# Trades',0) or 0))

FX_PAIRS = [
    ('GBP/USD','GBP_USD'),('EUR/JPY','EUR_JPY'),('CHF/JPY','CHF_JPY'),
    ('AUD/JPY','AUD_JPY'),('GBP/JPY','GBP_JPY'),('NZD/JPY','NZD_JPY'),('USD/JPY','USD_JPY'),
]
COMMODITIES = [
    ('XAU/USD','XAU_USD'),('XAG/USD','XAG_USD'),('BCO','BCO_USD'),('NATGAS','NATGAS_USD'),
]

def section(title, instruments):
    print(f'\n{"="*72}')
    print(title)
    print(f'{"="*72}')
    print(f"{'Instrument':<12} {'':>2} {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>5} {'N':>5}  Verdict")
    print('-'*72)
    for lbl, instr in instruments:
        df = prep(instr)
        sh0,ar0,dd0,wr0,n0 = run(df, PDHLNoFilter)
        sh1,ar1,dd1,wr1,n1 = run(df, PDHLTrendFilter)
        flag0 = 'KEEP' if sh0>=0.5 and ar0>0 else ('~' if sh0>=0 else 'WEAK')
        flag1 = 'KEEP' if sh1>=0.5 and ar1>0 else ('~' if sh1>=0 else 'WEAK')
        delta_sh = sh1 - sh0
        delta_wr = wr1 - wr0
        print(f'{lbl:<12} no-filt  {sh0:>5.2f} {ar0:>5.1f}% {dd0:>5.1f}% {wr0:>5.1f}% {n0:>5}  {flag0}')
        print(f'{"":12} trend    {sh1:>5.2f} {ar1:>5.1f}% {dd1:>5.1f}% {wr1:>5.1f}% {n1:>5}  {flag1}  (Sh {delta_sh:+.2f}  WR {delta_wr:+.1f}%)')
        print()

section('TEST 1 — PDHL FX pairs: no filter vs D1 200 EMA trend filter (10y)', FX_PAIRS)
section('TEST 2 — PDHL Commodities: no filter vs D1 200 EMA trend filter (10y)', COMMODITIES)
