"""
VSF parameter test: 1×ATR vs 2×ATR entry threshold, all 34 instruments.
Tests whether a lower volatility filter (1×ATR) generates more signals with
sufficient edge, or whether it dilutes the strategy into noise.
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
    vol_mult  = 2.0
    close_pct = 0.25
    rr        = 2.0
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
            sl = p - 1.5*av; sd = max(p-sl,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p + self.rr*sd
            if sl < p < tp: self.buy(size=u,sl=sl,tp=tp)
            self._wl = False
        elif self._ws:
            sl = p + 1.5*av; sd = max(sl-p,1e-9)
            u  = max(1,min(int(ra/sd),100_000)); tp = p - self.rr*sd
            if tp < p < sl: self.sell(size=u,sl=sl,tp=tp)
            self._ws = False

class VSF_1x_R2(VolSpikeFade): vol_mult=1.0; rr=2.0
class VSF_1x_R3(VolSpikeFade): vol_mult=1.0; rr=3.0
class VSF_2x_R2(VolSpikeFade): vol_mult=2.0; rr=2.0
class VSF_2x_R3(VolSpikeFade): vol_mult=2.0; rr=3.0

VARIANTS = [
    ('1x ATR R2', VSF_1x_R2),
    ('1x ATR R3', VSF_1x_R3),
    ('2x ATR R2', VSF_2x_R2),
    ('2x ATR R3', VSF_2x_R3),
]

INSTRUMENTS = [
    ('WHEAT','WHEAT_USD'),('NATGAS','NATGAS_USD'),('XAU/USD','XAU_USD'),
    ('XAG/USD','XAG_USD'),('WTI Oil','WTICO_USD'),('Brent','BCO_USD'),
    ('Corn','CORN_USD'),('Soybeans','SOYBN_USD'),('Sugar','SUGAR_USD'),
    ('Copper','XCUUSD'),('JP225','JP225_USD'),('UK100','UK100_GBP'),
    ('NAS100','NAS100_USD'),('DE30','DE30_EUR'),('SPX500','SPX500_USD'),
    ('HK33','HK33_HKD'),('AU200','AU200_AUD'),('FR40','FR40_EUR'),
    ('EU50','EU50_EUR'),('EUR/USD','EUR_USD'),('GBP/USD','GBP_USD'),
    ('USD/JPY','USD_JPY'),('USD/CAD','USD_CAD'),('AUD/USD','AUD_USD'),
    ('NZD/USD','NZD_USD'),('USD/CHF','USD_CHF'),('GBP/JPY','GBP_JPY'),
    ('EUR/JPY','EUR_JPY'),('GBP/CHF','GBP_CHF'),('EUR/GBP','EUR_GBP'),
    ('AUD/JPY','AUD_JPY'),('GBP/AUD','GBP_AUD'),('EUR/AUD','EUR_AUD'),
    ('CAD/JPY','CAD_JPY'),
]

print('VSF: 1×ATR vs 2×ATR entry threshold')
print('10y D1 backtest, 34 instruments, 1.5×ATR SL, R2 & R3')
print(f"\n{'Instr':<10} {'Variant':<12} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  Verdict")
print('─'*72)

rows = []
for lbl, instr in INSTRUMENTS:
    try:
        df = load_oanda_data(instr, period='10y', interval='1d')
    except Exception as e:
        print(f'{lbl}: load error {e}'); continue
    for var_lbl, cls in VARIANTS:
        try:
            bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                          margin=1/30, finalize_trades=True)
            s  = bt.run()
            sh  = float(s.get('Sharpe Ratio', 0) or 0)
            ann = float(s.get('Return (Ann.) [%]', 0) or 0)
            dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
            wr  = float(s.get('Win Rate [%]', 0) or 0)
            n   = int(s.get('# Trades', 0) or 0)
            pf  = float(s.get('Profit Factor', 0) or 0)
            ok  = sh >= 0.4 and ann > 0 and n >= 15
            verdict = 'ADD' if ok else ('WATCH' if sh >= 0.1 and ann > 0 else 'PASS')
            print(f'{lbl:<10} {var_lbl:<12} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% '
                  f'{wr:>5.1f}% {n:>5} {pf:>5.2f}  {verdict}')
            rows.append({'lbl':lbl,'instr':instr,'var':var_lbl,
                         'sh':sh,'ann':ann,'wr':wr,'n':n,'dd':dd,'pf':pf,'verdict':verdict})
        except Exception as e:
            print(f'{lbl:<10} {var_lbl:<12} error: {e}')
    print()

print('─'*72)
adds  = [r for r in rows if r['verdict'] == 'ADD']
watch = [r for r in rows if r['verdict'] == 'WATCH']
print(f'ADD:   {len(adds)}   WATCH: {len(watch)}')

print('\n--- 1×ATR ADD/WATCH ---')
one_x = [r for r in rows if r['var'].startswith('1x') and r['verdict'] in ('ADD','WATCH')]
for r in sorted(one_x, key=lambda x: x['sh'], reverse=True)[:15]:
    print(f"  {r['lbl']:<10} {r['var']:<12} Sh {r['sh']:+.2f}  WR {r['wr']:.1f}%  "
          f"Ann {r['ann']:+.1f}%  DD {r['dd']:.1f}%  N={r['n']}  PF={r['pf']:.2f}")

print('\n--- 2×ATR ADD/WATCH (reference) ---')
two_x = [r for r in rows if r['var'].startswith('2x') and r['verdict'] in ('ADD','WATCH')]
for r in sorted(two_x, key=lambda x: x['sh'], reverse=True)[:10]:
    print(f"  {r['lbl']:<10} {r['var']:<12} Sh {r['sh']:+.2f}  WR {r['wr']:.1f}%  "
          f"Ann {r['ann']:+.1f}%  DD {r['dd']:.1f}%  N={r['n']}  PF={r['pf']:.2f}")
