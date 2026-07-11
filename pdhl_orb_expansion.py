"""
PDHL + ORB expansion backtest — untested instruments only.
Also tests ORB Tokyo session (00:00 UTC) on JPY pairs.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

def _atr(high, low, close, period=14):
    h,l,c = np.array(high),np.array(low),np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

# ── PDHL H1 ──────────────────────────────────────────────────────────────────
class PDHL(Strategy):
    risk_pct = 0.0025; rr = 1.5
    def init(self):
        self.prev_high = np.nan; self.prev_low = np.nan
        self.last_date = None
        self.day_high = -np.inf; self.day_low = np.inf
    def next(self):
        t = self.data.index[-1]; date = t.date()
        if date != self.last_date:
            if self.last_date is not None:
                self.prev_high = self.day_high; self.prev_low = self.day_low
            self.day_high = float(self.data.High[-1])
            self.day_low  = float(self.data.Low[-1])
            self.last_date = date
        else:
            self.day_high = max(self.day_high, float(self.data.High[-1]))
            self.day_low  = min(self.day_low,  float(self.data.Low[-1]))
        if np.isnan(self.prev_high) or self.position: return
        price    = float(self.data.Close[-1])
        prev_mid = (self.prev_high + self.prev_low) / 2
        prev_rng = self.prev_high - self.prev_low
        if prev_rng <= 0: return
        ra = self.equity * self.risk_pct
        if price > self.prev_high:
            sd = price - prev_mid
            if sd <= 0: return
            sz = max(1, int(ra / sd))
            self.buy(size=sz, sl=prev_mid, tp=price + self.rr * prev_rng)
        elif price < self.prev_low:
            sd = prev_mid - price
            if sd <= 0: return
            sz = max(1, int(ra / sd))
            self.sell(size=sz, sl=prev_mid, tp=price - self.rr * prev_rng)

# ── ORB M30 ──────────────────────────────────────────────────────────────────
class ORB(Strategy):
    risk_pct   = 0.0025; rr = 1.5
    session_hour = 8   # UTC hour of session open bar

    def init(self):
        self.or_high = np.nan; self.or_low = np.nan
        self.or_set  = False; self.last_date = None

    def next(self):
        t = self.data.index[-1]; hour = t.hour; date = t.date()
        if date != self.last_date:
            self.or_high = np.nan; self.or_low = np.nan
            self.or_set  = False; self.last_date = date
        if hour == self.session_hour and not self.or_set:
            self.or_high = float(self.data.High[-1])
            self.or_low  = float(self.data.Low[-1])
            self.or_set  = True; return
        if not self.or_set or np.isnan(self.or_high) or self.position: return
        # Trade only within session window
        session_end = (self.session_hour + 7) % 24
        if not (self.session_hour < hour <= session_end): return
        price   = float(self.data.Close[-1])
        or_rng  = self.or_high - self.or_low
        if or_rng <= 0: return
        ra = self.equity * self.risk_pct
        if price > self.or_high:
            sd = price - self.or_low; sz = max(1, int(ra / sd))
            self.buy(size=sz, sl=self.or_low, tp=price + self.rr * or_rng)
        elif price < self.or_low:
            sd = self.or_high - price; sz = max(1, int(ra / sd))
            self.sell(size=sz, sl=self.or_high, tp=price - self.rr * or_rng)

def run(df, cls, **kw):
    bt = Backtest(df, cls, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
    s  = bt.run(**kw)
    return (float(s.get('Sharpe Ratio',0) or 0), float(s.get('Return (Ann.) [%]',0) or 0),
            float(s.get('Max. Drawdown [%]',0) or 0), float(s.get('Win Rate [%]',0) or 0),
            int(s.get('# Trades',0) or 0))

# ── Instruments not yet in PDHL ───────────────────────────────────────────────
PDHL_UNTESTED = [
    # FX majors not in PDHL
    ('EUR/USD',  'EUR_USD'), ('AUD/USD',  'AUD_USD'), ('USD/CAD',  'USD_CAD'),
    ('USD/CHF',  'USD_CHF'), ('NZD/USD',  'NZD_USD'),
    # FX crosses not in PDHL
    ('EUR/GBP',  'EUR_GBP'), ('EUR/AUD',  'EUR_AUD'), ('EUR/CAD',  'EUR_CAD'),
    ('GBP/AUD',  'GBP_AUD'), ('GBP/CAD',  'GBP_CAD'), ('GBP/CHF',  'GBP_CHF'),
    ('CAD/JPY',  'CAD_JPY'), ('AUD/CHF',  'AUD_CHF'), ('EUR/SGD',  'EUR_SGD'),
    # Indices not in PDHL
    ('NAS100',   'NAS100_USD'), ('DE30',  'DE30_EUR'), ('JP225',   'JP225_USD'),
    ('UK100',    'UK100_GBP'),  ('AU200', 'AU200_AUD'), ('EU50',   'EU50_EUR'),
    ('FR40',     'FR40_EUR'),   ('HK33',  'HK33_HKD'),
    # Commodities not in PDHL
    ('WTICO',    'WTICO_USD'), ('CORN',   'CORN_USD'), ('WHEAT',   'WHEAT_USD'),
    ('SOYBN',    'SOYBN_USD'), ('SUGAR',  'SUGAR_USD'),
]

# ── Instruments not yet in ORB (London 08:00 UTC) ────────────────────────────
ORB_LONDON_UNTESTED = [
    ('EUR/USD',  'EUR_USD'), ('GBP/USD',  'GBP_USD'), ('AUD/USD',  'AUD_USD'),
    ('USD/CHF',  'USD_CHF'), ('NZD/USD',  'NZD_USD'),
    ('EUR/GBP',  'EUR_GBP'), ('GBP/AUD',  'GBP_AUD'), ('GBP/CAD',  'GBP_CAD'),
    ('GBP/CHF',  'GBP_CHF'), ('GBP/SGD',  'GBP_SGD'),
    ('NAS100',   'NAS100_USD'), ('SPX500', 'SPX500_USD'), ('US30',  'US30_USD'),
    ('DE30',     'DE30_EUR'),   ('UK100',  'UK100_GBP'),
    ('XAU/USD',  'XAU_USD'),    ('XAG/USD','XAG_USD'),
    ('BCO',      'BCO_USD'),    ('WTICO',  'WTICO_USD'), ('NATGAS', 'NATGAS_USD'),
]

# ── ORB Tokyo session (00:00 UTC) on JPY pairs ───────────────────────────────
ORB_TOKYO = [
    ('EUR/JPY',  'EUR_JPY'), ('GBP/JPY',  'GBP_JPY'), ('USD/JPY',  'USD_JPY'),
    ('AUD/JPY',  'AUD_JPY'), ('CAD/JPY',  'CAD_JPY'), ('NZD/JPY',  'NZD_JPY'),
    ('CHF/JPY',  'CHF_JPY'),
]

# ── Run ───────────────────────────────────────────────────────────────────────
print('Loading data...')
all_instrs = set(i for _,i in PDHL_UNTESTED + ORB_LONDON_UNTESTED + ORB_TOKYO)
data = {}
for instr in all_instrs:
    try:
        tf = '1h' if instr in [i for _,i in PDHL_UNTESTED] else '30m'
        # Load both timeframes as needed
        data[(instr,'1h')]  = load_oanda_data(instr, period='10y', interval='1h')
        data[(instr,'30m')] = load_oanda_data(instr, period='10y', interval='30m')
    except Exception as e:
        pass
print(f'Done. {len(data)//2} instruments loaded.\n')

def print_results(title, results):
    print(f'\n{"="*65}')
    print(f'{title}')
    print(f'{"="*65}')
    print(f"{'Instrument':<13} {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>5} {'N':>5}  Verdict")
    print('-'*52)
    shown = False
    for sh,ar,dd,wr,n,lbl in sorted(results, reverse=True):
        flag = '✅ ADD' if sh>=1.0 and ar>0 else ('⚠  WATCH' if sh>=0.5 and ar>0 else '❌')
        print(f'{lbl:<13} {sh:>5.2f} {ar:>5.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5}  {flag}')
        shown = True
    if not shown:
        print('  (all below 0.0 Sharpe)')

# PDHL H1
results_pdhl = []
for lbl, instr in PDHL_UNTESTED:
    key = (instr,'1h')
    if key not in data: continue
    try:
        sh,ar,dd,wr,n = run(data[key], PDHL)
        results_pdhl.append((sh,ar,dd,wr,n,lbl))
    except: pass
print_results('PDHL H1 — untested instruments (10y, 0.25% risk)', results_pdhl)

# ORB London 08:00
results_orb_lon = []
for lbl, instr in ORB_LONDON_UNTESTED:
    key = (instr,'30m')
    if key not in data: continue
    try:
        sh,ar,dd,wr,n = run(data[key], ORB, session_hour=8)
        results_orb_lon.append((sh,ar,dd,wr,n,lbl))
    except: pass
print_results('ORB M30 London 08:00 UTC — untested instruments (10y)', results_orb_lon)

# ORB Tokyo 00:00
results_orb_tok = []
for lbl, instr in ORB_TOKYO:
    key = (instr,'30m')
    if key not in data: continue
    try:
        sh,ar,dd,wr,n = run(data[key], ORB, session_hour=0)
        results_orb_tok.append((sh,ar,dd,wr,n,lbl))
    except: pass
print_results('ORB M30 Tokyo 00:00 UTC — JPY pairs (10y)', results_orb_tok)

# ORB NY 13:00 on untested
results_orb_ny = []
for lbl, instr in ORB_LONDON_UNTESTED:
    key = (instr,'30m')
    if key not in data: continue
    try:
        sh,ar,dd,wr,n = run(data[key], ORB, session_hour=13)
        results_orb_ny.append((sh,ar,dd,wr,n,lbl))
    except: pass
print_results('ORB M30 NY 13:00 UTC — untested instruments (10y)', results_orb_ny)
