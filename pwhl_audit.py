"""
Previous Week High/Low (PWHL) breakout audit.

Same logic as PDHL but on a weekly timeframe:
  - D1 close breaks above last week's high  → BUY
  - D1 close breaks below last week's low   → SELL
  - SL at weekly midpoint, TP = entry + 1.5 × weekly range

Tests the same instruments as PDHL and shows PDHL side-by-side so we can
see directly whether weekly timeframe is better, worse, or just different.

Threshold for addition: Sharpe >= 0.5, positive Ann%, N >= 30 trades.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd


def prep(instr):
    df = load_oanda_data(instr, period='10y', interval='1d')
    weekly_high = df['High'].resample('W').max()
    weekly_low  = df['Low'].resample('W').min()
    df['PrevWeekHigh'] = weekly_high.shift(1).reindex(df.index, method='ffill').bfill()
    df['PrevWeekLow']  = weekly_low.shift(1).reindex(df.index, method='ffill').bfill()
    return df.dropna()


class PWHL(Strategy):
    risk_pct = 0.0025
    rr       = 1.5

    def init(self):
        self.prev_high = self.data.PrevWeekHigh
        self.prev_low  = self.data.PrevWeekLow

    def next(self):
        if self.position:
            return
        price = float(self.data.Close[-1])
        ph    = float(self.prev_high[-1])
        pl    = float(self.prev_low[-1])
        if np.isnan(ph) or np.isnan(pl) or ph <= pl:
            return
        prev_mid = (ph + pl) / 2
        prev_rng = ph - pl
        ra       = self.equity * self.risk_pct
        if price > ph:
            sd = price - prev_mid
            if sd <= 0:
                return
            self.buy(size=max(1, int(ra / sd)), sl=prev_mid, tp=price + self.rr * prev_rng)
        elif price < pl:
            sd = prev_mid - price
            if sd <= 0:
                return
            self.sell(size=max(1, int(ra / sd)), sl=prev_mid, tp=price - self.rr * prev_rng)


class PDHL(Strategy):
    """Previous Day High/Low — baseline comparison, same R:R."""
    risk_pct = 0.0025
    rr       = 1.5

    def init(self):
        self.prev_high = np.nan
        self.prev_low  = np.nan
        self.last_date = None
        self.day_high  = -np.inf
        self.day_low   = np.inf

    def next(self):
        t    = self.data.index[-1]
        date = t.date()
        if date != self.last_date:
            if self.last_date is not None:
                self.prev_high = self.day_high
                self.prev_low  = self.day_low
            self.day_high  = float(self.data.High[-1])
            self.day_low   = float(self.data.Low[-1])
            self.last_date = date
        else:
            self.day_high = max(self.day_high, float(self.data.High[-1]))
            self.day_low  = min(self.day_low,  float(self.data.Low[-1]))

        if np.isnan(self.prev_high) or self.position:
            return
        price    = float(self.data.Close[-1])
        prev_mid = (self.prev_high + self.prev_low) / 2
        prev_rng = self.prev_high - self.prev_low
        if prev_rng <= 0:
            return
        ra = self.equity * self.risk_pct
        if price > self.prev_high:
            sd = price - prev_mid
            if sd <= 0:
                return
            self.buy(size=max(1, int(ra / sd)), sl=prev_mid, tp=price + self.rr * prev_rng)
        elif price < self.prev_low:
            sd = prev_mid - price
            if sd <= 0:
                return
            self.sell(size=max(1, int(ra / sd)), sl=prev_mid, tp=price - self.rr * prev_rng)


def run(df, cls):
    bt = Backtest(df, cls, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
    s  = bt.run()
    return {
        'Sharpe': float(s.get('Sharpe Ratio', 0) or 0),
        'Ann%':   float(s.get('Return (Ann.) [%]', 0) or 0),
        'DD%':    float(s.get('Max. Drawdown [%]', 0) or 0),
        'WR%':    float(s.get('Win Rate [%]', 0) or 0),
        'N':      int(s.get('# Trades', 0) or 0),
    }


INSTRUMENTS = [
    # FX (all negative in PDHL static backtest — can PWHL fix them?)
    ('GBP/USD',  'GBP_USD'),
    ('EUR/JPY',  'EUR_JPY'),
    ('CHF/JPY',  'CHF_JPY'),
    ('AUD/JPY',  'AUD_JPY'),
    ('GBP/JPY',  'GBP_JPY'),
    ('NZD/JPY',  'NZD_JPY'),
    ('USD/JPY',  'USD_JPY'),
    # Commodities (mixed in PDHL)
    ('XAU',      'XAU_USD'),
    ('XAG',      'XAG_USD'),
    ('BCO',      'BCO_USD'),
    ('NATGAS',   'NATGAS_USD'),
    # Indices not yet in PDHL
    ('NAS100',   'NAS100_USD'),
    ('DE30',     'DE30_EUR'),
]

hdr = f"{'Instrument':<12}  {'--- PWHL (weekly) ---':^30}  {'--- PDHL (daily) ---':^30}"
print(f'\nPWHL vs PDHL — 10y backtest, 0.25% risk, SL=weekly mid, TP=1.5×range')
print(hdr)
print(f"{'':12}  {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>5} {'N':>4}  "
      f"  {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>5} {'N':>4}  Delta-Sh")
print('-' * 90)

for lbl, instr in INSTRUMENTS:
    try:
        df = prep(instr)
    except Exception as e:
        print(f'{lbl:<12}  ERROR loading: {e}')
        continue

    try:
        pw = run(df, PWHL)
    except Exception as e:
        print(f'{lbl:<12}  PWHL ERROR: {e}')
        continue

    try:
        pd_ = run(df, PDHL)
    except Exception as e:
        print(f'{lbl:<12}  PDHL ERROR: {e}')
        continue

    delta = pw['Sharpe'] - pd_['Sharpe']
    pw_ok = pw['Sharpe'] >= 0.5 and pw['Ann%'] > 0 and pw['N'] >= 30
    verdict = 'ADD' if pw_ok else ('WATCH' if pw['Sharpe'] >= 0.3 and pw['Ann%'] > 0 else '')

    print(f"{lbl:<12}  "
          f"{pw['Sharpe']:>5.2f} {pw['Ann%']:>5.1f}% {pw['DD%']:>5.1f}% {pw['WR%']:>5.1f}% {pw['N']:>4}  "
          f"  {pd_['Sharpe']:>5.2f} {pd_['Ann%']:>5.1f}% {pd_['DD%']:>5.1f}% {pd_['WR%']:>5.1f}% {pd_['N']:>4}  "
          f"{delta:+.2f}  {verdict}")

print()
print('ADD threshold: PWHL Sharpe >= 0.5, positive Ann%, N >= 30')
