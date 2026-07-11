"""
ConsecReversion D1 audit — test 3-consecutive-close mean reversion on
commodities and FX pairs to find instruments worth adding alongside WHEAT.

Logic (matches live runner exactly):
  - 3 consecutive down closes → BUY  (SL = 1.5 ATR, TP = 1.0 ATR)
  - 3 consecutive up closes   → SELL (SL = 1.5 ATR, TP = 1.0 ATR)

WHEAT_USD is the known baseline (10y Sharpe 1.04, WR 68%).
Threshold for addition: Sharpe >= 0.7, WR >= 55%, N >= 30 trades.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

STREAK    = 3
SL_ATR    = 1.5
TP_ATR    = 1.0
RISK_PCT  = 0.0025


def _atr(high, low, close, period=14):
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return pd.Series(np.concatenate([[tr[0]], tr])).ewm(span=period, adjust=False).mean().values


class ConsecReversion(Strategy):
    streak       = STREAK
    sl_atr_mult  = SL_ATR
    tp_atr_mult  = TP_ATR
    risk_pct     = RISK_PCT

    def init(self):
        self.atr = self.I(_atr, self.data.High, self.data.Low, self.data.Close)

    def next(self):
        if self.position:
            return

        n      = self.streak
        closes = np.array(self.data.Close)
        if len(closes) < n + 2:
            return

        down = all(closes[-(i+1)] < closes[-(i+2)] for i in range(n))
        up   = all(closes[-(i+1)] > closes[-(i+2)] for i in range(n))

        if not (down or up):
            return

        av = float(self.atr[-1])
        if np.isnan(av) or av <= 0:
            return

        price     = float(self.data.Close[-1])
        stop_dist = self.sl_atr_mult * av
        units     = max(1, int(self.equity * self.risk_pct / stop_dist))

        if down:
            self.buy(size=units, sl=price - stop_dist, tp=price + self.tp_atr_mult * av)
        else:
            self.sell(size=units, sl=price + stop_dist, tp=price - self.tp_atr_mult * av)


def run(instr, period='10y'):
    try:
        df = load_oanda_data(instr, period=period, interval='1d')
    except Exception as e:
        return None, str(e)
    if df is None or len(df) < 50:
        return None, f"only {0 if df is None else len(df)} bars"
    bt = Backtest(df, ConsecReversion, cash=100_000, commission=0.0002,
                  margin=1/30, finalize_trades=True)
    s = bt.run()
    result = {
        'Sharpe': float(s.get('Sharpe Ratio', 0) or 0),
        'Ann%':   float(s.get('Return (Ann.) [%]', 0) or 0),
        'DD%':    float(s.get('Max. Drawdown [%]', 0) or 0),
        'WR%':    float(s.get('Win Rate [%]', 0) or 0),
        'PF':     float(s.get('Profit Factor', 0) or 0),
        'N':      int(s.get('# Trades', 0) or 0),
    }
    return result, None


INSTRUMENTS = [
    # Baseline (known good)
    ('WHEAT',    'WHEAT_USD'),
    # Other agricultural commodities
    ('CORN',     'CORN_USD'),
    ('SOYBEAN',  'SOYBEAN_USD'),
    ('SUGAR',    'SUGAR_USD'),
    # Metals
    ('XAU',      'XAU_USD'),
    ('XAG',      'XAG_USD'),
    # Energy
    ('BCO',      'BCO_USD'),
    ('NATGAS',   'NATGAS_USD'),
    ('WTICO',    'WTICO_USD'),
    # FX (test if the pattern holds)
    ('GBP/USD',  'GBP_USD'),
    ('EUR/JPY',  'EUR_JPY'),
    ('USD/JPY',  'USD_JPY'),
    ('AUD/JPY',  'AUD_JPY'),
    ('GBP/JPY',  'GBP_JPY'),
]

print(f'\nConsecReversion D1 audit — {STREAK}-bar streak, SL={SL_ATR}×ATR, TP={TP_ATR}×ATR, 10y')
print(f'{"Instrument":<12} {"Sh":>5} {"Ann%":>6} {"DD%":>6} {"WR%":>5} {"PF":>5} {"N":>5}  Verdict')
print('-' * 65)

results = []
for lbl, instr in INSTRUMENTS:
    r, err = run(instr)
    if err:
        print(f'{lbl:<12} ERROR: {err}')
        continue
    add     = r['Sharpe'] >= 0.7 and r['WR%'] >= 55 and r['N'] >= 30
    watch   = not add and r['Sharpe'] >= 0.4 and r['Ann%'] > 0
    verdict = 'ADD' if add else ('WATCH' if watch else '---')
    if lbl == 'WHEAT':
        verdict = 'BASELINE'
    print(f"{lbl:<12} {r['Sharpe']:>5.2f} {r['Ann%']:>5.1f}% {r['DD%']:>5.1f}% {r['WR%']:>5.1f}% {r['PF']:>5.2f} {r['N']:>5}  {verdict}")
    results.append((lbl, instr, r, verdict))

print()
adds = [(lbl, instr, r) for lbl, instr, r, v in results if v == 'ADD']
if adds:
    print(f'Candidates for CONSEC_D1_SLEEVES (Sharpe >= 0.7, WR >= 55%, N >= 30):')
    for lbl, instr, r in adds:
        print(f'  ("consec_d1_{lbl.lower().replace("/","")}", "{instr}"),  '
              f'# Sharpe {r["Sharpe"]:.2f}, WR {r["WR%"]:.0f}%, DD {r["DD%"]:.1f}%')
else:
    print('No new instruments meet the ADD threshold.')
