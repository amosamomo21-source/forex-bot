"""
EMA/RSI momentum-fade backtest — EUR/USD M15, using OANDA data.
Tests short-only, long-only, and both directions.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

EMA_FAST   = 50
EMA_SLOW   = 200
RSI_PERIOD = 14
SL_PIPS    = 15
TP_PIPS    = 20
PIP_SIZE   = 0.0001

def _ema(arr, n):
    return pd.Series(np.array(arr, dtype=float)).ewm(span=n, adjust=False).mean().values

def _rsi(close, period=14):
    c = pd.Series(np.array(close, dtype=float))
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

class EMARSIFade(Strategy):
    """
    Downtrend: EMA50 < EMA200
    Short when price < EMA50 AND RSI crosses DOWN through 50
    Long  when price > EMA50 AND RSI crosses UP   through 50 (in uptrend)
    SL = 15 pips, TP = 20 pips, time stop = 48 bars (12h)
    """
    sl_pips     = SL_PIPS
    tp_pips     = TP_PIPS
    risk_pct    = 0.0025
    max_bars    = 48
    allow_long  = True
    allow_short = True

    def init(self):
        self.fast  = self.I(_ema, self.data.Close, EMA_FAST)
        self.slow  = self.I(_ema, self.data.Close, EMA_SLOW)
        self.rsi   = self.I(_rsi, self.data.Close, RSI_PERIOD)
        self.entry_bar = None

    def next(self):
        price = float(self.data.Close[-1])
        fast  = float(self.fast[-1])
        slow  = float(self.slow[-1])
        rsi   = float(self.rsi[-1])
        rsi_p = float(self.rsi[-2]) if len(self.rsi) > 1 else rsi

        sl_dist = self.sl_pips * PIP_SIZE
        tp_dist = self.tp_pips * PIP_SIZE
        ra = self.equity * self.risk_pct
        size = max(1, int(ra / sl_dist))

        # Time stop
        if self.position and self.entry_bar is not None:
            if len(self.data) - self.entry_bar >= self.max_bars:
                self.position.close()
                self.entry_bar = None
                return

        if self.position:
            return

        downtrend = fast < slow
        uptrend   = fast > slow

        # SHORT: downtrend, price < EMA50, RSI crosses down through 50
        if self.allow_short and downtrend and price < fast:
            if rsi_p >= 50 and rsi < 50:
                self.sell(size=size, sl=price + sl_dist, tp=price - tp_dist)
                self.entry_bar = len(self.data)

        # LONG: uptrend, price > EMA50, RSI crosses up through 50
        elif self.allow_long and uptrend and price > fast:
            if rsi_p <= 50 and rsi > 50:
                self.buy(size=size, sl=price - sl_dist, tp=price + tp_dist)
                self.entry_bar = len(self.data)

class ShortOnly(EMARSIFade):
    allow_long  = False
    allow_short = True

class LongOnly(EMARSIFade):
    allow_long  = True
    allow_short = False

def run(df, cls):
    bt = Backtest(df, cls, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
    s  = bt.run()
    return {
        'Sharpe':  float(s.get('Sharpe Ratio', 0) or 0),
        'Ann%':    float(s.get('Return (Ann.) [%]', 0) or 0),
        'DD%':     float(s.get('Max. Drawdown [%]', 0) or 0),
        'WR%':     float(s.get('Win Rate [%]', 0) or 0),
        'N':       int(s.get('# Trades', 0) or 0),
        'PF':      float(s.get('Profit Factor', 0) or 0),
    }

print('\nEMA/RSI Fade — EUR/USD M15 (0.25% risk, 15pip SL / 20pip TP, 12h time stop)')
print('='*68)
print(f"{'Direction':<14} {'Period':<6} {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>5} {'PF':>5} {'N':>5}  Verdict")
print('-'*68)

for period in ['2y','5y','10y']:
    df = load_oanda_data('EUR_USD', period=period, interval='15m')
    for label, cls in [('Short-only', ShortOnly), ('Long-only', LongOnly), ('Both dirs', EMARSIFade)]:
        r = run(df, cls)
        flag = 'YES' if r['Sharpe']>=1.0 and r['Ann%']>0 else ('WATCH' if r['Sharpe']>=0.5 and r['Ann%']>0 else '---')
        print(f"{label:<14} {period:<6} {r['Sharpe']:>5.2f} {r['Ann%']:>5.1f}% {r['DD%']:>5.1f}% {r['WR%']:>5.1f}% {r['PF']:>5.2f} {r['N']:>5}  {flag}")
    print()
