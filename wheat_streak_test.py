"""
WHEAT ConsecReversion streak-length test.
Tests streak = 2, 3, 4, 5, 6 consecutive same-direction closes.
SL and TP stay fixed (1.5 ATR / 1.0 ATR) so break-even WR is always 60%.
The question: does a longer streak push WR above 75% while staying profitable?
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

SL_ATR   = 1.5
TP_ATR   = 1.0
RISK_PCT = 0.0025
BREAKEVEN_WR = SL_ATR / (SL_ATR + TP_ATR) * 100   # 60.0%

def _atr(high, low, close, period=14):
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    return pd.Series(np.concatenate([[tr[0]], tr])).ewm(span=period, adjust=False).mean().values

class ConsecReversion(Strategy):
    streak      = 3
    sl_atr_mult = SL_ATR
    tp_atr_mult = TP_ATR
    risk_pct    = RISK_PCT

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

df = load_oanda_data('WHEAT_USD', period='10y', interval='1d')

print(f'\nWHEAT ConsecReversion — streak length sensitivity (10y, SL={SL_ATR}×ATR, TP={TP_ATR}×ATR)')
print(f'Break-even WR at this R:R = {BREAKEVEN_WR:.0f}%   (any WR above this = profitable)')
print()
print(f"{'Streak':<8} {'Sh':>5} {'Ann%':>6} {'DD%':>6} {'WR%':>6} {'Margin':>7} {'N':>5}  Verdict")
print('-' * 60)

for streak in [2, 3, 4, 5, 6]:
    class S(ConsecReversion):
        pass
    S.streak = streak

    bt = Backtest(df, S, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
    s  = bt.run()

    sharpe = float(s.get('Sharpe Ratio', 0) or 0)
    ann    = float(s.get('Return (Ann.) [%]', 0) or 0)
    dd     = float(s.get('Max. Drawdown [%]', 0) or 0)
    wr     = float(s.get('Win Rate [%]', 0) or 0)
    n      = int(s.get('# Trades', 0) or 0)
    margin = wr - BREAKEVEN_WR

    hits75 = '✓ 75%+' if wr >= 75 else ''
    live   = '<-- LIVE' if streak == 3 else ''
    print(f"streak={streak}  {sharpe:>5.2f} {ann:>5.1f}% {dd:>5.1f}% {wr:>6.1f}% {margin:>+6.1f}% {n:>5}  {hits75} {live}")

print()
print(f'Margin = WR% − {BREAKEVEN_WR:.0f}% break-even. Positive margin = profitable.')
print('Higher streak = fewer trades, higher WR, but smaller total profit if margin shrinks.')
