"""
Swing trading strategy backtest — three approaches at H4 and D1.

1. MA Pullback Swing:
   - Weekly trend filter (price > weekly 50 EMA = bullish, < = bearish)
   - Enter on pullback: RSI dips to 35-50 zone then bounces back above 50
   - SL: 1.5 ATR below entry
   - TP: 3 ATR above entry (3:2 R:R)

2. H4 EMA Crossover (our live strategy at higher timeframe):
   - Same EmaCrossoverAtr logic but run at H4 — wider stops, longer holds

3. Swing High/Low Breakout:
   - 3-bar swing high: middle bar has higher high than both neighbours
   - Buy when price closes above a recent swing high (last 5 swings)
   - SL below the swing low, TP at 2x the swing move
"""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from data import load_oanda_data


def _ema(arr, n):
    return pd.Series(np.array(arr, dtype=float)).ewm(span=n, adjust=False).mean().values

def _rsi(close, period=14):
    c   = pd.Series(np.array(close, dtype=float))
    d   = c.diff()
    up  = d.clip(lower=0).ewm(span=period, adjust=False).mean()
    dn  = (-d.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs  = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values

def _atr(high, low, close, period=14):
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.maximum(h[1:]-l[1:],
         np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


# ── 1. MA Pullback Swing ──────────────────────────────────────────────────────
class MAPullbackSwing(Strategy):
    """
    Trend-following swing: buy pullbacks in uptrends, sell rallies in downtrends.
    - Trend:  50 EMA direction on H4/D1 bars
    - Entry:  RSI dips below 45 then closes back above 50 (bullish swing)
              RSI rallies above 55 then closes back below 50 (bearish swing)
    - Stop:   1.5 ATR from entry
    - Target: 3.0 ATR from entry
    """
    ema_period  = 50
    rsi_period  = 14
    atr_period  = 14
    sl_atr      = 1.5
    tp_atr      = 3.0
    risk_pct    = 0.0025

    def init(self):
        cl = self.data.Close
        hi = self.data.High
        lo = self.data.Low

        self.ema  = self.I(_ema, cl, self.ema_period, name='EMA50')
        self.rsi  = self.I(_rsi, cl, self.rsi_period, name='RSI')
        self.atr  = self.I(_atr, hi, lo, cl, self.atr_period, name='ATR')

    def next(self):
        if self.position:
            return

        price = float(self.data.Close[-1])
        ema   = float(self.ema[-1])
        rsi   = float(self.rsi[-1])
        rsi_p = float(self.rsi[-2])  # previous RSI
        atr   = float(self.atr[-1])

        if atr <= 0 or np.isnan(atr) or np.isnan(ema):
            return

        risk_amount = self.equity * self.risk_pct
        stop_dist   = self.sl_atr * atr
        size        = max(1, int(risk_amount / stop_dist))

        # Bullish swing: price above EMA (uptrend), RSI pulled back then bounced
        if price > ema and rsi_p < 45 and rsi >= 50:
            sl = price - stop_dist
            tp = price + self.tp_atr * atr
            self.buy(size=size, sl=sl, tp=tp)

        # Bearish swing: price below EMA (downtrend), RSI bounced then fell back
        elif price < ema and rsi_p > 55 and rsi <= 50:
            sl = price + stop_dist
            tp = price - self.tp_atr * atr
            self.sell(size=size, sl=sl, tp=tp)


# ── 2. H4 EMA Crossover (our live strategy on higher TF) ─────────────────────
class H4EMACross(Strategy):
    """
    Fast/slow EMA crossover at H4 — same concept as our live H1 sleeve
    but wider timeframe = fewer signals, longer holds, cleaner trends.
    """
    fast_per   = 9
    slow_per   = 21
    atr_period = 14
    sl_atr     = 2.0
    tp_atr     = 4.0
    risk_pct   = 0.0025

    def init(self):
        cl = self.data.Close
        hi = self.data.High
        lo = self.data.Low
        self.fast = self.I(_ema, cl, self.fast_per, name='FastEMA')
        self.slow = self.I(_ema, cl, self.slow_per, name='SlowEMA')
        self.atr  = self.I(_atr, hi, lo, cl, self.atr_period, name='ATR')

    def next(self):
        if self.position:
            return

        fast  = float(self.fast[-1])
        fast_p = float(self.fast[-2])
        slow  = float(self.slow[-1])
        slow_p = float(self.slow[-2])
        atr   = float(self.atr[-1])
        price = float(self.data.Close[-1])

        if atr <= 0 or np.isnan(atr):
            return

        cross_up = fast > slow and fast_p <= slow_p
        cross_dn = fast < slow and fast_p >= slow_p

        risk_amount = self.equity * self.risk_pct
        stop_dist   = self.sl_atr * atr
        size        = max(1, int(risk_amount / stop_dist))

        if cross_up:
            self.buy(size=size, sl=price - stop_dist, tp=price + self.tp_atr * atr)
        elif cross_dn:
            self.sell(size=size, sl=price + stop_dist, tp=price - self.tp_atr * atr)


# ── 3. Swing High/Low Structure Breakout ─────────────────────────────────────
class SwingStructure(Strategy):
    """
    3-bar swing highs/lows: middle bar with higher high (or lower low) than
    both neighbours. Buy break of recent swing high, sell break of swing low.
    Targets 2x the distance from swing low to breakout point.
    """
    lookback = 10   # how many bars back to find the last swing point
    rr       = 2.0
    risk_pct = 0.0025

    def init(self):
        self.atr = self.I(_atr, self.data.High, self.data.Low,
                          self.data.Close, 14, name='ATR')

    def _last_swing_high(self):
        highs = self.data.High
        for i in range(2, min(self.lookback + 2, len(highs) - 1)):
            idx = -(i)
            if float(highs[idx]) > float(highs[idx-1]) and float(highs[idx]) > float(highs[idx+1]):
                return float(highs[idx])
        return np.nan

    def _last_swing_low(self):
        lows = self.data.Low
        for i in range(2, min(self.lookback + 2, len(lows) - 1)):
            idx = -(i)
            if float(lows[idx]) < float(lows[idx-1]) and float(lows[idx]) < float(lows[idx+1]):
                return float(lows[idx])
        return np.nan

    def next(self):
        if self.position:
            return

        price = float(self.data.Close[-1])
        atr   = float(self.atr[-1])
        if atr <= 0 or np.isnan(atr):
            return

        swing_high = self._last_swing_high()
        swing_low  = self._last_swing_low()

        risk_amount = self.equity * self.risk_pct

        if not np.isnan(swing_high) and price > swing_high:
            if np.isnan(swing_low):
                return
            stop_dist = max(price - swing_low, atr)
            size      = max(1, int(risk_amount / stop_dist))
            self.buy(size=size, sl=swing_low,
                     tp=price + self.rr * (price - swing_low))

        elif not np.isnan(swing_low) and price < swing_low:
            if np.isnan(swing_high):
                return
            stop_dist = max(swing_high - price, atr)
            size      = max(1, int(risk_amount / stop_dist))
            self.sell(size=size, sl=swing_high,
                      tp=price - self.rr * (swing_high - price))


# ── Run tests ─────────────────────────────────────────────────────────────────
INSTRUMENTS = [
    ("UK100  FTSE",    "UK100_GBP"),
    ("NAS100 Tech",    "NAS100_USD"),
    ("SPX500 S&P",     "SPX500_USD"),
    ("US30   Dow",     "US30_USD"),
    ("DE30   DAX",     "DE30_EUR"),
    ("XAU/USD Gold",   "XAU_USD"),
    ("GBP/USD",        "GBP_USD"),
    ("GBP/JPY",        "GBP_JPY"),
    ("USD/JPY",        "USD_JPY"),
    ("BCO    Brent",   "BCO_USD"),
]

STRATEGIES = [
    ("MA Pullback Swing", MAPullbackSwing, ["4h", "1d"]),
    ("H4 EMA Crossover",  H4EMACross,      ["4h"]),
    ("Swing Structure",   SwingStructure,  ["4h", "1d"]),
]

print("\nSwing Trading Backtest — 5y, 0.25% risk/trade")

for strat_label, cls, timeframes in STRATEGIES:
    for tf in timeframes:
        print(f"\n── {strat_label} @ {tf} {'─'*(50-len(strat_label))}")
        print(f"{'Instrument':<20} {'Sharpe':>7} {'AnnRet%':>9} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7}  Verdict")
        print("-" * 72)
        for label, instr in INSTRUMENTS:
            try:
                df = load_oanda_data(instr, period="5y", interval=tf)
                bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                              margin=1/30, finalize_trades=True)
                s  = bt.run()
                sharpe   = float(s.get("Sharpe Ratio", 0) or 0)
                ann_ret  = float(s.get("Return (Ann.) [%]", 0) or 0)
                max_dd   = float(s.get("Max. Drawdown [%]", 0) or 0)
                win_rate = float(s.get("Win Rate [%]", 0) or 0)
                n_trades = int(s.get("# Trades", 0) or 0)
                flag = "✅" if sharpe >= 1.0 and ann_ret > 0 else ("⚠️" if sharpe >= 0.5 and ann_ret > 0 else "❌")
                print(f"{label:<20} {sharpe:>7.2f} {ann_ret:>8.1f}% {max_dd:>7.1f}% {win_rate:>8.1f}% {n_trades:>7}  {flag}")
            except Exception as e:
                print(f"{label:<20}  ERROR: {e}")
