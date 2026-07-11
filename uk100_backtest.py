"""
UK100 strategy search — ORB and PDHL with proper risk-based position sizing.
"""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from backtesting import Backtest, Strategy
from data import load_oanda_data


class UK100_ORB(Strategy):
    """
    Opening Range Breakout — 08:00 UTC bar defines the range,
    trade breakouts 09:00-15:00 UTC with risk-based sizing.
    """
    risk_pct = 0.0025
    rr       = 1.5

    def init(self):
        self.or_high   = np.nan
        self.or_low    = np.nan
        self.or_set    = False
        self.last_date = None

    def next(self):
        t    = self.data.index[-1]
        hour = t.hour
        date = t.date()

        if date != self.last_date:
            self.or_high   = np.nan
            self.or_low    = np.nan
            self.or_set    = False
            self.last_date = date

        if hour == 8 and not self.or_set:
            self.or_high = float(self.data.High[-1])
            self.or_low  = float(self.data.Low[-1])
            self.or_set  = True
            return

        if not self.or_set or np.isnan(self.or_high) or self.position:
            return
        if hour < 9 or hour > 15:
            return

        price    = float(self.data.Close[-1])
        or_range = self.or_high - self.or_low
        if or_range <= 0:
            return

        risk_amount = self.equity * self.risk_pct

        if price > self.or_high:
            stop_dist = price - self.or_low
            size = max(1, int(risk_amount / stop_dist))
            self.buy(size=size, sl=self.or_low, tp=price + self.rr * or_range)
        elif price < self.or_low:
            stop_dist = self.or_high - price
            size = max(1, int(risk_amount / stop_dist))
            self.sell(size=size, sl=self.or_high, tp=price - self.rr * or_range)


class UK100_PDHL(Strategy):
    """
    Previous Day High/Low breakout with risk-based sizing.
    """
    risk_pct = 0.0025
    rr       = 1.5

    def init(self):
        self.prev_high = np.nan
        self.prev_low  = np.nan
        self.last_date = None
        self.day_high  = -np.inf
        self.day_low   =  np.inf

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

        risk_amount = self.equity * self.risk_pct

        if price > self.prev_high:
            stop_dist = price - prev_mid
            if stop_dist <= 0:
                return
            size = max(1, int(risk_amount / stop_dist))
            self.buy(size=size, sl=prev_mid, tp=price + self.rr * prev_rng)
        elif price < self.prev_low:
            stop_dist = prev_mid - price
            if stop_dist <= 0:
                return
            size = max(1, int(risk_amount / stop_dist))
            self.sell(size=size, sl=prev_mid, tp=price - self.rr * prev_rng)


print("\nUK100 Strategy Search — 5y, 0.25% risk/trade")
print(f"\n{'Strategy':<28} {'TF':<6} {'Sharpe':>7} {'AnnRet%':>9} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7}  Verdict")
print("-" * 88)

tests = [
    ("ORB (London 08:00 UTC)", "30m", UK100_ORB),
    ("ORB (London 08:00 UTC)", "1h",  UK100_ORB),
    ("PDHL Breakout",          "1h",  UK100_PDHL),
    ("PDHL Breakout",          "1d",  UK100_PDHL),
]

for label, interval, cls in tests:
    try:
        df = load_oanda_data("UK100_GBP", period="5y", interval=interval)
        bt = Backtest(df, cls, cash=100_000, commission=0.0002, margin=1/20, finalize_trades=True)
        s  = bt.run()
        sharpe   = float(s.get("Sharpe Ratio", 0) or 0)
        ann_ret  = float(s.get("Return (Ann.) [%]", 0) or 0)
        max_dd   = float(s.get("Max. Drawdown [%]", 0) or 0)
        win_rate = float(s.get("Win Rate [%]", 0) or 0)
        n_trades = int(s.get("# Trades", 0) or 0)
        flag = "✅ ADD" if sharpe >= 1.0 and ann_ret > 0 else ("⚠️  WATCH" if sharpe >= 0.5 and ann_ret > 0 else "❌ SKIP")
        print(f"{label:<28} {interval:<6} {sharpe:>7.2f} {ann_ret:>8.1f}% {max_dd:>7.1f}% {win_rate:>8.1f}% {n_trades:>7}  {flag}")
    except Exception as e:
        print(f"{label:<28} {interval:<6}  ERROR: {e}")
