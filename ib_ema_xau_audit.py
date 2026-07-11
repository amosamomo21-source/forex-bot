"""
IB-EMA Breakout — XAU/USD (Gold) M30 audit  v3 — native limit orders

Uses backtesting.py's built-in limit orders so fill logic is handled correctly.
When a breakout is confirmed by the EMA filter, a limit order is placed at the
IB level (IB_high for longs, IB_low for shorts) to catch the retracement.
The order is cancelled after limit_expiry bars if not filled.

SL  = IB midpoint  (IB_range / 2 from IB_high/low)
TP  = IB_high + rr × (IB_range / 2)   [longs]
    = IB_low  − rr × (IB_range / 2)   [shorts]
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

def _ema(s, n):
    return pd.Series(np.array(s, float)).ewm(span=n, adjust=False).mean().values

class IBEMARetracement(Strategy):
    ema_fast    = 8
    ema_slow    = 21
    rr          = 2.0
    max_bars    = 8       # 4 hours post-IB: stop looking for new signals
    limit_expiry = 6      # bars to keep limit order alive (3 hours)
    risk_pct    = RISK_PCT

    def init(self):
        self.ef      = self.I(_ema, self.data.Close, self.ema_fast)
        self.es      = self.I(_ema, self.data.Close, self.ema_slow)
        self.ib_high = np.nan
        self.ib_low  = np.nan
        self.ib_mid  = np.nan
        self.ib_bar  = -999
        self.sig_bar = -999    # bar when limit order was placed
        self.traded  = False

    def _cancel_pending(self):
        for o in self.orders:
            o.cancel()

    def next(self):
        t   = self.data.index[-1]
        idx = len(self.data) - 1

        # Cancel pending limit orders that have expired
        if self.orders and self.sig_bar >= 0 and (idx - self.sig_bar) >= self.limit_expiry:
            self._cancel_pending()
            self.sig_bar = -999

        # New session IB bar → reset state
        if t.hour in (8, 13) and t.minute == 0:
            self._cancel_pending()
            h = float(self.data.High[-1])
            l = float(self.data.Low[-1])
            self.traded  = False
            self.sig_bar = -999
            if h > l and (h - l) >= 0.50:   # valid IB (min 50¢ range for Gold)
                self.ib_high = h
                self.ib_low  = l
                self.ib_mid  = (h + l) / 2
                self.ib_bar  = idx
            else:
                self.ib_high = np.nan
            return

        if np.isnan(self.ib_high) or self.traded or self.position or self.orders:
            return
        if idx - self.ib_bar > self.max_bars:
            return

        p  = float(self.data.Close[-1])
        ef = float(self.ef[-1])
        es = float(self.es[-1])
        if np.isnan(ef) or np.isnan(es):
            return

        sl_dist = (self.ib_high - self.ib_low) / 2
        if sl_dist <= 0:
            return

        ra    = self.equity * self.risk_pct
        units = max(1, int(ra / sl_dist))

        # BUY: breakout above IB high, EMA bullish → limit buy in a zone above IB_high
        # Zone: IB_high to IB_high + 0.4×range (catches partial retracements too)
        if p > self.ib_high and p > ef and p > es:
            limit_entry = self.ib_high + 0.2 * (self.ib_high - self.ib_low)
            sl   = self.ib_mid
            tp   = self.ib_high + self.rr * sl_dist
            if limit_entry > sl and tp > limit_entry:
                self.buy(size=units, limit=limit_entry, sl=sl, tp=tp)
                self.sig_bar = idx
                self.traded  = True

        # SELL: breakdown below IB low, EMA bearish → limit sell in zone below IB_low
        elif p < self.ib_low and p < ef and p < es:
            limit_entry = self.ib_low - 0.2 * (self.ib_high - self.ib_low)
            sl   = self.ib_mid
            tp   = self.ib_low - self.rr * sl_dist
            if limit_entry < sl and tp < limit_entry:
                self.sell(size=units, limit=limit_entry, sl=sl, tp=tp)
                self.sig_bar = idx
                self.traded  = True

class IBEMARetracement3R(IBEMARetracement):
    rr = 3.0

# ── Run ──────────────────────────────────────────────────────────────────────
print('IB-EMA Breakout (retracement entry) — XAU/USD Gold, M30, 10y')
print('Sessions: London 08:00 UTC + NY 13:00 UTC')
print('EMA(8/21) filter | SL=IB mid | TP=2×/3× risk | max 1 trade/session')
print()

df = load_oanda_data('XAU_USD', period='10y', interval='30m')
print(f'Data: {len(df)} bars  ({df.index[0].date()} → {df.index[-1].date()})')
print()
print(f"{'':10} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  EV/R   BE-WR  Verdict")
print('-' * 80)

for label, cls in [('RR 2:1', IBEMARetracement), ('RR 3:1', IBEMARetracement3R)]:
    b  = Backtest(df, cls, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
    s  = b.run()
    sh  = float(s.get('Sharpe Ratio', 0) or 0)
    ann = float(s.get('Return (Ann.) [%]', 0) or 0)
    dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
    wr  = float(s.get('Win Rate [%]', 0) or 0)
    n   = int(s.get('# Trades', 0) or 0)
    pf  = float(s.get('Profit Factor', 0) or 0)
    rr  = cls.rr
    be  = 100 / (rr + 1)
    ev  = (wr/100) * rr - (1 - wr/100)
    ok  = sh >= 0.4 and ann > 0 and n >= 30
    verdict = 'ADD' if ok else ('WATCH' if sh >= 0.2 and ann > 0 else 'PASS')
    print(f'{label:<10} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}  {ev:+.3f}R  {be:.0f}%    {verdict}')
