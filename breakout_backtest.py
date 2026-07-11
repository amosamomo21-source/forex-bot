"""
Breakout strategy comparison — three approaches tested across all instruments.

1. BB Squeeze Breakout (TTM Squeeze):
   - Bollinger Bands narrow inside Keltner Channel = "squeeze" (coiled spring)
   - On squeeze release, trade the breakout direction
   - Exit: ATR trailing stop

2. Volatility Expansion Breakout:
   - ATR expands significantly (big move bar) in a trend direction
   - Enter on momentum candle, trail out
   - Filters choppy markets

3. Consolidation Range Breakout:
   - Detect tight N-bar range (low - high spread < X * ATR)
   - Trade break of that range
   - TP = 2x range, SL = opposite edge
"""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from data import load_oanda_data


def _atr(high, low, close, period=14):
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.maximum(h[1:]-l[1:],
         np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values

def _sma(arr, n):
    return pd.Series(arr).rolling(n).mean().values

def _ema(arr, n):
    return pd.Series(arr).ewm(span=n, adjust=False).mean().values


# ── 1. BB Squeeze Breakout ────────────────────────────────────────────────────
class BBSqueeze(Strategy):
    """
    Bollinger Band Squeeze: BB contracts inside Keltner Channel, then
    price breaks out. Classic coiled-spring setup.
    """
    bb_per   = 20
    bb_std   = 2.0
    kc_mult  = 1.5
    atr_per  = 14
    trail_mult = 2.0
    risk_pct = 0.0025

    def init(self):
        cl = self.data.Close
        hi = self.data.High
        lo = self.data.Low

        sma       = self.I(_sma, cl, self.bb_per, name='SMA')
        atr       = self.I(_atr, hi, lo, cl, self.atr_per, name='ATR')

        std_arr   = self.I(lambda c: pd.Series(c).rolling(self.bb_per).std().values,
                           cl, name='Std')
        self.bb_up  = self.I(lambda s, sd: s + self.bb_std * sd, sma, std_arr, name='BB_up')
        self.bb_dn  = self.I(lambda s, sd: s - self.bb_std * sd, sma, std_arr, name='BB_dn')
        self.kc_up  = self.I(lambda s, a: s + self.kc_mult * a, sma, atr, name='KC_up')
        self.kc_dn  = self.I(lambda s, a: s - self.kc_mult * a, sma, atr, name='KC_dn')
        self.sma    = sma
        self.atr    = atr

        self.peak   = np.nan
        self.trough = np.nan

    def _in_squeeze(self, i=-1):
        return (self.bb_up[i] < self.kc_up[i] and
                self.bb_dn[i] > self.kc_dn[i])

    def next(self):
        price = float(self.data.Close[-1])
        atr   = float(self.atr[-1])
        if atr <= 0 or np.isnan(atr):
            return

        # Manage trailing stop
        if self.position.is_long:
            self.peak = max(self.peak, price)
            if price < self.peak - self.trail_mult * atr:
                self.position.close()
                self.peak = self.trough = np.nan
            return
        if self.position.is_short:
            self.trough = min(self.trough, price)
            if price > self.trough + self.trail_mult * atr:
                self.position.close()
                self.peak = self.trough = np.nan
            return

        # Entry: squeeze was on previous bar, now released (BB outside KC)
        was_squeeze = self._in_squeeze(-2)
        now_free    = not self._in_squeeze(-1)
        if not (was_squeeze and now_free):
            return

        stop_dist   = self.trail_mult * atr
        risk_amount = self.equity * self.risk_pct
        size        = max(1, int(risk_amount / stop_dist))

        # Direction: price vs SMA at squeeze release
        if price > float(self.bb_up[-1]):
            self.peak = price
            self.buy(size=size)
        elif price < float(self.bb_dn[-1]):
            self.trough = price
            self.sell(size=size)


# ── 2. Volatility Expansion Breakout ─────────────────────────────────────────
class VolExpansion(Strategy):
    """
    Enter when a single bar's body is > vol_mult * ATR (momentum candle).
    Confirms trend is accelerating. Trail out with ATR stop.
    """
    atr_per    = 14
    vol_mult   = 1.5   # candle body must be > this * ATR
    trail_mult = 2.0
    risk_pct   = 0.0025

    def init(self):
        self.atr = self.I(_atr, self.data.High, self.data.Low,
                          self.data.Close, self.atr_per, name='ATR')
        self.peak   = np.nan
        self.trough = np.nan

    def next(self):
        price  = float(self.data.Close[-1])
        open_  = float(self.data.Open[-1])
        atr    = float(self.atr[-1])
        if atr <= 0 or np.isnan(atr):
            return

        body = price - open_  # positive = bull candle, negative = bear

        if self.position.is_long:
            self.peak = max(self.peak, price)
            if price < self.peak - self.trail_mult * atr:
                self.position.close()
                self.peak = self.trough = np.nan
            return
        if self.position.is_short:
            self.trough = min(self.trough, price)
            if price > self.trough + self.trail_mult * atr:
                self.position.close()
                self.peak = self.trough = np.nan
            return

        stop_dist   = self.trail_mult * atr
        risk_amount = self.equity * self.risk_pct
        size        = max(1, int(risk_amount / stop_dist))

        if body > self.vol_mult * atr:
            self.peak = price
            self.buy(size=size)
        elif body < -self.vol_mult * atr:
            self.trough = price
            self.sell(size=size)


# ── 3. Consolidation Range Breakout ──────────────────────────────────────────
class RangeBreakout(Strategy):
    """
    Find N-bar consolidation (tight range < squeeze_pct * ATR),
    then trade the breakout of that range.
    TP = 2x range width, SL = opposite edge.
    """
    lookback    = 10
    squeeze_pct = 1.5   # range must be < squeeze_pct * ATR to qualify
    rr          = 2.0
    atr_per     = 14
    risk_pct    = 0.0025

    def init(self):
        self.atr = self.I(_atr, self.data.High, self.data.Low,
                          self.data.Close, self.atr_per, name='ATR')

    def next(self):
        if len(self.data.Close) < self.lookback + 1:
            return
        if self.position:
            return

        atr = float(self.atr[-1])
        if atr <= 0 or np.isnan(atr):
            return

        # Look at the previous N bars (not current)
        highs = self.data.High[-(self.lookback+1):-1]
        lows  = self.data.Low[-(self.lookback+1):-1]
        rng_high = float(max(highs))
        rng_low  = float(min(lows))
        rng      = rng_high - rng_low

        # Tight range = consolidation
        if rng >= self.squeeze_pct * atr:
            return

        price = float(self.data.Close[-1])
        stop_dist   = rng
        if stop_dist <= 0:
            return
        risk_amount = self.equity * self.risk_pct
        size        = max(1, int(risk_amount / stop_dist))

        if price > rng_high:
            self.buy(size=size,
                     sl=rng_low,
                     tp=price + self.rr * rng)
        elif price < rng_low:
            self.sell(size=size,
                      sl=rng_high,
                      tp=price - self.rr * rng)


# ── Instruments ───────────────────────────────────────────────────────────────
INSTRUMENTS = [
    ("UK100  FTSE",      "UK100_GBP"),
    ("NAS100 Tech",      "NAS100_USD"),
    ("SPX500 S&P",       "SPX500_USD"),
    ("US30   Dow",       "US30_USD"),
    ("DE30   DAX",       "DE30_EUR"),
    ("XAU/USD Gold",     "XAU_USD"),
    ("GBP/USD",          "GBP_USD"),
    ("GBP/JPY",          "GBP_JPY"),
    ("USD/JPY",          "USD_JPY"),
    ("BCO    Brent",     "BCO_USD"),
    ("WTICO  WTI",       "WTICO_USD"),
]

STRATEGIES = [
    ("BB Squeeze",     BBSqueeze,     {}),
    ("Vol Expansion",  VolExpansion,  {}),
    ("Range Breakout", RangeBreakout, {}),
]

print("\nBreakout Strategy Backtest — H1, 5y, 0.25% risk/trade")

for strat_label, cls, params in STRATEGIES:
    print(f"\n── {strat_label} {'─'*(55-len(strat_label))}")
    print(f"{'Instrument':<20} {'Sharpe':>7} {'AnnRet%':>9} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7}  Verdict")
    print("-" * 72)
    for label, instr in INSTRUMENTS:
        try:
            df = load_oanda_data(instr, period="5y", interval="1h")
            bt = Backtest(df, cls, cash=100_000, commission=0.0002,
                          margin=1/30, finalize_trades=True)
            s  = bt.run(**params)
            sharpe   = float(s.get("Sharpe Ratio", 0) or 0)
            ann_ret  = float(s.get("Return (Ann.) [%]", 0) or 0)
            max_dd   = float(s.get("Max. Drawdown [%]", 0) or 0)
            win_rate = float(s.get("Win Rate [%]", 0) or 0)
            n_trades = int(s.get("# Trades", 0) or 0)
            flag = "✅" if sharpe >= 1.0 and ann_ret > 0 else ("⚠️" if sharpe >= 0.5 and ann_ret > 0 else "❌")
            print(f"{label:<20} {sharpe:>7.2f} {ann_ret:>8.1f}% {max_dd:>7.1f}% {win_rate:>8.1f}% {n_trades:>7}  {flag}")
        except Exception as e:
            print(f"{label:<20}  ERROR: {e}")
