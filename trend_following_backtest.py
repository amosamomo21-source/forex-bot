"""
Donchian Channel Trend Following — buys N-bar highs, sells N-bar lows,
exits on ATR trailing stop. Tests UK100 + all live instruments.

Classic trend following logic:
  BUY  when price closes above highest high of last N bars
  SELL when price closes below lowest low of last N bars
  Exit when trailing stop (ATR * mult) is hit
"""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from data import load_oanda_data


def _ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean().values

def _atr(high, low, close, period):
    h, l, c = np.array(high), np.array(low), np.array(close)
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


class DonchianTrend(Strategy):
    """
    Donchian channel breakout with ATR trailing stop.
    Entry: close breaks above/below N-bar high/low.
    Exit:  ATR trailing stop (atr_mult * ATR from entry peak/trough).
    """
    n        = 20       # lookback period for channel
    atr_per  = 14       # ATR period
    atr_mult = 2.5      # trail stop distance in ATRs
    risk_pct = 0.0025   # 0.25% risk per trade

    def init(self):
        hi = self.data.High
        lo = self.data.Low
        cl = self.data.Close

        # N-bar high and low (Donchian channel)
        self.don_high = self.I(
            lambda h: pd.Series(h).rolling(self.n).max().values, hi, name='DonHigh')
        self.don_low  = self.I(
            lambda l: pd.Series(l).rolling(self.n).min().values, lo, name='DonLow')

        # ATR for position sizing and trailing stop
        self.atr = self.I(_atr, hi, lo, cl, self.atr_per, name='ATR')

        self.peak   = np.nan   # highest price since long entry
        self.trough = np.nan   # lowest price since short entry

    def next(self):
        price = float(self.data.Close[-1])
        atr   = float(self.atr[-1])
        if atr <= 0 or np.isnan(atr):
            return

        # Use previous bar's channel — current bar's price can't exceed its own high
        don_high = float(self.don_high[-2])
        don_low  = float(self.don_low[-2])
        if np.isnan(don_high) or np.isnan(don_low):
            return

        # ── Manage existing position ──────────────────────────────────────────
        if self.position.is_long:
            self.peak = max(self.peak, price)
            trail_sl  = self.peak - self.atr_mult * atr
            if price < trail_sl:
                self.position.close()
                self.peak   = np.nan
                self.trough = np.nan
            return

        if self.position.is_short:
            self.trough = min(self.trough, price)
            trail_sl    = self.trough + self.atr_mult * atr
            if price > trail_sl:
                self.position.close()
                self.peak   = np.nan
                self.trough = np.nan
            return

        # ── Entry signals ─────────────────────────────────────────────────────
        stop_dist   = self.atr_mult * atr
        risk_amount = self.equity * self.risk_pct
        size        = max(1, int(risk_amount / stop_dist))

        # Buy breakout: price closes above N-bar high
        if price > don_high:
            self.peak = price
            self.buy(size=size)

        # Sell breakout: price closes below N-bar low
        elif price < don_low:
            self.trough = price
            self.sell(size=size)


# ── Instruments to test ───────────────────────────────────────────────────────
TESTS = [
    # UK100 — the main question
    ("UK100  FTSE",         "UK100_GBP",   "1h"),
    ("UK100  FTSE",         "UK100_GBP",   "1d"),
    # Our live indices (benchmark — do we beat EMA?)
    ("NAS100 US Tech",      "NAS100_USD",  "1h"),
    ("SPX500 S&P 500",      "SPX500_USD",  "1h"),
    ("US30   Dow Jones",    "US30_USD",    "1h"),
    ("DE30   DAX",          "DE30_EUR",    "1h"),
    # FX pairs
    ("GBP/USD",             "GBP_USD",     "1h"),
    ("USD/JPY",             "USD_JPY",     "1h"),
    ("EUR/JPY",             "EUR_JPY",     "1h"),
    # Energy (already in bot)
    ("BCO    Brent",        "BCO_USD",     "1h"),
    ("WTICO  WTI",          "WTICO_USD",   "1h"),
    # Metals
    ("XAU/USD Gold",        "XAU_USD",     "1h"),
]

PARAM_SETS = [
    ("N=20 ATR=2.5", dict(n=20, atr_mult=2.5)),
    ("N=50 ATR=3.0", dict(n=50, atr_mult=3.0)),
]

print("\nDonchian Trend Following Backtest — 5y, 0.25% risk/trade")

for param_label, params in PARAM_SETS:
    print(f"\n── {param_label} ──────────────────────────────────────────────")
    print(f"{'Instrument':<22} {'TF':<5} {'Sharpe':>7} {'AnnRet%':>9} {'MaxDD%':>8} {'WinRate%':>9} {'Trades':>7}  Verdict")
    print("-" * 82)

    for label, instr, tf in TESTS:
        try:
            df = load_oanda_data(instr, period="5y", interval=tf)
            bt = Backtest(df, DonchianTrend,
                          cash=100_000, commission=0.0002,
                          margin=1/30, finalize_trades=True)
            s  = bt.run(**params, risk_pct=0.0025)
            sharpe   = float(s.get("Sharpe Ratio", 0) or 0)
            ann_ret  = float(s.get("Return (Ann.) [%]", 0) or 0)
            max_dd   = float(s.get("Max. Drawdown [%]", 0) or 0)
            win_rate = float(s.get("Win Rate [%]", 0) or 0)
            n_trades = int(s.get("# Trades", 0) or 0)
            flag = "✅" if sharpe >= 1.0 and ann_ret > 0 else ("⚠️" if sharpe >= 0.5 and ann_ret > 0 else "❌")
            print(f"{label:<22} {tf:<5} {sharpe:>7.2f} {ann_ret:>8.1f}% {max_dd:>7.1f}% {win_rate:>8.1f}% {n_trades:>7}  {flag}")
        except Exception as e:
            print(f"{label:<22} {tf:<5}  ERROR: {e}")

print("\nNote: Win rate for trend following is typically 30-40% — that's normal.")
print("What matters is large wins vs small losses (high profit factor).")
