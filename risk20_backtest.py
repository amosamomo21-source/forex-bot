"""Simulate the full 10-sleeve portfolio at 20% risk per trade on $100k demo account."""
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import atr, ema, rsi, rolling_std, sma, BollingerMeanReversionTrendFilter as Strat

RISK_PCT    = 0.20
INITIAL_EQ  = 100_000
N_SLEEVES   = 10
SLEEVE_EQ   = INITIAL_EQ / N_SLEEVES  # $10,000 each

M30_PAIRS   = ["EUR_USD", "GBP_USD", "EUR_JPY", "CHF_JPY", "AUD_CHF"]
DAILY_PAIRS = [("EUR_USD","bbmrt"), ("GBP_USD","bbmrt"),
               ("GBP_USD","ema"),   ("USD_JPY","ema"),   ("AUD_USD","ema")]
PERIOD = "5y"

def run_m30(instrument):
    daily = load_oanda_data(instrument, period=PERIOD, interval="1d")
    m30   = load_oanda_data(instrument, period=PERIOD, interval="30m")
    if daily is None or m30 is None or len(daily) < Strat.trend_period + 5:
        return pd.Series(dtype=float)

    bb_mid = sma(daily["Close"], Strat.bb_period)
    bb_std = rolling_std(daily["Close"], Strat.bb_period)
    trend  = ema(daily["Close"], Strat.trend_period)
    atr_d  = atr(daily["High"], daily["Low"], daily["Close"], Strat.atr_period)
    lower  = bb_mid - Strat.bb_k * bb_std
    upper  = bb_mid + Strat.bb_k * bb_std

    long_cond  = (daily["Close"].shift(1) < lower.shift(1)) & (daily["Close"].shift(1) > trend.shift(1))
    short_cond = (daily["Close"].shift(1) > upper.shift(1)) & (daily["Close"].shift(1) < trend.shift(1))
    bias = pd.Series(0, index=daily.index)
    bias[long_cond] = 1; bias[short_cond] = -1

    daily_bias = bias.reindex(m30.index, method="ffill")
    daily_atr  = atr_d.shift(1).reindex(m30.index, method="ffill")
    daily_mid  = bb_mid.shift(1).reindex(m30.index, method="ffill")
    m30_rsi    = rsi(m30["Close"], 14)
    rsi_prev   = m30_rsi.shift(1)

    equity = SLEEVE_EQ
    position = 0; entry_price = sl = tp = 0.0
    trades = []

    for i in range(1, len(m30)):
        price = m30["Close"].iloc[i]; b = daily_bias.iloc[i]
        a = daily_atr.iloc[i]; mid = daily_mid.iloc[i]
        rn = m30_rsi.iloc[i]; rp = rsi_prev.iloc[i]
        if np.isnan(a) or a <= 0 or np.isnan(rn) or np.isnan(rp): continue

        if position != 0:
            sl_hit = (position==1 and price<=sl) or (position==-1 and price>=sl)
            tp_hit = (position==1 and price>=tp) or (position==-1 and price<=tp)
            mean_hit = (position==1 and price>=mid) or (position==-1 and price<=mid)
            if sl_hit or tp_hit or mean_hit:
                exit_p = sl if sl_hit else (tp if tp_hit else price)
                pnl_r  = position * (exit_p - entry_price) / abs(entry_price - sl)
                pnl    = pnl_r * equity * RISK_PCT
                equity += pnl
                trades.append({"date": m30.index[i], "pnl": pnl, "sleeve": instrument})
                position = 0
            continue

        stop_dist = 2.0 * a
        if b==1 and rp<40 and rn>=40:
            position=1; entry_price=price; sl=price-stop_dist; tp=price+3.0*a
        elif b==-1 and rp>60 and rn<=60:
            position=-1; entry_price=price; sl=price+stop_dist; tp=price-3.0*a

    return pd.DataFrame(trades)

all_trades = []
for pair in M30_PAIRS:
    print(f"  M30  {pair}...")
    df = run_m30(pair)
    if len(df): all_trades.append(df)

if not all_trades:
    print("No trades generated")
else:
    trades = pd.concat(all_trades).sort_values("date").reset_index(drop=True)

    total_pnl  = trades["pnl"].sum()
    n          = len(trades)
    wins       = (trades["pnl"] > 0).sum()
    eq_curve   = INITIAL_EQ + trades["pnl"].cumsum()
    peak       = eq_curve.cummax()
    max_dd     = ((eq_curve - peak) / peak).min()
    final_eq   = INITIAL_EQ + total_pnl
    ann_ret    = (final_eq / INITIAL_EQ) ** (1/5) - 1

    print(f"\n{'='*50}")
    print(f"Portfolio simulation: 20% risk, $100k demo, 5y")
    print(f"{'='*50}")
    print(f"Starting equity:   ${INITIAL_EQ:>10,.0f}")
    print(f"Final equity:      ${final_eq:>10,.0f}")
    print(f"Total P&L:         ${total_pnl:>10,.0f}")
    print(f"Annual return:     {ann_ret:>10.1%}")
    print(f"Total trades:      {n:>10}")
    print(f"Win rate:          {wins/n:>10.0%}")
    print(f"Max drawdown:      {max_dd:>10.1%}")
    print(f"\nPer-trade stats:")
    print(f"  Avg win:         ${trades[trades['pnl']>0]['pnl'].mean():>8,.0f}")
    print(f"  Avg loss:        ${trades[trades['pnl']<0]['pnl'].mean():>8,.0f}")
    print(f"  Largest win:     ${trades['pnl'].max():>8,.0f}")
    print(f"  Largest loss:    ${trades['pnl'].min():>8,.0f}")

    print(f"\nBy sleeve:")
    for sleeve, g in trades.groupby("sleeve"):
        sl_wins = (g["pnl"]>0).sum()
        print(f"  {sleeve:<12} {len(g):>3} trades  {sl_wins/len(g):>4.0%} WR  ${g['pnl'].sum():>8,.0f} P&L")
