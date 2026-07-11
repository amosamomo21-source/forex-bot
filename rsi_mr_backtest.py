import sys
sys.path.insert(0, "/Users/bamznizzy/forex-bot")

from dotenv import load_dotenv
load_dotenv("/Users/bamznizzy/forex-bot/.env")

import numpy as np
import pandas as pd
from data import load_oanda_data
from strategies import atr, rsi

RSI_PERIOD   = 14
OVERSOLD     = 30
OVERBOUGHT   = 70
SL_ATR_MULT  = 2.0
TP_ATR_MULT  = 3.0
RISK_PCT     = 0.01
INITIAL_EQ   = 10_000

def backtest(ticker, period="5y"):
    df = load_oanda_data(ticker, period=period, interval="1d")
    if df is None or len(df) < RSI_PERIOD + 5:
        return None

    rsi_s = rsi(df["Close"], RSI_PERIOD)
    atr_s = atr(df["High"], df["Low"], df["Close"], RSI_PERIOD)

    # shift to prevent lookahead: signals use yesterday's completed bar
    rsi_prev = rsi_s.shift(1)
    rsi_now  = rsi_s
    atr_now  = atr_s.shift(1)

    equity = INITIAL_EQ
    position = 0   # 1 = long, -1 = short, 0 = flat
    entry_price = sl = tp = 0.0
    entry_rsi = 0.0
    trades = []

    for i in range(1, len(df)):
        price  = df["Close"].iloc[i]
        r_prev = rsi_prev.iloc[i]
        r_now  = rsi_now.iloc[i]
        a      = atr_now.iloc[i]

        if np.isnan(r_prev) or np.isnan(r_now) or np.isnan(a) or a <= 0:
            continue

        if position != 0:
            # check exits
            stop_hit = (position == 1 and price <= sl) or (position == -1 and price >= sl)
            tp_hit   = (position == 1 and price >= tp) or (position == -1 and price <= tp)
            rsi_mid  = (position == 1 and r_now >= 50) or (position == -1 and r_now <= 50)

            if stop_hit or tp_hit or rsi_mid:
                if stop_hit:
                    exit_price = sl
                    reason = "sl"
                elif tp_hit:
                    exit_price = tp
                    reason = "tp"
                else:
                    exit_price = price
                    reason = "rsi_mid"

                pnl_r = position * (exit_price - entry_price) / (abs(entry_price - sl))
                risk_amount = equity * RISK_PCT
                pnl = pnl_r * risk_amount
                equity += pnl
                trades.append({"pnl": pnl, "pnl_r": pnl_r, "reason": reason})
                position = 0
            else:
                continue

        # entry
        long_signal  = r_prev < OVERSOLD  and r_now >= OVERSOLD
        short_signal = r_prev > OVERBOUGHT and r_now <= OVERBOUGHT

        if long_signal:
            position    = 1
            entry_price = price
            sl          = price - SL_ATR_MULT * a
            tp          = price + TP_ATR_MULT * a
        elif short_signal:
            position    = -1
            entry_price = price
            sl          = price + SL_ATR_MULT * a
            tp          = price - TP_ATR_MULT * a

    if not trades:
        return None

    df_t    = pd.DataFrame(trades)
    n       = len(df_t)
    wins    = (df_t["pnl"] > 0).sum()
    total   = df_t["pnl"].sum()
    eq_curve = INITIAL_EQ + df_t["pnl"].cumsum()
    peak    = eq_curve.cummax()
    dd      = ((eq_curve - peak) / peak).min()
    annual_ret = (equity / INITIAL_EQ) ** (1 / (int(period[0]))) - 1
    daily_ret = df_t["pnl"] / INITIAL_EQ
    sharpe  = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    return {
        "ticker": ticker, "period": period,
        "trades": n, "win_rate": wins/n,
        "total_pnl": total, "annual_ret": annual_ret,
        "sharpe": sharpe, "max_dd": dd,
    }

instruments = ["EUR_USD", "GBP_USD", "XAU_USD", "USD_JPY", "AUD_USD"]
print(f"{'Instrument':<14} {'Period':<6} {'Trades':>6} {'Win%':>6} {'Sharpe':>7} {'Ann%':>7} {'MaxDD':>7}")
print("-" * 60)
for inst in instruments:
    for period in ["30d"]:
        r = backtest(inst, period)
        if r:
            print(f"{r['ticker']:<14} {r['period']:<6} {r['trades']:>6} {r['win_rate']:>6.0%} {r['sharpe']:>7.2f} {r['annual_ret']:>7.1%} {r['max_dd']:>7.1%}")
        else:
            print(f"{inst:<14} {period:<6} {'no data':>6}")
