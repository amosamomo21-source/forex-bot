"""
ORB M30 audit — all 10 live sleeves, 10y per instrument.
Same logic as the live runner: entry when M30 bar following 08:00 or 13:00 UTC
breaks above/below the opening range. SL = opposite end of range, TP = 1.5× range.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

class ORB(Strategy):
    risk_pct = RISK_PCT
    tp_mult  = 1.5

    def init(self): pass

    def next(self):
        if self.position or len(self.data) < 2: return
        t_prev = self.data.index[-2]
        if t_prev.hour not in (8, 13) or t_prev.minute != 0: return
        or_high = float(self.data.High[-2])
        or_low  = float(self.data.Low[-2])
        or_range = or_high - or_low
        if or_range <= 0: return
        p  = float(self.data.Close[-1])
        ra = self.equity * self.risk_pct
        units = min(max(1, int(ra / or_range)), 50_000)
        if p > or_high:
            self.buy( size=units, sl=or_low,  tp=p + self.tp_mult * or_range)
        elif p < or_low:
            self.sell(size=units, sl=or_high, tp=p - self.tp_mult * or_range)

SLEEVES = [
    ('orb_m30_eurjpy', 'EUR_JPY'),
    ('orb_m30_chfjpy', 'CHF_JPY'),
    ('orb_m30_cadjpy', 'CAD_JPY'),
    ('orb_m30_audjpy', 'AUD_JPY'),
    ('orb_m30_gbpjpy', 'GBP_JPY'),
    ('orb_m30_nzdjpy', 'NZD_JPY'),
    ('orb_m30_audchf', 'AUD_CHF'),
    ('orb_m30_euraud', 'EUR_AUD'),
    ('orb_m30_usdjpy', 'USD_JPY'),
    ('orb_m30_eurcad', 'EUR_CAD'),
]

print('ORB M30 audit — 10y, London (08:00 UTC) + NY (13:00 UTC) sessions')
print(f"{'Sleeve':<22} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} {'N':>5} {'PF':>5}  EV/R   Verdict")
print('─' * 80)

rows = []
for tag, instr in SLEEVES:
    try:
        df = load_oanda_data(instr, period='10y', interval='30m')
        b  = Backtest(df, ORB, cash=100_000, commission=0.0002, margin=1/30, finalize_trades=True)
        s  = b.run()
        sh  = float(s.get('Sharpe Ratio', 0) or 0)
        ann = float(s.get('Return (Ann.) [%]', 0) or 0)
        dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
        wr  = float(s.get('Win Rate [%]', 0) or 0)
        n   = int(s.get('# Trades', 0) or 0)
        pf  = float(s.get('Profit Factor', 0) or 0)
        # EV in R: WR×1.5R − (1−WR)×1R; break-even WR = 40%
        ev  = (wr/100) * 1.5 - (1 - wr/100) * 1.0
        ok  = sh >= 0.3 and ann > 0
        verdict = 'KEEP' if ok else ('WATCH' if sh >= 0.0 and ann > 0 else 'REMOVE')
        print(f'{tag:<22} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% {wr:>5.1f}% {n:>5} {pf:>5.2f}  {ev:+.3f}R  {verdict}')
        rows.append({'sh': sh, 'ann': ann, 'wr': wr, 'n': n, 'verdict': verdict})
    except Exception as e:
        print(f'{tag:<22} ERROR: {e}')

if rows:
    total_n   = sum(r['n'] for r in rows)
    total_wins = sum(int(r['n'] * r['wr'] / 100) for r in rows)
    agg_wr    = total_wins / total_n * 100 if total_n else 0
    avg_sh    = sum(r['sh'] for r in rows) / len(rows)
    n_keep    = sum(1 for r in rows if r['verdict'] == 'KEEP')
    n_remove  = sum(1 for r in rows if r['verdict'] == 'REMOVE')
    print('─' * 80)
    print(f'Portfolio ORB: avg Sharpe {avg_sh:+.2f} | agg WR {agg_wr:.1f}% | {n_keep} KEEP, {n_remove} REMOVE')
    print(f'Break-even WR at 1.5:1 R:R = 40.0%')
