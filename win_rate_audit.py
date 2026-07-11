"""
Win rate audit — all 4 live agents, current 27-sleeve portfolio.

Agent → Strategy type:
  com.bamznizzy.forex-bot      → BBMRT D1, EMA D1, PDHL NATGAS, CONSEC WHEAT
  com.bamznizzy.forex-bot.m30  → BBMRT M30 (17 pairs)
  com.bamznizzy.forex-bot.h1   → PDHL NATGAS (H1 entry, shared sleeve)
  com.bamznizzy.forex-bot.fvg  → FVG M30 (NATGAS, GBP/CHF)
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
from strategies import BollingerMeanReversionTrendFilter, EmaCrossoverAtr, atr, ema, rsi, rolling_std, sma
import numpy as np
import pandas as pd

RISK_PCT = 0.0025

# ── Shared helpers ─────────────────────────────────────────────────────────────
def _ema(s, n):
    return pd.Series(np.array(s, float)).ewm(span=n, adjust=False).mean().values

def _run(df, cls, label):
    try:
        b = Backtest(df, cls, cash=100_000, commission=0.0002,
                     margin=1/30, finalize_trades=True)
        s = b.run()
        sh  = float(s.get('Sharpe Ratio', 0) or 0)
        ann = float(s.get('Return (Ann.) [%]', 0) or 0)
        wr  = float(s.get('Win Rate [%]', 0) or 0)
        n   = int(s.get('# Trades', 0) or 0)
        pf  = float(s.get('Profit Factor', 0) or 0)
        return sh, ann, wr, n, pf
    except Exception as e:
        print(f'  ERROR {label}: {e}')
        return None

def _row(label, sh, ann, wr, n, pf, be_wr):
    margin = wr - be_wr
    verdict = 'PASS' if wr >= be_wr and ann > 0 and sh > 0 else 'FAIL'
    return f'{label:<28} {sh:>+6.2f} {ann:>+6.1f}% {wr:>6.1f}%  {margin:>+5.1f}pp  {n:>5}  {pf:>5.2f}  {verdict}'

HDR = f"{'Sleeve / Strategy':<28} {'Sh':>6} {'Ann%':>7} {'WR%':>7}  {'vs BE':>6}  {'N':>5}  {'PF':>5}  Verdict"
SEP = '─' * 80

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — com.bamznizzy.forex-bot  (live_runner.py, fires 22:30 UTC)
# Strategies: BBMRT D1, EMA D1, PDHL NATGAS, CONSEC WHEAT
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('AGENT 1: com.bamznizzy.forex-bot  (live_runner.py, 22:30 UTC)')
print('BBMRT D1, EMA D1, PDHL NATGAS, CONSEC WHEAT')
print(SEP)
print(HDR)
print(SEP)

class BBMRT_D1(Strategy):
    risk_pct = RISK_PCT
    def init(self):
        S = BollingerMeanReversionTrendFilter
        self.bb_mid = self.I(sma, self.data.Close, S.bb_period)
        std = self.I(rolling_std, self.data.Close, S.bb_period)
        self.upper = self.bb_mid + S.bb_k * std
        self.lower = self.bb_mid - S.bb_k * std
        self.trend = self.I(ema, self.data.Close, S.trend_period)
        self.atr_i = self.I(atr, self.data.High, self.data.Low, self.data.Close, S.atr_period)
    def next(self):
        if self.position or len(self.data) < 60: return
        p = float(self.data.Close[-1])
        av = float(self.atr_i[-1]); tr = float(self.trend[-1])
        lo = float(self.lower[-1]); hi = float(self.upper[-1])
        if av <= 0: return
        ra = self.equity * self.risk_pct
        units = max(1, int(ra / (2 * av)))
        if p < lo and p > tr:
            self.buy(size=units, sl=p - 2*av, tp=float(self.bb_mid[-1]))
        elif p > hi and p < tr:
            self.sell(size=units, sl=p + 2*av, tp=float(self.bb_mid[-1]))

D1_BBMRT = [('bbmrt_eurusd', 'EUR_USD'), ('bbmrt_gbpusd', 'GBP_USD')]
D1_EMA   = [('ema_gbpusd', 'GBP_USD'), ('ema_usdjpy', 'USD_JPY'), ('ema_audusd', 'AUD_USD')]

a1_rows = []
for tag, instr in D1_BBMRT:
    df = load_oanda_data(instr, period='10y', interval='1d')
    r  = _run(df, BBMRT_D1, tag)
    if r: a1_rows.append(r); print(_row(tag, *r, be_wr=40))

class EMA_D1(Strategy):
    risk_pct = RISK_PCT
    def init(self):
        S = EmaCrossoverAtr
        self.ef = self.I(ema, self.data.Close, S.fast)
        self.es = self.I(ema, self.data.Close, S.slow)
        self.ai = self.I(atr, self.data.High, self.data.Low, self.data.Close, S.atr_period)
    def next(self):
        if self.position: return
        ef = float(self.ef[-1]); es = float(self.es[-1])
        ef_p = float(self.ef[-2]); es_p = float(self.es[-2])
        av = float(self.ai[-1])
        if av <= 0: return
        ra = self.equity * self.risk_pct
        units = max(1, int(ra / (EmaCrossoverAtr.sl_atr_mult * av)))
        if ef_p < es_p and ef > es:
            self.buy(size=units, sl=float(self.data.Close[-1]) - EmaCrossoverAtr.sl_atr_mult*av)
        elif ef_p > es_p and ef < es:
            self.sell(size=units, sl=float(self.data.Close[-1]) + EmaCrossoverAtr.sl_atr_mult*av)

for tag, instr in D1_EMA:
    df = load_oanda_data(instr, period='10y', interval='1d')
    r  = _run(df, EMA_D1, tag)
    if r: a1_rows.append(r); print(_row(tag, *r, be_wr=50))

class PDHL(Strategy):
    risk_pct = RISK_PCT
    def init(self): pass
    def next(self):
        if self.position or len(self.data) < 3: return
        prev_high = float(self.data.High[-2])
        prev_low  = float(self.data.Low[-2])
        prev_mid  = (prev_high + prev_low) / 2
        prev_rng  = prev_high - prev_low
        if prev_rng <= 0: return
        p = float(self.data.Close[-1])
        sl_dist = abs(p - prev_mid)
        if sl_dist <= 0: return
        ra = self.equity * self.risk_pct
        units = max(1, int(ra / sl_dist))
        if p > prev_high:
            self.buy(size=units, sl=prev_mid, tp=p + 1.5*prev_rng)
        elif p < prev_low:
            self.sell(size=units, sl=prev_mid, tp=p - 1.5*prev_rng)

df_ng = load_oanda_data('NATGAS_USD', period='10y', interval='1d')
r = _run(df_ng, PDHL, 'pdhl_natgas (D1 approx)')
if r: a1_rows.append(r); print(_row('pdhl_natgas (D1 approx)', *r, be_wr=40))

class CONSEC(Strategy):
    streak   = 3
    sl_atr   = 1.5
    tp_atr   = 1.0
    risk_pct = RISK_PCT
    def init(self):
        self.atr_i = self.I(atr, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        if self.position or len(self.data) < self.streak + 20: return
        closes = [float(self.data.Close[-i-1]) for i in range(self.streak + 1)]
        all_up   = all(closes[i] > closes[i+1] for i in range(self.streak))
        all_down = all(closes[i] < closes[i+1] for i in range(self.streak))
        av = float(self.atr_i[-1])
        if av <= 0: return
        ra = self.equity * self.risk_pct
        units = max(1, int(ra / (self.sl_atr * av)))
        p = float(self.data.Close[-1])
        if all_up:
            self.sell(size=units, sl=p + self.sl_atr*av, tp=p - self.tp_atr*av)
        elif all_down:
            self.buy(size=units,  sl=p - self.sl_atr*av, tp=p + self.tp_atr*av)

class CONSEC4(CONSEC):
    streak = 4

df_w = load_oanda_data('WHEAT_USD', period='10y', interval='1d')
for tag, cls in [('consec_d1_wheatusd_3', CONSEC), ('consec_d1_wheatusd_4', CONSEC4)]:
    r = _run(df_w, cls, tag)
    if r: a1_rows.append(r); print(_row(tag, *r, be_wr=60))

print(SEP)
if a1_rows:
    avg_wr = sum(r[2] for r in a1_rows) / len(a1_rows)
    avg_sh = sum(r[0] for r in a1_rows) / len(a1_rows)
    total_n = sum(r[3] for r in a1_rows)
    wins = sum(int(r[3]*r[2]/100) for r in a1_rows)
    agg_wr = wins/total_n*100 if total_n else 0
    print(f'Agent 1 summary: {len(a1_rows)} strategies | avg Sh {avg_sh:+.2f} | '
          f'avg WR {avg_wr:.1f}% | agg WR {agg_wr:.1f}% ({total_n} trades)')

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — com.bamznizzy.forex-bot.m30  (m30_runner.py, every 30 min)
# Strategy: BBMRT M30 (17 pairs)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('AGENT 2: com.bamznizzy.forex-bot.m30  (m30_runner.py, every 30 min)')
print('BBMRT M30 — 17 pairs')
print(SEP)
print(HDR)
print(SEP)

class BBMRT_M30(Strategy):
    risk_pct = RISK_PCT
    def init(self):
        S = BollingerMeanReversionTrendFilter
        self.bb_mid = self.I(sma, self.data.Close, S.bb_period)
        std = self.I(rolling_std, self.data.Close, S.bb_period)
        self.upper = self.bb_mid + S.bb_k * std
        self.lower = self.bb_mid - S.bb_k * std
        self.trend = self.I(ema, self.data.Close, S.trend_period)
        self.rsi_i = self.I(rsi, self.data.Close, 14)
    def next(self):
        if self.position or len(self.data) < 60: return
        p = float(self.data.Close[-1])
        tr = float(self.trend[-1])
        lo = float(self.lower[-1]); hi = float(self.upper[-1])
        mid = float(self.bb_mid[-1])
        rsi_v = float(self.rsi_i[-1])
        band = hi - lo
        if band <= 0: return
        ra = self.equity * self.risk_pct
        units = max(1, int(ra / (band / 2)))
        if p < lo and p > tr and rsi_v < 40:
            self.buy(size=units, sl=p - band/2, tp=mid)
        elif p > hi and p < tr and rsi_v > 60:
            self.sell(size=units, sl=p + band/2, tp=mid)

M30_PAIRS = [
    ('bbmrt_m30_eurusd','EUR_USD'), ('bbmrt_m30_gbpusd','GBP_USD'),
    ('bbmrt_m30_eurcad','EUR_CAD'), ('bbmrt_m30_eurjpy','EUR_JPY'),
    ('bbmrt_m30_chfjpy','CHF_JPY'), ('bbmrt_m30_audchf','AUD_CHF'),
    ('bbmrt_m30_eursgd','EUR_SGD'), ('bbmrt_m30_gbpaud','GBP_AUD'),
    ('bbmrt_m30_cadjpy','CAD_JPY'), ('bbmrt_m30_audsgd','AUD_SGD'),
    ('bbmrt_m30_euraud','EUR_AUD'), ('bbmrt_m30_gbpcad','GBP_CAD'),
    ('bbmrt_m30_gbpsgd','GBP_SGD'), ('bbmrt_m30_gbpjpy','GBP_JPY'),
    ('bbmrt_m30_gbpchf','GBP_CHF'), ('bbmrt_m30_audjpy','AUD_JPY'),
    ('bbmrt_m30_nzdjpy','NZD_JPY'),
]

a2_rows = []
for tag, instr in M30_PAIRS:
    try:
        df = load_oanda_data(instr, period='10y', interval='30m')
        r  = _run(df, BBMRT_M30, tag)
        if r: a2_rows.append(r); print(_row(tag, *r, be_wr=50))
    except Exception as e:
        print(f'  ERROR {tag}: {e}')

print(SEP)
if a2_rows:
    avg_wr = sum(r[2] for r in a2_rows) / len(a2_rows)
    avg_sh = sum(r[0] for r in a2_rows) / len(a2_rows)
    total_n = sum(r[3] for r in a2_rows)
    wins = sum(int(r[3]*r[2]/100) for r in a2_rows)
    agg_wr = wins/total_n*100 if total_n else 0
    print(f'Agent 2 summary: {len(a2_rows)} strategies | avg Sh {avg_sh:+.2f} | '
          f'avg WR {avg_wr:.1f}% | agg WR {agg_wr:.1f}% ({total_n} trades)')

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — com.bamznizzy.forex-bot.h1  (h1_runner.py, every hour)
# Active strategy: PDHL NATGAS (H1 bars)
# H1 EMA and MACD sleeves are empty (removed Jul 2026)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('AGENT 3: com.bamznizzy.forex-bot.h1  (h1_runner.py, every hour)')
print('PDHL NATGAS (H1 entry)')
print(SEP)
print(HDR)
print(SEP)

class PDHL_H1(Strategy):
    risk_pct = RISK_PCT
    def init(self):
        self.atr_i = self.I(atr, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        if self.position or len(self.data) < 30: return
        # Approximate PDHL H1: use previous 24-bar high/low as the range
        window = 24
        prev_high = max(float(self.data.High[-i]) for i in range(2, window+2))
        prev_low  = min(float(self.data.Low[-i])  for i in range(2, window+2))
        prev_mid  = (prev_high + prev_low) / 2
        prev_rng  = prev_high - prev_low
        if prev_rng <= 0: return
        p = float(self.data.Close[-1])
        sl_dist = abs(p - prev_mid)
        if sl_dist <= 0: return
        ra = self.equity * self.risk_pct
        units = max(1, int(ra / sl_dist))
        if p > prev_high:
            self.buy(size=units, sl=prev_mid, tp=p + 1.5*prev_rng)
        elif p < prev_low:
            self.sell(size=units, sl=prev_mid, tp=p - 1.5*prev_rng)

a3_rows = []
df_ng_h1 = load_oanda_data('NATGAS_USD', period='10y', interval='1h')
r = _run(df_ng_h1, PDHL_H1, 'pdhl_natgas (H1)')
if r: a3_rows.append(r); print(_row('pdhl_natgas (H1)', *r, be_wr=40))

print(SEP)
if a3_rows:
    avg_wr = sum(r[2] for r in a3_rows) / len(a3_rows)
    avg_sh = sum(r[0] for r in a3_rows) / len(a3_rows)
    total_n = sum(r[3] for r in a3_rows)
    print(f'Agent 3 summary: {len(a3_rows)} strategies | avg Sh {avg_sh:+.2f} | '
          f'avg WR {avg_wr:.1f}% ({total_n} trades)')

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4 — com.bamznizzy.forex-bot.fvg  (fvg_session_runner.py, session opens)
# Strategy: FVG M30 — NATGAS_USD, GBP_CHF
# ═══════════════════════════════════════════════════════════════════════════════
print()
print('AGENT 4: com.bamznizzy.forex-bot.fvg  (fvg_session_runner.py, session opens)')
print('FVG M30 — NATGAS USD, GBP/CHF')
print(SEP)
print(HDR)
print(SEP)

class FVG_M30(Strategy):
    ema_period  = 200
    rr          = 3.0
    fvg_expiry  = 8
    risk_pct    = RISK_PCT
    sl_buffer   = 0.0005
    min_gap_pct = 0.0015
    def init(self):
        self.trend    = self.I(_ema, self.data.Close, self.ema_period)
        self._dir     = None
        self._top     = np.nan
        self._bot     = np.nan
        self._fvg_idx = -999
    def next(self):
        idx = len(self.data) - 1
        if idx < 3: return
        t = self.data.index[-1]
        if not (7 <= t.hour < 9 or 13 <= t.hour < 15): return
        p  = float(self.data.Close[-1])
        tr = float(self.trend[-1])
        if np.isnan(tr): return
        h1 = float(self.data.High[-3]); l1 = float(self.data.Low[-3])
        h3 = float(self.data.High[-1]); l3 = float(self.data.Low[-1])
        min_gap = abs(p) * self.min_gap_pct
        if not self.position:
            if h1 < l3 and (l3 - h1) >= min_gap and p > tr:
                self._dir='bull'; self._bot=h1; self._top=l3; self._fvg_idx=idx
            elif l1 > h3 and (l1 - h3) >= min_gap and p < tr:
                self._dir='bear'; self._bot=h3; self._top=l1; self._fvg_idx=idx
        if self._dir and (idx - self._fvg_idx) > self.fvg_expiry:
            self._dir = None
        if not self._dir or self.position: return
        ra = self.equity * self.risk_pct
        if self._dir == 'bull' and float(self.data.Low[-1]) <= self._top:
            sl = self._bot * (1 - self.sl_buffer)
            sl_dist = max(p - sl, 1e-9)
            units = max(1, min(int(ra / sl_dist), 50_000))
            tp = p + self.rr * sl_dist
            if sl < p < tp: self.buy(size=units, sl=sl, tp=tp)
            self._dir = None
        elif self._dir == 'bear' and float(self.data.High[-1]) >= self._bot:
            sl = self._top * (1 + self.sl_buffer)
            sl_dist = max(sl - p, 1e-9)
            units = max(1, min(int(ra / sl_dist), 50_000))
            tp = p - self.rr * sl_dist
            if tp < p < sl: self.sell(size=units, sl=sl, tp=tp)
            self._dir = None

a4_rows = []
FVG_PAIRS = [('fvg_m30_natgas', 'NATGAS_USD'), ('fvg_m30_gbpchf', 'GBP_CHF')]
for tag, instr in FVG_PAIRS:
    try:
        df = load_oanda_data(instr, period='10y', interval='30m')
        r  = _run(df, FVG_M30, tag)
        if r: a4_rows.append(r); print(_row(tag, *r, be_wr=25))
    except Exception as e:
        print(f'  ERROR {tag}: {e}')

print(SEP)
if a4_rows:
    avg_wr = sum(r[2] for r in a4_rows) / len(a4_rows)
    avg_sh = sum(r[0] for r in a4_rows) / len(a4_rows)
    total_n = sum(r[3] for r in a4_rows)
    wins = sum(int(r[3]*r[2]/100) for r in a4_rows)
    agg_wr = wins/total_n*100 if total_n else 0
    print(f'Agent 4 summary: {len(a4_rows)} strategies | avg Sh {avg_sh:+.2f} | '
          f'avg WR {avg_wr:.1f}% | agg WR {agg_wr:.1f}% ({total_n} trades)')

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
all_rows = a1_rows + a2_rows + a3_rows + a4_rows
print()
print('═' * 80)
print('PORTFOLIO SUMMARY — all 27 sleeves')
print('═' * 80)
if all_rows:
    total_n = sum(r[3] for r in all_rows)
    wins    = sum(int(r[3]*r[2]/100) for r in all_rows)
    agg_wr  = wins/total_n*100 if total_n else 0
    avg_sh  = sum(r[0] for r in all_rows) / len(all_rows)
    pos_ann = sum(1 for r in all_rows if r[1] > 0)
    print(f'  Total sleeves tested : {len(all_rows)}')
    print(f'  Portfolio avg Sharpe : {avg_sh:+.2f}')
    print(f'  Aggregate WR         : {agg_wr:.1f}%  ({wins}/{total_n} trades)')
    print(f'  Positive Ann%        : {pos_ann}/{len(all_rows)} sleeves')
    print()
    print('  By agent:')
    for label, rows in [
        ('Agent 1 (D1/PDHL/CONSEC)', a1_rows),
        ('Agent 2 (BBMRT M30)',       a2_rows),
        ('Agent 3 (PDHL H1)',         a3_rows),
        ('Agent 4 (FVG M30)',         a4_rows),
    ]:
        if rows:
            wr  = sum(r[2] for r in rows) / len(rows)
            sh  = sum(r[0] for r in rows) / len(rows)
            n   = sum(r[3] for r in rows)
            w   = sum(int(r[3]*r[2]/100) for r in rows)
            agg = w/n*100 if n else 0
            print(f'    {label:<30} avg WR {wr:.1f}%  agg WR {agg:.1f}%  avg Sh {sh:+.2f}')
