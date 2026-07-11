"""
Full portfolio audit — every active sleeve in live_runner.py.
10y D1 backtest (D1 for daily strategies, M30 for FVG).
Reports Sharpe, Ann%, DD%, WR%, N, PF side-by-side with live journal P/L.
"""
from dotenv import load_dotenv; load_dotenv()
from backtesting import Backtest, Strategy
from data import load_oanda_data
import numpy as np, pandas as pd

RISK_PCT = 0.0025

# ── Indicator helpers ─────────────────────────────────────────────────────────
def _rsi(close, n=14):
    c = pd.Series(np.array(close, float))
    d = c.diff()
    u = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    v = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return (100 * u / (u + v)).values

def _atr(high, low, close, n=14):
    h = pd.Series(np.array(high, float))
    l = pd.Series(np.array(low, float))
    c = pd.Series(np.array(close, float))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values

def _sma(close, n):
    return pd.Series(np.array(close, float)).rolling(n).mean().values

def _ema(close, n):
    return pd.Series(np.array(close, float)).ewm(span=n, adjust=False).mean().values

def _rolling_std(close, n):
    return pd.Series(np.array(close, float)).rolling(n).std().values

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BBMRT — Bollinger Mean Reversion Trend Filter
# ═══════════════════════════════════════════════════════════════════════════════
class BBMRT(Strategy):
    bb_period = 20; bb_k = 2.0; trend_period = 200; atr_period = 14
    sl_atr_mult = 2.0; max_hold = 20

    def init(self):
        self.mid   = self.I(_sma,         self.data.Close, self.bb_period)
        self.sd    = self.I(_rolling_std,  self.data.Close, self.bb_period)
        self.trend = self.I(_ema,          self.data.Close, self.trend_period)
        self.av    = self.I(_atr,          self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self._entry = None; self._entry_bar = 0

    def next(self):
        if len(self.data) < self.trend_period + 5: return
        m, s, t, av = self.mid[-1], self.sd[-1], self.trend[-1], self.av[-1]
        p = float(self.data.Close[-1])
        upper, lower = m + self.bb_k * s, m - self.bb_k * s
        if np.isnan(m) or np.isnan(s) or av <= 0: return
        sd = self.sl_atr_mult * av
        ra = self.equity * RISK_PCT
        units = max(1, min(int(ra / sd), 100_000))
        if self.position:
            bars_held = len(self.data) - self._entry_bar
            is_long = self.position.size > 0
            if (is_long and p >= m) or (not is_long and p <= m):
                self.position.close()
            elif bars_held >= self.max_hold:
                self.position.close()
            return
        if p < lower and p > t:
            self.buy(size=units, sl=p - sd)
            self._entry_bar = len(self.data)
        elif p > upper and p < t:
            self.sell(size=units, sl=p + sd)
            self._entry_bar = len(self.data)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EMA Crossover D1
# ═══════════════════════════════════════════════════════════════════════════════
class EMACross(Strategy):
    fast = 10; slow = 30; atr_period = 14; sl_atr_mult = 2.0; tp_atr_mult = 4.0

    def init(self):
        self.ef = self.I(_ema, self.data.Close, self.fast)
        self.es = self.I(_ema, self.data.Close, self.slow)
        self.av = self.I(_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        if len(self.data) < self.slow + 5: return
        av = self.av[-1]
        if np.isnan(av) or av <= 0: return
        cross_up = self.ef[-2] <= self.es[-2] and self.ef[-1] > self.es[-1]
        cross_dn = self.ef[-2] >= self.es[-2] and self.ef[-1] < self.es[-1]
        p = float(self.data.Close[-1])
        sd = self.sl_atr_mult * av
        ra = self.equity * RISK_PCT
        units = max(1, min(int(ra / sd), 100_000))
        if self.position:
            is_long = self.position.size > 0
            if (is_long and cross_dn) or (not is_long and cross_up):
                self.position.close()
            return
        if cross_up:
            self.buy(size=units, sl=p - sd, tp=p + self.tp_atr_mult * av)
        elif cross_dn:
            self.sell(size=units, sl=p + sd, tp=p - self.tp_atr_mult * av)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PDHL — Previous Day High/Low breakout (D1 proxy: use previous bar H/L)
# ═══════════════════════════════════════════════════════════════════════════════
class PDHL(Strategy):
    sl_frac = 0.5; tp_mult = 1.5   # SL at midpoint, TP = 1.5× range

    def init(self):
        self.av = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        if len(self.data) < 5: return
        prev_h = float(self.data.High[-2])
        prev_l = float(self.data.Low[-2])
        prev_rng = prev_h - prev_l
        if prev_rng <= 0: return
        prev_mid = (prev_h + prev_l) / 2
        p = float(self.data.Close[-1])
        if self.position: return
        av = self.av[-1]
        if np.isnan(av) or av <= 0: return
        ra = self.equity * RISK_PCT
        if p > prev_h:
            sd = abs(p - prev_mid)
            if sd <= 0: return
            units = max(1, min(int(ra / sd), 100_000))
            self.buy(size=units, sl=prev_mid, tp=p + self.tp_mult * prev_rng)
        elif p < prev_l:
            sd = abs(prev_mid - p)
            if sd <= 0: return
            units = max(1, min(int(ra / sd), 100_000))
            self.sell(size=units, sl=prev_mid, tp=p - self.tp_mult * prev_rng)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Consecutive D1 mean reversion (WHEAT)
# ═══════════════════════════════════════════════════════════════════════════════
class ConsecD1(Strategy):
    streak = 3; sl_atr = 1.5; tp_atr = 1.0

    def init(self):
        self.av = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        n = self.streak
        if len(self.data) < n + 5: return
        av = self.av[-1]
        if np.isnan(av) or av <= 0: return
        closes = self.data.Close
        down = all(float(closes[-(i+1)]) < float(closes[-(i+2)]) for i in range(n))
        up   = all(float(closes[-(i+1)]) > float(closes[-(i+2)]) for i in range(n))
        p = float(self.data.Close[-1])
        if self.position: return
        ra = self.equity * RISK_PCT
        sd = self.sl_atr * av
        units = max(1, min(int(ra / sd), 100_000))
        if down:
            self.buy(size=units, sl=p - sd, tp=p + self.tp_atr * av)
        elif up:
            self.sell(size=units, sl=p + sd, tp=p - self.tp_atr * av)

class ConsecD1_S4(ConsecD1): streak = 4

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RSI Extreme Fade D1
# ═══════════════════════════════════════════════════════════════════════════════
class RSIExtreme(Strategy):
    oversold = 20; overbought = 80; sl_atr = 1.5; tp_mult = 0.8

    def init(self):
        self.rs = self.I(_rsi,  self.data.Close, 14)
        self.av = self.I(_atr,  self.data.High, self.data.Low, self.data.Close, 14)

    def next(self):
        if len(self.data) < 20: return
        rn, av = self.rs[-1], self.av[-1]
        if np.isnan(rn) or np.isnan(av) or av <= 0: return
        if self.position: return
        p  = float(self.data.Close[-1])
        sd = self.sl_atr * av
        ra = self.equity * RISK_PCT
        units = max(1, min(int(ra / sd), 100_000))
        if rn < self.oversold:
            sl = p - sd; tp = p + self.tp_mult * sd
            if sl < p < tp: self.buy(size=units, sl=sl, tp=tp)
        elif rn > self.overbought:
            sl = p + sd; tp = p - self.tp_mult * sd
            if tp < p < sl: self.sell(size=units, sl=sl, tp=tp)

class RSIExt_Wheat(RSIExtreme):  oversold=15; overbought=85; tp_mult=1.0
class RSIExt_JP225(RSIExtreme):  oversold=20; overbought=80; tp_mult=0.8
class RSIExt_UK100(RSIExtreme):  oversold=20; overbought=80; tp_mult=0.8
class RSIExt_USDCAD(RSIExtreme): oversold=15; overbought=85; tp_mult=0.8

# ═══════════════════════════════════════════════════════════════════════════════
# 6. RSI Divergence D1
# ═══════════════════════════════════════════════════════════════════════════════
class RSIDiv(Strategy):
    pivot_n = 5; rr = 3.0; extreme_flt = False; sl_atr = 1.5

    def init(self):
        self.rs = self.I(_rsi, self.data.Close, 14)
        self.av = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._s_lows = []; self._s_highs = []
        self._wl = False; self._ws = False

    def _is_pivot_low(self):
        n = self.pivot_n
        if len(self.data) < 2*n+2: return False
        c = -(n+1); cl = float(self.data.Low[c])
        return all(float(self.data.Low[c+j]) > cl for j in range(-n, n+1) if j != 0)

    def _is_pivot_high(self):
        n = self.pivot_n
        if len(self.data) < 2*n+2: return False
        c = -(n+1); ch = float(self.data.High[c])
        return all(float(self.data.High[c+j]) < ch for j in range(-n, n+1) if j != 0)

    def next(self):
        if len(self.data) < 30: return
        n = self.pivot_n; av = float(self.av[-1])
        if np.isnan(av) or av <= 0: return
        if self._is_pivot_low():
            pl_p = float(self.data.Low[-(n+1)]); pl_r = float(self.rs[-(n+1)])
            if not np.isnan(pl_r) and self._s_lows:
                pp, pr = self._s_lows[-1]
                if pl_p < pp and pl_r > pr:
                    ok = (not self.extreme_flt) or pl_r < 45
                    if ok: self._wl = True
            if not np.isnan(pl_r):
                self._s_lows.append((pl_p, pl_r))
                if len(self._s_lows) > 6: self._s_lows.pop(0)
        if self._is_pivot_high():
            ph_p = float(self.data.High[-(n+1)]); ph_r = float(self.rs[-(n+1)])
            if not np.isnan(ph_r) and self._s_highs:
                pp, pr = self._s_highs[-1]
                if ph_p > pp and ph_r < pr:
                    ok = (not self.extreme_flt) or ph_r > 55
                    if ok: self._ws = True
            if not np.isnan(ph_r):
                self._s_highs.append((ph_p, ph_r))
                if len(self._s_highs) > 6: self._s_highs.pop(0)
        if self.position:
            self._wl = self._ws = False; return
        p = float(self.data.Close[-1]); ra = self.equity * RISK_PCT
        if self._wl:
            sd = self.sl_atr * av; units = max(1, min(int(ra / sd), 100_000))
            sl = p - sd; tp = p + self.rr * sd
            if sl < p < tp: self.buy(size=units, sl=sl, tp=tp)
            self._wl = False
        elif self._ws:
            sd = self.sl_atr * av; units = max(1, min(int(ra / sd), 100_000))
            sl = p + sd; tp = p - self.rr * sd
            if tp < p < sl: self.sell(size=units, sl=sl, tp=tp)
            self._ws = False

class RSIDiv_AUDJPY(RSIDiv): pivot_n=5; rr=3.0; extreme_flt=False
class RSIDiv_NZDUSD(RSIDiv): pivot_n=5; rr=3.0; extreme_flt=True
class RSIDiv_FR40(RSIDiv):   pivot_n=5; rr=2.0; extreme_flt=False

# ═══════════════════════════════════════════════════════════════════════════════
# Run all active sleeves
# ═══════════════════════════════════════════════════════════════════════════════
SLEEVES = [
    # (label, instrument, Strategy, period, interval, live_pl_so_far)
    # -- BBMRT --
    ("bbmrt_eurusd",           "EUR_USD",  BBMRT,          "10y", "1d",  None),
    ("bbmrt_gbpusd",           "GBP_USD",  BBMRT,          "10y", "1d",  None),
    # -- EMA D1 --
    ("ema_gbpusd",             "GBP_USD",  EMACross,       "10y", "1d",  None),
    ("ema_usdjpy",             "USD_JPY",  EMACross,       "10y", "1d",  None),
    ("ema_audusd",             "AUD_USD",  EMACross,       "10y", "1d",  None),
    # -- PDHL NATGAS --
    ("pdhl_natgas",            "NATGAS_USD", PDHL,         "10y", "1d", -9622.15),
    # -- CONSEC D1 --
    ("consec_d1_wheat_s3",     "WHEAT_USD", ConsecD1,      "10y", "1d",  None),
    ("consec_d1_wheat_s4",     "WHEAT_USD", ConsecD1_S4,   "10y", "1d",  None),
    # -- RSI Extreme --
    ("rsi_extreme_wheat",      "WHEAT_USD", RSIExt_Wheat,  "10y", "1d",  None),
    ("rsi_extreme_jp225",      "JP225_USD", RSIExt_JP225,  "10y", "1d",  None),
    ("rsi_extreme_uk100",      "UK100_GBP", RSIExt_UK100,  "10y", "1d",  None),
    ("rsi_extreme_usdcad",     "USD_CAD",   RSIExt_USDCAD, "10y", "1d",  None),
    # -- RSI Divergence --
    ("rsi_div_audjpy",         "AUD_JPY",  RSIDiv_AUDJPY,  "10y", "1d",  None),
    ("rsi_div_nzdusd",         "NZD_USD",  RSIDiv_NZDUSD,  "10y", "1d",  None),
    ("rsi_div_fr40",           "FR40_EUR", RSIDiv_FR40,    "10y", "1d",  None),
]

print("FULL PORTFOLIO BACKTEST — all 15 active D1 sleeves + FVG (noted separately)")
print("Period: 10y D1   Risk: 0.25% per sleeve   Commission: 0.02%")
print()
print(f"{'Sleeve':<26} {'Instr':<10} {'Sh':>6} {'Ann%':>7} {'DD%':>6} {'WR%':>6} "
      f"{'N':>5} {'PF':>5}  {'Live P/L':>10}  Verdict")
print("─" * 95)

results = []
for lbl, instr, cls, period, interval, live_pl in SLEEVES:
    try:
        df = load_oanda_data(instr, period=period, interval=interval)
    except Exception as e:
        print(f"{lbl:<26} load error: {e}")
        continue
    try:
        b  = Backtest(df, cls, cash=100_000, commission=0.0002,
                      margin=1/30, finalize_trades=True)
        s  = b.run()
        sh  = float(s.get('Sharpe Ratio', 0) or 0)
        ann = float(s.get('Return (Ann.) [%]', 0) or 0)
        dd  = float(s.get('Max. Drawdown [%]', 0) or 0)
        wr  = float(s.get('Win Rate [%]', 0) or 0)
        n   = int(s.get('# Trades', 0) or 0)
        pf  = float(s.get('Profit Factor', 0) or 0)
        ok  = sh >= 0.4 and ann > 0 and n >= 10
        warn = sh < 0 or ann < 0
        verdict = 'KEEP' if ok else ('WATCH' if sh >= 0.1 else 'REVIEW')
        lp_str = f"{live_pl:>+10.2f}" if live_pl is not None else f"{'no trades':>10}"
        flag = "⚠" if (live_pl is not None and live_pl < -500) else ""
        print(f"{lbl:<26} {instr:<10} {sh:>+6.2f} {ann:>+6.1f}% {dd:>5.1f}% "
              f"{wr:>5.1f}% {n:>5} {pf:>5.2f}  {lp_str}  {verdict} {flag}")
        results.append(dict(lbl=lbl, instr=instr, sh=sh, ann=ann, wr=wr, n=n,
                            verdict=verdict, live_pl=live_pl))
    except Exception as e:
        print(f"{lbl:<26} {instr:<10} error: {e}")

print("─" * 95)
keeps   = [r for r in results if r['verdict'] == 'KEEP']
watches = [r for r in results if r['verdict'] == 'WATCH']
reviews = [r for r in results if r['verdict'] == 'REVIEW']
print(f"KEEP   (Sh ≥ 0.4, Ann > 0): {len(keeps)}")
print(f"WATCH  (Sh ≥ 0.1):          {len(watches)}")
print(f"REVIEW (Sh < 0.1):          {len(reviews)}")
if reviews:
    print("\n** SLEEVES TO REVIEW (weak or negative backtest): **")
    for r in sorted(reviews, key=lambda x: x['sh']):
        lp = f"  live={r['live_pl']:+.0f}" if r['live_pl'] is not None else ""
        print(f"   {r['lbl']:<26}  Sh {r['sh']:+.2f}  WR {r['wr']:.1f}%  Ann {r['ann']:+.1f}%{lp}")

print("\nNote: FVG M30 sleeves (NATGAS, GBP/CHF) omitted — require M30 data,")
print("      previously backtested: NATGAS Sh +0.38, GBP/CHF Sh +0.38")
