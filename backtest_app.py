"""Backtest Dashboard — all active portfolio sleeves."""
from dotenv import load_dotenv; load_dotenv()
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from backtesting import Backtest, Strategy
from data import load_oanda_data

st.set_page_config(page_title="Forex Bot — Backtest", layout="wide", page_icon="📈")

# ── Strategy helpers ──────────────────────────────────────────────────────────
def _rsi(close, n=14):
    c = pd.Series(np.array(close, float))
    d = c.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return (100 - 100 / (1 + g / l.replace(0, np.nan))).values

def _atr(high, low, close, n=14):
    h = pd.Series(np.array(high, float))
    l = pd.Series(np.array(low, float))
    c = pd.Series(np.array(close, float))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean().values

def _ema(close, n):
    return pd.Series(np.array(close, float)).ewm(span=n, adjust=False).mean().values

RISK_PCT = 0.0025
COMM     = 0.0002
MARGIN   = 1/30

# ── Strategy classes ──────────────────────────────────────────────────────────
class BBMRT(Strategy):
    n=20; std_mult=2.0; trend_ema=200; rr=2.0
    def init(self):
        c = self.data.Close
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.n).mean().values, c)
        self.std = self.I(lambda x: pd.Series(x).rolling(self.n).std().values, c)
        self.trend = self.I(_ema, c, self.trend_ema)
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, c, 14)
    def next(self):
        if len(self.data) < self.trend_ema + 5: return
        p = float(self.data.Close[-1])
        upper = float(self.sma[-1]) + self.std_mult * float(self.std[-1])
        lower = float(self.sma[-1]) - self.std_mult * float(self.std[-1])
        trend = float(self.trend[-1])
        av = float(self.atr_i[-1])
        if np.isnan(upper) or np.isnan(trend) or np.isnan(av) or av <= 0: return
        ra = self.equity * RISK_PCT
        if self.position: return
        if p < lower and p < trend:
            sd = 1.5 * av; u = max(1, int(ra / sd)); sl = p - sd; tp = p + self.rr * sd
            if sl < p < tp: self.buy(size=u, sl=sl, tp=tp)
        elif p > upper and p > trend:
            sd = 1.5 * av; u = max(1, int(ra / sd)); sl = p + sd; tp = p - self.rr * sd
            if tp < p < sl: self.sell(size=u, sl=sl, tp=tp)

class EMACross(Strategy):
    fast=9; slow=21; atr_n=14
    def init(self):
        self.fast_i = self.I(_ema, self.data.Close, self.fast)
        self.slow_i = self.I(_ema, self.data.Close, self.slow)
        self.atr_i  = self.I(_atr, self.data.High, self.data.Low, self.data.Close, self.atr_n)
    def next(self):
        if len(self.data) < self.slow + 5: return
        f, s = float(self.fast_i[-1]), float(self.slow_i[-1])
        f1, s1 = float(self.fast_i[-2]), float(self.slow_i[-2])
        av = float(self.atr_i[-1])
        if np.isnan(f) or np.isnan(s) or np.isnan(av) or av <= 0: return
        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT
        if self.position: return
        if f1 < s1 and f > s:
            sd = 1.5*av; u = max(1,int(ra/sd)); sl=p-sd; tp=p+2*sd
            if sl<p<tp: self.buy(size=u,sl=sl,tp=tp)
        elif f1 > s1 and f < s:
            sd = 1.5*av; u = max(1,int(ra/sd)); sl=p+sd; tp=p-2*sd
            if tp<p<sl: self.sell(size=u,sl=sl,tp=tp)

class PDHL(Strategy):
    tp_mult=1.5
    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
        self._dl = self._ds = None
    def next(self):
        if len(self.data) < 3: return
        ph = float(self.data.High[-2]); pl = float(self.data.Low[-2])
        pm = (ph+pl)/2; pr = ph-pl
        if pr <= 0: return
        p  = float(self.data.Close[-1])
        d  = self.data.index[-1]
        ra = self.equity * RISK_PCT
        if self.position: return
        if p > ph and self._dl != d:
            sd = abs(p-pm); u = max(1,int(ra/sd))
            tp = p+self.tp_mult*pr
            if pm<p<tp: self.buy(size=u,sl=pm,tp=tp); self._dl=d
        elif p < pl and self._ds != d:
            sd = abs(p-pm); u = max(1,int(ra/sd))
            tp = p-self.tp_mult*pr
            if tp<p<pm: self.sell(size=u,sl=pm,tp=tp); self._ds=d

class ConsecD1(Strategy):
    streak=3; rr=2.0
    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        if len(self.data) < self.streak+2: return
        closes = [float(self.data.Close[-(i+1)]) for i in range(self.streak+1)]
        av = float(self.atr_i[-1])
        if np.isnan(av) or av<=0: return
        p  = float(self.data.Close[-1])
        ra = self.equity * RISK_PCT
        if self.position: return
        down = all(closes[i]<closes[i+1] for i in range(self.streak))
        up   = all(closes[i]>closes[i+1] for i in range(self.streak))
        sd = 1.5*av; u = max(1,int(ra/sd))
        if down:
            sl=p-sd; tp=p+self.rr*sd
            if sl<p<tp: self.buy(size=u,sl=sl,tp=tp)
        elif up:
            sl=p+sd; tp=p-self.rr*sd
            if tp<p<sl: self.sell(size=u,sl=sl,tp=tp)
class ConsecD1_S4(ConsecD1): streak=4

class RSIExtreme(Strategy):
    oversold=25; overbought=75; tp_mult=0.8; sl_mult=1.5
    def init(self):
        self.rsi_i = self.I(_rsi, self.data.Close, 14)
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        if len(self.data)<20: return
        r=float(self.rsi_i[-1]); av=float(self.atr_i[-1])
        if np.isnan(r) or np.isnan(av) or av<=0: return
        p=float(self.data.Close[-1]); ra=self.equity*RISK_PCT
        if self.position: return
        sd=self.sl_mult*av; u=max(1,int(ra/sd))
        if r<self.oversold:
            sl=p-sd; tp=p+self.tp_mult*p*0.01
            if sl<p: self.buy(size=u,sl=sl,tp=tp)
        elif r>self.overbought:
            sl=p+sd; tp=p-self.tp_mult*p*0.01
            if tp<p: self.sell(size=u,sl=sl,tp=tp)
class RSIExt_Wheat(RSIExtreme): oversold=15; overbought=85; tp_mult=1.0
class RSIExt_JP225(RSIExtreme): oversold=20; overbought=80; tp_mult=0.8
class RSIExt_UK100(RSIExtreme): oversold=20; overbought=80; tp_mult=0.8
class RSIExt_USDCAD(RSIExtreme): oversold=15; overbought=85; tp_mult=0.8

class RSIDiv(Strategy):
    pivot_n=5; rr=3.0; extreme_flt=False
    def init(self):
        self.rsi_i = self.I(_rsi, self.data.Close, 14)
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
    def _pivots(self, arr, n):
        lows=[]; highs=[]
        for i in range(n, len(arr)-n):
            if all(arr[i]<=arr[i-j] and arr[i]<=arr[i+j] for j in range(1,n+1)):
                lows.append((i,arr[i]))
            if all(arr[i]>=arr[i-j] and arr[i]>=arr[i+j] for j in range(1,n+1)):
                highs.append((i,arr[i]))
        return lows, highs
    def next(self):
        if len(self.data)<40: return
        av=float(self.atr_i[-1])
        if np.isnan(av) or av<=0: return
        c=np.array(self.data.Close); r=self.rsi_i
        lows,highs=self._pivots(c,self.pivot_n)
        rlows,rhighs=self._pivots(np.array(r),self.pivot_n)
        p=float(c[-1]); ra=self.equity*RISK_PCT
        if self.position: return
        bull=False; bear=False
        if len(lows)>=2 and len(rlows)>=2:
            if lows[-1][1]<lows[-2][1] and rlows[-1][1]>rlows[-2][1]:
                bull=True
        if len(highs)>=2 and len(rhighs)>=2:
            if highs[-1][1]>highs[-2][1] and rhighs[-1][1]<rhighs[-2][1]:
                bear=True
        sd=1.5*av; u=max(1,int(ra/sd))
        if bull:
            sl=p-sd; tp=p+self.rr*sd
            if sl<p<tp: self.buy(size=u,sl=sl,tp=tp)
        elif bear:
            sl=p+sd; tp=p-self.rr*sd
            if tp<p<sl: self.sell(size=u,sl=sl,tp=tp)
class RSIDiv_AUDJPY(RSIDiv): pivot_n=5; rr=3.0; extreme_flt=False
class RSIDiv_NZDUSD(RSIDiv): pivot_n=5; rr=3.0; extreme_flt=True
class RSIDiv_FR40(RSIDiv):   pivot_n=5; rr=2.0; extreme_flt=False

class EngulfD1(Strategy):
    rr=3.0; sl_atr=1.5
    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        if len(self.data)<4: return
        o1=float(self.data.Open[-2]); c1=float(self.data.Close[-2])
        o2=float(self.data.Open[-3]); c2=float(self.data.Close[-3])
        av=float(self.atr_i[-1])
        if np.isnan(av) or av<=0: return
        b1=abs(c1-o1); b2=abs(c2-o2)
        if b2<=0 or b1<b2*1.1: return
        p=float(self.data.Close[-1]); ra=self.equity*RISK_PCT
        if self.position: return
        sd=self.sl_atr*av; u=max(1,int(ra/sd))
        if c2<o2 and c1>o1 and o1<=c2 and c1>=o2:
            sl=p-sd; tp=p+self.rr*sd
            if sl<p<tp: self.buy(size=u,sl=sl,tp=tp)
        elif c2>o2 and c1<o1 and o1>=c2 and c1<=o2:
            sl=p+sd; tp=p-self.rr*sd
            if tp<p<sl: self.sell(size=u,sl=sl,tp=tp)

class DonchianD1(Strategy):
    period=20; rr=3.0; sl_atr=1.5
    def init(self):
        self.atr_i = self.I(_atr, self.data.High, self.data.Low, self.data.Close, 14)
    def next(self):
        if len(self.data)<self.period+3: return
        av=float(self.atr_i[-1])
        if np.isnan(av) or av<=0: return
        highs=np.array(self.data.High[-(self.period+2):-1])
        lows =np.array(self.data.Low[-(self.period+2):-1])
        ch=highs.max(); cl=lows.min()
        p=float(self.data.Close[-1]); ra=self.equity*RISK_PCT
        if self.position: return
        sd=self.sl_atr*av; u=max(1,int(ra/sd))
        if p>ch:
            sl=p-sd; tp=p+self.rr*sd
            if sl<p<tp: self.buy(size=u,sl=sl,tp=tp)
        elif p<cl:
            sl=p+sd; tp=p-self.rr*sd
            if tp<p<sl: self.sell(size=u,sl=sl,tp=tp)

class VolSpikeFade(Strategy):
    vol_mult=2.0; close_pct=0.25; rr=3.0; sl_atr=1.5
    def init(self):
        self.atr_i=self.I(_atr,self.data.High,self.data.Low,self.data.Close,14)
        self._wl=self._ws=False
    def next(self):
        if len(self.data)<3: return
        av=float(self.atr_i[-1])
        if np.isnan(av) or av<=0: return
        h1=float(self.data.High[-2]); l1=float(self.data.Low[-2]); c1=float(self.data.Close[-2])
        rng=h1-l1
        if rng<self.vol_mult*av: self._wl=self._ws=False
        else:
            pos=(c1-l1)/rng
            if pos<=self.close_pct: self._wl,self._ws=True,False
            elif pos>=1-self.close_pct: self._ws,self._wl=True,False
            else: self._wl=self._ws=False
        if self.position: self._wl=self._ws=False; return
        p=float(self.data.Close[-1]); ra=self.equity*RISK_PCT
        if self._wl:
            sl=p-self.sl_atr*av; sd=max(p-sl,1e-9)
            u=max(1,min(int(ra/sd),100_000)); tp=p+self.rr*sd
            if sl<p<tp: self.buy(size=u,sl=sl,tp=tp)
            self._wl=False
        elif self._ws:
            sl=p+self.sl_atr*av; sd=max(sl-p,1e-9)
            u=max(1,min(int(ra/sd),100_000)); tp=p-self.rr*sd
            if tp<p<sl: self.sell(size=u,sl=sl,tp=tp)
            self._ws=False

# ── Sleeve registry ───────────────────────────────────────────────────────────
SLEEVES = [
    ("BBMRT EUR/USD",    "EUR_USD",   BBMRT,         "10y","1d"),
    ("BBMRT GBP/USD",    "GBP_USD",   BBMRT,         "10y","1d"),
    ("EMA GBP/USD",      "GBP_USD",   EMACross,      "10y","1d"),
    ("EMA USD/JPY",      "USD_JPY",   EMACross,      "10y","1d"),
    ("EMA AUD/USD",      "AUD_USD",   EMACross,      "10y","1d"),
    ("PDHL NATGAS",      "NATGAS_USD",PDHL,           "10y","1d"),
    ("Consec Wheat S3",  "WHEAT_USD", ConsecD1,      "10y","1d"),
    ("Consec Wheat S4",  "WHEAT_USD", ConsecD1_S4,   "10y","1d"),
    ("RSI Wheat",        "WHEAT_USD", RSIExt_Wheat,  "10y","1d"),
    ("RSI JP225",        "JP225_USD", RSIExt_JP225,  "10y","1d"),
    ("RSI UK100",        "UK100_GBP", RSIExt_UK100,  "10y","1d"),
    ("RSI USD/CAD",      "USD_CAD",   RSIExt_USDCAD, "10y","1d"),
    ("RSI Div AUD/JPY",  "AUD_JPY",   RSIDiv_AUDJPY, "10y","1d"),
    ("RSI Div NZD/USD",  "NZD_USD",   RSIDiv_NZDUSD, "10y","1d"),
    ("RSI Div FR40",     "FR40_EUR",  RSIDiv_FR40,   "10y","1d"),
    ("Engulf SPX500",    "SPX500_USD",EngulfD1,       "10y","1d"),
    ("Donchian XAU/USD", "XAU_USD",   DonchianD1,    "10y","1d"),
    ("VSF UK100",        "UK100_GBP", VolSpikeFade,  "10y","1d"),
]

# ── Cached backtest runner ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_bt(label, instrument, strategy_name, period, interval):
    try:
        df = load_oanda_data(instrument, period=period, interval=interval)
    except Exception as e:
        return None, str(e)
    cls_map = {c.__name__: c for c in [
        BBMRT, EMACross, PDHL, ConsecD1, ConsecD1_S4,
        RSIExt_Wheat, RSIExt_JP225, RSIExt_UK100, RSIExt_USDCAD,
        RSIDiv_AUDJPY, RSIDiv_NZDUSD, RSIDiv_FR40,
        EngulfD1, DonchianD1, VolSpikeFade,
    ]}
    cls = cls_map.get(strategy_name)
    if cls is None:
        return None, f"Unknown strategy: {strategy_name}"
    try:
        bt = Backtest(df, cls, cash=100_000, commission=COMM, margin=MARGIN, finalize_trades=True)
        s  = bt.run()
        eq = s._equity_curve["Equity"]
        trades = s._trades.copy() if len(s._trades) else pd.DataFrame()
        stats = {
            "Sharpe":    round(float(s.get("Sharpe Ratio", 0) or 0), 2),
            "Ann %":     round(float(s.get("Return (Ann.) [%]", 0) or 0), 1),
            "Max DD %":  round(float(s.get("Max. Drawdown [%]", 0) or 0), 1),
            "Win Rate %":round(float(s.get("Win Rate [%]", 0) or 0), 1),
            "Trades":    int(s.get("# Trades", 0) or 0),
            "Prof Factor":round(float(s.get("Profit Factor", 0) or 0), 2),
        }
        return {"stats": stats, "equity": eq, "trades": trades}, None
    except Exception as e:
        return None, str(e)

def verdict(sh, ann):
    if sh >= 0.4 and ann > 0: return "✅ KEEP"
    if sh >= 0.1:              return "👀 WATCH"
    return "❌ REVIEW"

VERDICT_EXPLAIN = {
    "✅ KEEP":   ("green",  "This strategy has a strong, consistent track record. It's live."),
    "👀 WATCH":  ("orange", "Shows some promise but not consistent enough to fully trust yet."),
    "❌ REVIEW": ("red",    "This strategy lost money or was too inconsistent over 10 years."),
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Trading Bot — Strategy Results")
st.caption("10 years of historical data · All 18 strategies tested · Starting with $100,000")

page = st.sidebar.radio("📂 Section", ["🏠 All Strategies", "🔍 Strategy Detail"])

# ── Sidebar glossary ──────────────────────────────────────────────────────────
with st.sidebar.expander("❓ What do the numbers mean?"):
    st.markdown("""
**Quality Score**
How consistent and smooth the profits are.
- Above 0.4 = good ✅
- 0.1–0.4 = borderline 👀
- Below 0 = losing ❌

**Avg Yearly Profit %**
How much the strategy grew the account each year on average.

**Worst Loss Period**
The biggest drop from a peak before it recovered.
Smaller is better — e.g. -10% means the account fell 10% at its worst point.

**Trades Won %**
Out of every 100 trades, how many made money.

**Profit Factor**
For every $1 lost, how much did we make?
Above 1.0 = profitable overall.
""")

# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 All Strategies":

    st.subheader("Which strategies are working?")
    st.markdown("Click the button below to run all 18 strategies through 10 years of history. "
                "Green = strong, Orange = borderline, Red = not good enough to run live.")

    if st.button("▶  Run All 18 Strategies", type="primary"):
        st.session_state["run_all"] = True

    if st.session_state.get("run_all"):
        rows = []
        prog = st.progress(0)
        status = st.empty()
        for i, (lbl, instr, cls, period, interval) in enumerate(SLEEVES):
            status.text(f"Testing {lbl} ({i+1}/{len(SLEEVES)})…")
            result, err = run_bt(lbl, instr, cls.__name__, period, interval)
            if result:
                s = result["stats"]
                vrd = verdict(s["Sharpe"], s["Ann %"])
                rows.append({
                    "Status":           vrd,
                    "Strategy":         lbl,
                    "Avg Yearly Profit":f"{s['Ann %']:+.1f}%",
                    "Worst Loss Period":f"{s['Max DD %']:.1f}%",
                    "Trades Won %":     f"{s['Win Rate %']:.1f}%",
                    "Quality Score":    s["Sharpe"],
                    "Total Trades":     s["Trades"],
                    "_sh": s["Sharpe"], "_ann": s["Ann %"],
                })
            else:
                rows.append({
                    "Status": "⚠ Error", "Strategy": lbl,
                    "Avg Yearly Profit": "—", "Worst Loss Period": "—",
                    "Trades Won %": "—", "Quality Score": None,
                    "Total Trades": 0, "_sh": -99, "_ann": 0,
                })
            prog.progress((i + 1) / len(SLEEVES))
        status.empty(); prog.empty()
        st.session_state["summary_df"] = pd.DataFrame(rows)

    if "summary_df" in st.session_state:
        df_sum = st.session_state["summary_df"].copy()

        # Traffic light counts
        keeps  = (df_sum["Status"] == "✅ KEEP").sum()
        watches = (df_sum["Status"] == "👀 WATCH").sum()
        reviews = (df_sum["Status"] == "❌ REVIEW").sum()
        k, w, r = st.columns(3)
        k.metric("✅ Strong — running live", keeps)
        w.metric("👀 Borderline — monitoring", watches)
        r.metric("❌ Weak — not running", reviews)
        st.divider()

        # Sort: KEEP first, then WATCH, then REVIEW
        order = {"✅ KEEP": 0, "👀 WATCH": 1, "❌ REVIEW": 2}
        df_sum["_order"] = df_sum["Status"].map(order).fillna(3)
        df_sum = df_sum.sort_values(["_order", "_sh"], ascending=[True, False])

        # Display table — plain columns only
        display_cols = ["Status", "Strategy", "Avg Yearly Profit",
                        "Worst Loss Period", "Trades Won %", "Total Trades"]

        def colour_row(row):
            colour_map = {"✅ KEEP": "#14532d", "👀 WATCH": "#451a03", "❌ REVIEW": "#450a0a"}
            bg = colour_map.get(row["Status"], "")
            return [f"background-color:{bg};color:white" if bg else "" for _ in row]

        styled = df_sum[display_cols].style.apply(colour_row, axis=1)
        st.dataframe(styled, hide_index=True, width='stretch')
        st.divider()

        # Bar chart — Quality Score
        df_plot = df_sum[df_sum["Quality Score"].notna()].copy()
        colours = df_plot["_sh"].apply(
            lambda x: "#22c55e" if x >= 0.4 else ("#f59e0b" if x >= 0.1 else "#ef4444"))
        fig = go.Figure(go.Bar(
            x=df_plot["Strategy"],
            y=df_plot["_sh"],
            marker_color=colours.tolist(),
            text=df_plot["_sh"].apply(lambda x: f"{x:+.2f}"),
            textposition="outside",
        ))
        fig.add_hline(y=0.4, line_dash="dash", line_color="#22c55e",
                      annotation_text="Minimum to run live (0.4)",
                      annotation_position="top left")
        fig.add_hline(y=0, line_color="white", line_width=0.5)
        fig.update_layout(
            title="Quality Score — how consistent are the profits?  (higher = better)",
            xaxis_tickangle=-35, height=440,
            margin=dict(t=60, b=120),
            yaxis_title="Quality Score",
        )
        st.plotly_chart(fig, width='stretch')

        # Profit vs Risk scatter
        df_scatter = df_sum[df_sum["_sh"].notna()].copy()
        df_scatter["Ann %"]  = df_scatter["_ann"]
        df_scatter["DD %"]   = df_scatter["Worst Loss Period"].str.replace("%","").astype(float)
        df_scatter["Colour"] = df_scatter["Status"].map(
            {"✅ KEEP":"#22c55e","👀 WATCH":"#f59e0b","❌ REVIEW":"#ef4444"}).fillna("#888")
        fig2 = go.Figure()
        for vrd_label, colour in [("✅ KEEP","#22c55e"),("👀 WATCH","#f59e0b"),("❌ REVIEW","#ef4444")]:
            sub = df_scatter[df_scatter["Status"] == vrd_label]
            if sub.empty: continue
            fig2.add_trace(go.Scatter(
                x=sub["DD %"], y=sub["Ann %"],
                mode="markers+text", name=vrd_label,
                text=sub["Strategy"], textposition="top center",
                marker=dict(color=colour, size=14),
            ))
        fig2.update_layout(
            title="Profit vs Risk — you want to be top-left (high profit, low risk)",
            xaxis_title="Worst Loss Period %  ← lower is safer",
            yaxis_title="Avg Yearly Profit %  ↑ higher is better",
            height=450, margin=dict(t=60),
        )
        fig2.add_annotation(text="Best zone", x=df_scatter["DD %"].min()+1,
                            y=df_scatter["Ann %"].max(), showarrow=False,
                            font=dict(color="#22c55e", size=13))
        st.plotly_chart(fig2, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
else:  # Strategy Detail
    sleeve_names = [s[0] for s in SLEEVES]
    choice = st.sidebar.selectbox("Pick a strategy", sleeve_names)
    idx    = sleeve_names.index(choice)
    lbl, instr, cls, period, interval = SLEEVES[idx]

    st.subheader(f"🔍  {lbl}")
    st.caption(f"Instrument: {instr}  ·  10 years of daily data")

    with st.spinner("Running 10 years of history…"):
        result, err = run_bt(lbl, instr, cls.__name__, period, interval)

    if err:
        st.error(f"Could not run backtest: {err}")
    else:
        s      = result["stats"]
        eq     = result["equity"]
        trades = result["trades"]
        vrd    = verdict(s["Sharpe"], s["Ann %"])
        colour, explain = VERDICT_EXPLAIN[vrd][:2]

        # Big verdict banner
        st.markdown(f"## {vrd}")
        st.info(explain)
        st.divider()

        # Plain English KPI cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Avg Yearly Profit",
            f"{s['Ann %']:+.1f}%",
            help="How much the account grew each year on average over 10 years."
        )
        c2.metric(
            "Worst Loss Period",
            f"{s['Max DD %']:.1f}%",
            help="The biggest drop the account took from its highest point before recovering. Smaller is safer."
        )
        c3.metric(
            "Trades Won",
            f"{s['Win Rate %']:.1f}%",
            help="Out of every 100 trades, this many made money."
        )
        c4.metric(
            "Total Trades (10y)",
            s["Trades"],
            help="How many trades were placed over the full 10 years."
        )
        st.caption(f"Quality Score: **{s['Sharpe']:+.2f}**  (above 0.4 = good enough to run live)")
        st.divider()

        # Equity curve
        final = eq.values[-1]
        gain  = final - 100_000
        st.markdown(f"### 📈 Account Growth  —  started $100,000, ended **${final:,.0f}** ({gain:+,.0f})")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            mode="lines", name="Account value",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        ))
        fig_eq.add_hline(y=100_000, line_dash="dot", line_color="gray",
                         annotation_text="Started here ($100k)", annotation_position="top left")
        fig_eq.update_layout(
            height=360,
            yaxis_title="Account Value ($)",
            xaxis_title="",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_eq, width='stretch')

        # Year-by-year
        if not trades.empty:
            pnl_col = "PnLComm" if "PnLComm" in trades.columns else "PnL"
            trades["Year"] = pd.to_datetime(trades["ExitTime"]).dt.year
            yoy = trades.groupby("Year")[pnl_col].sum().reset_index()
            profitable_years = (yoy[pnl_col] > 0).sum()
            total_years      = len(yoy)

            st.markdown(f"### 📅 Year by Year  —  made money in **{profitable_years} out of {total_years} years**")
            yoy["Colour"] = yoy[pnl_col].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444")
            yoy["Label"]  = yoy[pnl_col].apply(lambda x: f"${x:+,.0f}")
            fig_yoy = go.Figure(go.Bar(
                x=yoy["Year"],
                y=yoy[pnl_col],
                marker_color=yoy["Colour"].tolist(),
                text=yoy["Label"],
                textposition="outside",
            ))
            fig_yoy.add_hline(y=0, line_color="white", line_width=0.8)
            fig_yoy.update_layout(
                height=340,
                xaxis=dict(dtick=1, title=""),
                yaxis_title="Profit / Loss ($)",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_yoy, width='stretch')

            # Walk-forward (plain English)
            st.divider()
            st.markdown("### 🧪 Did it work on data it had never seen before?")
            st.markdown(
                "We trained the strategy on the **first 7 years** of data, "
                "then tested it on the **last 3 years** it had never seen. "
                "If it still works on those unseen years, the edge is real — not just luck."
            )
            split_date = eq.index[int(len(eq) * 0.70)]
            is_t  = trades[pd.to_datetime(trades["ExitTime"]) <  split_date]
            oos_t = trades[pd.to_datetime(trades["ExitTime"]) >= split_date]
            wc1, wc2 = st.columns(2)
            for col, label, wf_t, emoji in [
                (wc1, "First 7 years (used to build strategy)", is_t,  "🔧"),
                (wc2, "Last 3 years (strategy had never seen)",  oos_t, "🎯"),
            ]:
                if len(wf_t) < 3:
                    col.info(f"{emoji} {label}: not enough trades")
                    continue
                pnl  = wf_t[pnl_col]
                sh   = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std()>0 else 0
                wr   = (pnl>0).mean()*100
                tot  = pnl.sum()
                good = sh >= 0.3 and tot > 0
                col.markdown(f"**{emoji} {label}**")
                col.metric("Quality Score", f"{sh:+.2f}")
                col.metric("Trades Won",    f"{wr:.0f}%")
                col.metric("Total Profit",  f"${tot:+,.0f}")
                col.success("✅ Passed" ) if good else col.error("❌ Failed")

            # Trade list
            st.divider()
            with st.expander("📋 See every individual trade"):
                show_cols = [c for c in
                    ["EntryTime","ExitTime","EntryPrice","ExitPrice", pnl_col]
                    if c in trades.columns]
                rename = {pnl_col: "Profit / Loss ($)", "EntryTime": "Opened",
                          "ExitTime": "Closed", "EntryPrice": "Entry Price",
                          "ExitPrice": "Exit Price"}
                disp = trades[show_cols].rename(columns=rename).sort_values(
                    "Closed", ascending=False)
                st.dataframe(disp, hide_index=True, width='stretch')
