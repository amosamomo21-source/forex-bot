"""Live trade progression dashboard.

Run with: uv run streamlit run dashboard.py --server.headless true --server.port 8501
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv

load_dotenv()

import broker  # noqa: E402
import journal  # noqa: E402

st.set_page_config(page_title="Forex Bot", layout="wide", page_icon="📈")

# Auto-refresh every 15 seconds
st_autorefresh(interval=15_000, key="live_refresh")

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .digital-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    font-family: 'Courier New', monospace;
  }
  .digital-label {
    color: #8b949e;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .digital-value {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: 2px;
  }
  .digital-sub {
    font-size: 13px;
    margin-top: 4px;
    letter-spacing: 1px;
  }
  .green  { color: #3fb950; }
  .red    { color: #f85149; }
  .white  { color: #e6edf3; }
  .yellow { color: #d29922; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 Forex Bot — Live Dashboard")
st.caption("Auto-refreshes every 15 seconds · OANDA practice account")

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_broker() -> broker.OandaBroker:
    return broker.from_env()

@st.cache_data(ttl=15)
def fetch_account():
    return get_broker().account_summary()["account"]

@st.cache_data(ttl=15)
def fetch_open_trades():
    return get_broker().get_open_trades()

@st.cache_data(ttl=15)
def fetch_transactions():
    return get_broker().get_transaction_history()

@st.cache_data(ttl=60)
def fetch_journal():
    if not journal.JOURNAL_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(journal.JOURNAL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data(ttl=15)
def fetch_candles(instrument: str, granularity: str, count: int = 120):
    raw = get_broker().get_candles(instrument, granularity=granularity, count=count)
    rows = []
    for c in raw:
        if not c.get("complete"):
            continue
        mid = c["mid"]
        rows.append({
            "time":  pd.to_datetime(c["time"]),
            "Open":  float(mid["o"]),
            "High":  float(mid["h"]),
            "Low":   float(mid["l"]),
            "Close": float(mid["c"]),
        })
    return pd.DataFrame(rows)

# ── Account metrics (digital cards) ──────────────────────────────────────────
try:
    acct       = fetch_account()
    balance    = float(acct["balance"])
    unreal_pl  = float(acct["unrealizedPL"])
    nav        = float(acct.get("NAV", balance + unreal_pl))
    margin_used = float(acct.get("marginUsed", 0))
    open_trades = fetch_open_trades()
except Exception as e:
    st.error(f"Could not reach OANDA: {e}")
    st.stop()

pl_color   = "green" if unreal_pl >= 0 else "red"
pl_sign    = "+" if unreal_pl >= 0 else ""
nav_color  = "green" if nav >= 100_000 else "red"
nav_delta  = nav - 100_000
nd_color   = "green" if nav_delta >= 0 else "red"
nd_sign    = "+" if nav_delta >= 0 else ""

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""
<div class="digital-card">
  <div class="digital-label">Balance</div>
  <div class="digital-value white">${balance:,.2f}</div>
</div>""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="digital-card">
  <div class="digital-label">Unrealized P/L</div>
  <div class="digital-value {pl_color}">{pl_sign}${unreal_pl:,.2f}</div>
  <div class="digital-sub {pl_color}">{pl_sign}{unreal_pl/balance*100:.2f}%</div>
</div>""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="digital-card">
  <div class="digital-label">NAV vs Start</div>
  <div class="digital-value {nav_color}">${nav:,.2f}</div>
  <div class="digital-sub {nd_color}">{nd_sign}${nav_delta:,.2f}</div>
</div>""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="digital-card">
  <div class="digital-label">Open Trades</div>
  <div class="digital-value yellow">{len(open_trades)}</div>
  <div class="digital-sub white">Margin: ${margin_used:,.0f}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── P&L Summary row ───────────────────────────────────────────────────────────
df_j_top = fetch_journal()
realized_pl = 0.0
n_closed_top = 0
wr_top = 0.0
if not df_j_top.empty:
    closes_top = df_j_top[df_j_top["event"] == "close"].dropna(subset=["realized_pl"])
    closes_top = closes_top.copy()
    closes_top["realized_pl"] = pd.to_numeric(closes_top["realized_pl"], errors="coerce")
    realized_pl   = closes_top["realized_pl"].sum()
    n_closed_top  = len(closes_top)
    wr_top        = (closes_top["realized_pl"] > 0).mean() * 100 if n_closed_top else 0.0

total_pl    = realized_pl + unreal_pl
rpl_color   = "green" if realized_pl >= 0 else "red"
tpl_color   = "green" if total_pl    >= 0 else "red"
wr_color_t  = "green" if wr_top >= 50 else "red"

st.markdown("### 💰 Portfolio P&L")
p1, p2, p3, p4 = st.columns(4)

p1.markdown(f"""
<div class="digital-card">
  <div class="digital-label">Realized P/L</div>
  <div class="digital-value {rpl_color}">${realized_pl:+,.2f}</div>
  <div class="digital-sub white">{n_closed_top} closed trades</div>
</div>""", unsafe_allow_html=True)

p2.markdown(f"""
<div class="digital-card">
  <div class="digital-label">Unrealized P/L</div>
  <div class="digital-value {pl_color}">${unreal_pl:+,.2f}</div>
  <div class="digital-sub white">{len(open_trades)} open trades</div>
</div>""", unsafe_allow_html=True)

p3.markdown(f"""
<div class="digital-card">
  <div class="digital-label">Total P/L</div>
  <div class="digital-value {tpl_color}">${total_pl:+,.2f}</div>
  <div class="digital-sub {tpl_color}">{total_pl/100_000*100:+.2f}% of $100k</div>
</div>""", unsafe_allow_html=True)

p4.markdown(f"""
<div class="digital-card">
  <div class="digital-label">Win Rate</div>
  <div class="digital-value {wr_color_t}">{wr_top:.1f}%</div>
  <div class="digital-sub white">closed trades</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Cumulative equity curve ───────────────────────────────────────────────────
if not df_j_top.empty and n_closed_top > 0:
    eq = (
        df_j_top[df_j_top["event"] == "close"]
        .dropna(subset=["realized_pl"])
        .copy()
    )
    eq["realized_pl"] = pd.to_numeric(eq["realized_pl"], errors="coerce")
    eq = eq.sort_values("timestamp")
    eq["cumulative_pl"] = eq["realized_pl"].cumsum()

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq["timestamp"], y=eq["cumulative_pl"],
        mode="lines+markers",
        line=dict(color="#3fb950" if realized_pl >= 0 else "#f85149", width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.08)" if realized_pl >= 0 else "rgba(248,81,73,0.08)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Cumulative P/L: $%{y:+,.2f}<extra></extra>",
    ))
    fig_eq.add_hline(y=0, line_dash="dash", line_color="#8b949e")
    fig_eq.update_layout(
        title="Cumulative Realized P/L",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="Courier New"),
        margin=dict(t=40, b=40, l=10, r=10), height=260,
        xaxis=dict(showgrid=False, tickformat="%b %d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", tickprefix="$",
                   zeroline=False, tickformat=",.0f"),
        hovermode="x unified", showlegend=False,
    )
    st.plotly_chart(fig_eq, width='stretch')

st.divider()

# ── Open trades P/L bar chart ─────────────────────────────────────────────────
st.subheader("Open Positions")

if not open_trades:
    st.info("No open trades.")
else:
    rows = []
    for t in open_trades:
        units     = float(t["currentUnits"])
        direction = "LONG" if units > 0 else "SHORT"
        entry     = float(t["price"])
        current   = float(t.get("currentPrice", entry))
        unreal    = float(t.get("unrealizedPL", 0))
        sl        = float(t.get("stopLossOrder", {}).get("price", 0)) or None
        tp        = float(t.get("takeProfitOrder", {}).get("price", 0)) or None
        tag       = t.get("clientExtensions", {}).get("tag") or \
                    t.get("tradeClientExtensions", {}).get("tag", "—")
        pct = (current - entry) / entry * 100 * (1 if direction == "LONG" else -1) if entry else 0.0
        rows.append({
            "Tag": tag, "Instrument": t["instrument"], "Dir": direction,
            "Entry": entry, "Current": current, "Move%": round(pct, 3),
            "Unreal P/L": unreal, "SL": sl, "TP": tp, "Units": int(abs(units)),
        })

    df_trades = pd.DataFrame(rows).sort_values("Unreal P/L")

    # Bar chart of unrealized P/L
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in df_trades["Unreal P/L"]]
    fig = go.Figure(go.Bar(
        x=df_trades["Tag"],
        y=df_trades["Unreal P/L"],
        marker_color=colors,
        text=[f"${v:+.0f}" for v in df_trades["Unreal P/L"]],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="Courier New"),
        margin=dict(t=20, b=40, l=10, r=10), height=280,
        xaxis=dict(showgrid=False, tickangle=-35, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=True,
                   zerolinecolor="#8b949e", tickprefix="$"),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')

    # Table
    display_df = df_trades.copy()
    display_df["Unreal P/L"] = display_df["Unreal P/L"].apply(lambda v: f"${v:+.2f}")
    display_df["Move%"] = display_df["Move%"].apply(lambda v: f"{v:+.3f}%")
    st.dataframe(display_df, hide_index=True, width='stretch')

    real_total = sum(float(t.get("unrealizedPL", 0)) for t in open_trades)
    st.markdown(f'<div style="font-family:Courier New;font-size:16px;color:{"#3fb950" if real_total>=0 else "#f85149"};font-weight:700;">Total unrealized: ${real_total:+,.2f} across {len(open_trades)} trades</div>', unsafe_allow_html=True)

st.divider()

# ── Balance history chart ─────────────────────────────────────────────────────
st.subheader("Balance History")

try:
    txns = fetch_transactions()
    points = [
        {"time": pd.to_datetime(t["time"]), "Balance": float(t["accountBalance"])}
        for t in txns if "accountBalance" in t
    ]
    if points:
        df_bal = pd.DataFrame(points).sort_values("time")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_bal["time"], y=df_bal["Balance"],
            mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=4, color="#58a6ff"),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
            name="Balance",
        ))
        # Start line at $100k
        fig2.add_hline(y=100_000, line_dash="dash", line_color="#8b949e",
                       annotation_text="Start $100k", annotation_font_color="#8b949e")
        fig2.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", family="Courier New"),
            margin=dict(t=10, b=40, l=10, r=10), height=300,
            xaxis=dict(showgrid=False, tickformat="%b %d %H:%M"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", tickprefix="$",
                       tickformat=",.0f"),
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No balance history yet — appears once a trade closes.")
except Exception as e:
    st.warning(f"Could not load transaction history: {e}")

st.divider()

# ── Trade journal ─────────────────────────────────────────────────────────────
st.subheader("Trade Journal")

df_j = fetch_journal()
if df_j.empty:
    st.info("No journal entries yet.")
else:
    closes = df_j[df_j["event"] == "close"].dropna(subset=["realized_pl"])
    if not closes.empty:
        summary = (
            closes.groupby("tag")["realized_pl"]
            .agg(trades="count", wins=lambda s: (s > 0).sum(),
                 total_pl="sum", avg_pl="mean")
            .reset_index()
            .sort_values("total_pl", ascending=False)
        )
        summary["win_rate"] = (summary["wins"] / summary["trades"] * 100).round(1).astype(str) + "%"
        summary["total_pl"] = summary["total_pl"].round(2)
        summary["avg_pl"]   = summary["avg_pl"].round(2)

        # P/L per sleeve bar chart
        fig3 = go.Figure(go.Bar(
            x=summary["tag"],
            y=summary["total_pl"],
            marker_color=["#3fb950" if v >= 0 else "#f85149" for v in summary["total_pl"]],
            text=[f"${v:+.0f}" for v in summary["total_pl"]],
            textposition="outside",
        ))
        fig3.update_layout(
            title="Realized P/L by Sleeve",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", family="Courier New"),
            margin=dict(t=40, b=60, l=10, r=10), height=300,
            xaxis=dict(showgrid=False, tickangle=-40, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=True,
                       zerolinecolor="#8b949e", tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig3, width='stretch')

        # Summary stats row
        total_closed_pl = closes["realized_pl"].sum()
        total_wr = (closes["realized_pl"] > 0).mean() * 100
        n_closed = len(closes)
        cs1, cs2, cs3 = st.columns(3)
        wr_color = "#3fb950" if total_wr >= 50 else "#f85149"
        pl_col   = "#3fb950" if total_closed_pl >= 0 else "#f85149"
        cs1.markdown(f'<div class="digital-card"><div class="digital-label">Closed Trades</div><div class="digital-value white">{n_closed}</div></div>', unsafe_allow_html=True)
        cs2.markdown(f'<div class="digital-card"><div class="digital-label">Win Rate</div><div class="digital-value" style="color:{wr_color}">{total_wr:.1f}%</div></div>', unsafe_allow_html=True)
        cs3.markdown(f'<div class="digital-card"><div class="digital-label">Total Realized P/L</div><div class="digital-value" style="color:{pl_col}">${total_closed_pl:+,.2f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.caption("Performance by sleeve")
        st.dataframe(summary[["tag","trades","win_rate","total_pl","avg_pl"]], hide_index=True, width='stretch')
        st.divider()

    st.caption("Full journal (newest first)")
    st.dataframe(df_j.iloc[::-1], hide_index=True, width='stretch')

st.divider()

# ── Live price chart ──────────────────────────────────────────────────────────
st.subheader("📊 Live Chart")

ALL_INSTRUMENTS = [
    "NATGAS_USD", "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "GBP_CHF", "GBP_JPY", "WHEAT_USD", "XAU_USD", "XAG_USD",
    "JP225_USD", "UK100_GBP", "USD_CAD", "NAS100_USD", "WTICO_USD",
    "AUD_JPY", "NZD_USD", "FR40_EUR",
]

# Default to open-trade instruments, fall back to full list
open_instruments = [t["instrument"] for t in open_trades] if open_trades else []
default_instr    = open_instruments[0] if open_instruments else "NATGAS_USD"

lc1, lc2 = st.columns([3, 1])
with lc1:
    chart_instr = st.selectbox(
        "Instrument",
        options=open_instruments + [i for i in ALL_INSTRUMENTS if i not in open_instruments],
        index=0,
        label_visibility="collapsed",
    )
with lc2:
    tf_options = {"M5": "M5", "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4", "D": "D"}
    chart_tf   = st.selectbox("Timeframe", list(tf_options.keys()),
                               index=2, label_visibility="collapsed")

try:
    df_c = fetch_candles(chart_instr, tf_options[chart_tf], count=120)
    if df_c.empty:
        st.info("No candle data returned.")
    else:
        # Find open trade on this instrument for level lines
        trade_on_chart = next(
            (t for t in open_trades if t["instrument"] == chart_instr), None
        )

        fig_c = go.Figure()

        # Candlestick
        fig_c.add_trace(go.Candlestick(
            x=df_c["time"],
            open=df_c["Open"], high=df_c["High"],
            low=df_c["Low"],   close=df_c["Close"],
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950",
            decreasing_fillcolor="#f85149",
            line_width=1,
            name=chart_instr,
        ))

        # Trade levels
        if trade_on_chart:
            entry = float(trade_on_chart["price"])
            sl    = float(trade_on_chart.get("stopLossOrder",   {}).get("price", 0)) or None
            tp    = float(trade_on_chart.get("takeProfitOrder", {}).get("price", 0)) or None
            units = float(trade_on_chart["currentUnits"])
            direction = "LONG" if units > 0 else "SHORT"
            unreal_pl = float(trade_on_chart.get("unrealizedPL", 0))
            pl_col = "#3fb950" if unreal_pl >= 0 else "#f85149"

            fig_c.add_hline(y=entry, line_color="#58a6ff", line_width=1.5,
                            line_dash="dash",
                            annotation_text=f"Entry {entry:.5g}",
                            annotation_font_color="#58a6ff",
                            annotation_position="left")
            if sl:
                fig_c.add_hline(y=sl, line_color="#f85149", line_width=1.5,
                                annotation_text=f"SL {sl:.5g}",
                                annotation_font_color="#f85149",
                                annotation_position="left")
            if tp:
                fig_c.add_hline(y=tp, line_color="#3fb950", line_width=1.5,
                                annotation_text=f"TP {tp:.5g}",
                                annotation_font_color="#3fb950",
                                annotation_position="left")

            st.markdown(
                f'<div style="font-family:Courier New;font-size:13px;margin-bottom:6px;">'
                f'<span style="color:#58a6ff">■</span> {direction} {int(abs(units)):,} units &nbsp;|&nbsp;'
                f'Entry <b>{entry:.5g}</b> &nbsp;|&nbsp;'
                f'<span style="color:{pl_col}">P/L ${unreal_pl:+,.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        fig_c.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3", family="Courier New"),
            margin=dict(t=10, b=40, l=60, r=80),
            height=480,
            xaxis=dict(
                showgrid=False, rangeslider_visible=False,
                type="date",
                tickformat="%d %b %H:%M" if chart_tf in ("M5","M15","M30","H1","H4") else "%d %b %Y",
            ),
            yaxis=dict(showgrid=True, gridcolor="#21262d", side="right"),
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig_c, width='stretch')

except Exception as e:
    st.warning(f"Could not load chart: {e}")
