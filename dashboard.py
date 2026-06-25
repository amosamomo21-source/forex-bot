"""Local trade-progression dashboard.

Run with: uv run streamlit run dashboard.py

Shows current balance and trade/transaction history for both the OANDA
practice account and the IC Markets cTrader demo account. Read-only -- never
places, modifies, or closes any order.
"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import broker  # noqa: E402
import ctrader_broker  # noqa: E402 -- installs the asyncio/Twisted reactor on import
import journal  # noqa: E402

st.set_page_config(page_title="Trade Progression", layout="wide")
st.title("Trade Progression")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()


@st.cache_resource
def get_oanda_broker() -> broker.OandaBroker:
    return broker.from_env()


@st.cache_resource
def get_ctrader_broker() -> ctrader_broker.CTraderBroker:
    b = ctrader_broker.from_env()
    ctrader_broker.loop.run_until_complete(b.connect())
    return b


@st.cache_data(ttl=15)
def fetch_oanda_data():
    b = get_oanda_broker()
    summary = b.account_summary()["account"]
    txns = b.get_transaction_history()
    return summary, txns


async def _fetch_ctrader_async(b: ctrader_broker.CTraderBroker):
    summary = await b.account_summary()
    currency = await b.get_deposit_currency()
    deals = await b.get_deal_history(days=90)
    return summary, currency, deals


@st.cache_data(ttl=15)
def fetch_ctrader_data():
    b = get_ctrader_broker()
    return ctrader_broker.loop.run_until_complete(_fetch_ctrader_async(b))


def render_oanda() -> None:
    try:
        summary, txns = fetch_oanda_data()
    except Exception as e:
        st.error(f"Could not reach OANDA: {e}")
        return

    currency = summary["currency"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Balance", f"{float(summary['balance']):,.2f} {currency}")
    col2.metric("Unrealized P/L", f"{float(summary['unrealizedPL']):,.2f} {currency}")
    col3.metric("Open positions", summary["openPositionCount"])

    balance_points = [
        {"time": t["time"], "balance": float(t["accountBalance"])}
        for t in txns
        if "accountBalance" in t
    ]
    if balance_points:
        df = pd.DataFrame(balance_points)
        df["time"] = pd.to_datetime(df["time"])
        st.line_chart(df.set_index("time")["balance"])
    else:
        st.info("No balance history yet -- place a trade to see it here.")

    if txns:
        st.subheader("Transaction history")
        df = pd.DataFrame(txns)
        cols = [c for c in ["time", "type", "amount", "accountBalance"] if c in df.columns]
        st.dataframe(df[cols].iloc[::-1], width="stretch", hide_index=True)
    else:
        st.info("No transactions yet.")


def render_ctrader() -> None:
    try:
        summary, currency, deals = fetch_ctrader_data()
    except Exception as e:
        st.error(f"Could not reach cTrader: {e}")
        return

    col1, col2 = st.columns(2)
    col1.metric("Balance", f"{summary['balance']:,.2f} {currency}")
    col2.metric("Leverage", f"{summary['leverage']:.0f}:1" if summary["leverage"] else "n/a")

    balance_points = [
        {
            "time": datetime.fromtimestamp(d["execution_timestamp_ms"] / 1000, tz=timezone.utc),
            "balance": d["balance_after"],
        }
        for d in deals
        if "balance_after" in d
    ]
    if balance_points:
        df = pd.DataFrame(balance_points)
        st.line_chart(df.set_index("time")["balance"])
    else:
        st.info("No closed trades yet -- balance history appears once a position is closed.")

    if deals:
        st.subheader("Deal history")
        df = pd.DataFrame(deals)
        df["time"] = pd.to_datetime(df["execution_timestamp_ms"], unit="ms", utc=True)
        cols = [
            c
            for c in [
                "time", "symbol", "trade_side", "volume", "execution_price",
                "commission", "gross_profit", "balance_after", "status",
            ]
            if c in df.columns
        ]
        st.dataframe(df[cols].iloc[::-1], width="stretch", hide_index=True)
    else:
        st.info("No deals yet.")


@st.cache_data(ttl=15)
def fetch_journal_data():
    if not journal.JOURNAL_PATH.exists():
        return None
    df = pd.read_csv(journal.JOURNAL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def render_journal() -> None:
    df = fetch_journal_data()
    if df is None or df.empty:
        st.info("No journal entries yet -- live_runner.py writes here on every open/close.")
        return

    closes = df[df["event"] == "close"].dropna(subset=["realized_pl"])
    if not closes.empty:
        st.subheader("Per-sleeve performance")
        summary = (
            closes.groupby("tag")["realized_pl"]
            .agg(trades="count", win_rate=lambda s: (s > 0).mean(), total_pl="sum", avg_pl="mean")
            .reset_index()
        )
        summary["win_rate"] = (summary["win_rate"] * 100).round(1)
        summary[["total_pl", "avg_pl"]] = summary[["total_pl", "avg_pl"]].round(2)
        st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Trade journal")
    st.dataframe(df.iloc[::-1], width="stretch", hide_index=True)


tab_oanda, tab_ctrader, tab_journal = st.tabs(
    ["OANDA (practice)", "cTrader (IC Markets demo)", "Trade journal"]
)
with tab_oanda:
    render_oanda()
with tab_ctrader:
    render_ctrader()
with tab_journal:
    render_journal()
