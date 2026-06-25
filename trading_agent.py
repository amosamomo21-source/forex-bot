import json
import subprocess
import sys

from dotenv import load_dotenv
from claude_agent_sdk import (
    query,
    tool,
    create_sdk_mcp_server,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
)

load_dotenv()

import ctrader_broker  # noqa: E402 -- must come after load_dotenv, installs the asyncio/Twisted reactor on import

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are a quantitative trading strategy developer working in this forex-bot project \
(data.py, strategies.py, backtest.py, broker.py). Your job is to design, implement, and \
rigorously test trading strategies and report HONEST results from real backtests.

Hard rules, non-negotiable:
- Never claim or target a specific win rate. Win rate is not a proxy for profitability --
  profit factor, Sharpe ratio, and max drawdown matter far more. A 30% win rate can be very
  profitable; a 90% win rate can still blow up an account.
- Always size positions by risk (a fixed % of equity per trade, sized off the stop
  distance), never by raw leverage on full account equity -- that is how accounts blow up.
- After every change to strategies.py, call the run_backtest tool and report the actual
  numbers in your reply, including unflattering ones. Never cherry-pick favorable windows
  or omit a bad result.
- If a strategy has been re-tuned multiple times against the same historical window,
  explicitly flag overfitting risk and recommend testing on a held-out period or different
  instrument before trusting the result.
- Never attempt to place live trades or remove/bypass the live-trading guard in broker.py.
  Live order placement requires the human operator to set OANDA_LIVE_TRADING_CONFIRMED
  themselves -- that decision is never yours to make.
- The ctrader_* tools are read-only market data (account summary, live price, candles)
  from the IC Markets cTrader demo account. You have no order-placement capability through
  them -- use them only to inform analysis, never imply a trade was placed.
"""


@tool(
    "run_backtest",
    "Run a backtest for a strategy defined in strategies.py against historical FX data. "
    "Returns real performance metrics: return %, Sharpe ratio, max drawdown, win rate, "
    "profit factor, number of trades, and more. Always call this after editing "
    "strategies.py, and report the numbers verbatim -- do not paraphrase them optimistically. "
    "source defaults to 'oanda' (the real broker feed) -- only use 'yfinance' if explicitly "
    "asked to compare against the proxy data.",
    {"strategy_name": str, "ticker": str, "period": str, "source": str},
)
async def run_backtest_tool(args: dict) -> dict:
    strategy_name = args.get("strategy_name", "ema_crossover")
    ticker = args.get("ticker", "EURUSD=X")
    period = args.get("period", "5y")
    source = args.get("source", "oanda")

    proc = subprocess.run(
        [
            sys.executable, "backtest.py", strategy_name,
            "--ticker", ticker, "--period", period, "--source", source, "--json",
        ],
        capture_output=True,
        text=True,
        cwd=".",
    )

    if proc.returncode != 0:
        return {"content": [{"type": "text", "text": f"Backtest failed:\n{proc.stderr}"}]}

    stats = json.loads(proc.stdout)
    return {"content": [{"type": "text", "text": json.dumps(stats, indent=2)}]}


backtest_server = create_sdk_mcp_server(
    name="backtest",
    version="1.0.0",
    tools=[run_backtest_tool],
)


_ctrader_broker: ctrader_broker.CTraderBroker | None = None


async def _get_ctrader_broker() -> ctrader_broker.CTraderBroker:
    global _ctrader_broker
    if _ctrader_broker is None:
        _ctrader_broker = ctrader_broker.from_env()
        await _ctrader_broker.connect()
    return _ctrader_broker


@tool(
    "ctrader_account_summary",
    "Get the IC Markets cTrader demo account's balance and leverage. Read-only.",
    {},
)
async def ctrader_account_summary_tool(args: dict) -> dict:
    try:
        broker = await _get_ctrader_broker()
        summary = await broker.account_summary()
        return {"content": [{"type": "text", "text": json.dumps(summary, indent=2)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"cTrader account_summary failed: {e}"}]}


@tool(
    "ctrader_price",
    "Get the current live bid/ask for an FX ticker (e.g. 'EURUSD') from the IC Markets "
    "cTrader demo feed. Read-only.",
    {"symbol": str},
)
async def ctrader_price_tool(args: dict) -> dict:
    try:
        broker = await _get_ctrader_broker()
        symbol_id = await broker.get_symbol_id(args["symbol"])
        price = await broker.get_price(symbol_id)
        return {"content": [{"type": "text", "text": json.dumps(price, indent=2)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"cTrader price lookup failed: {e}"}]}


@tool(
    "ctrader_candles",
    "Get recent OHLC candles for an FX ticker (e.g. 'EURUSD') from the IC Markets cTrader "
    "demo feed. period is one of M1/M5/M15/M30/H1/H4/D1. Read-only.",
    {"symbol": str, "period": str, "count": int},
)
async def ctrader_candles_tool(args: dict) -> dict:
    try:
        broker = await _get_ctrader_broker()
        symbol_id = await broker.get_symbol_id(args["symbol"])
        candles = await broker.get_candles(
            symbol_id, period=args.get("period", "H1"), count=args.get("count", 50)
        )
        return {"content": [{"type": "text", "text": json.dumps(candles, indent=2)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"cTrader candles lookup failed: {e}"}]}


ctrader_server = create_sdk_mcp_server(
    name="ctrader",
    version="1.0.0",
    tools=[ctrader_account_summary_tool, ctrader_price_tool, ctrader_candles_tool],
)


async def run(prompt: str) -> None:
    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        cwd=".",
        mcp_servers={"backtest": backtest_server, "ctrader": ctrader_server},
        allowed_tools=[
            "Read", "Write", "Edit", "Glob", "Grep",
            "mcp__backtest__run_backtest",
            "mcp__ctrader__ctrader_account_summary",
            "mcp__ctrader__ctrader_price",
            "mcp__ctrader__ctrader_candles",
        ],
        permission_mode="acceptEdits",
    )

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
                elif hasattr(block, "name"):
                    print(f"\n[tool] {block.name}({getattr(block, 'input', '')})")
        elif isinstance(message, ResultMessage):
            print(f"\n--- {message.subtype} ---")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run trading_agent.py "<task for the strategy developer>"')
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    # Not asyncio.run() -- ctrader_broker installed Twisted's reactor on a specific loop
    # at import time, and Twisted's connection machinery only works when driven by that
    # same loop (see ctrader_broker.py's module docstring).
    ctrader_broker.loop.run_until_complete(run(prompt))


if __name__ == "__main__":
    main()
