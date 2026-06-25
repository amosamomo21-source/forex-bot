"""IC Markets / cTrader Open API wrapper.

Needs CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, and
CTRADER_CTID_TRADER_ACCOUNT_ID in .env (run get_token.py once, then this
module's __main__ block once, to get them).

Defaults hard to the demo endpoint. Live trading requires the same double
opt-in as broker.py: allow_live=True AND CTRADER_LIVE_TRADING_CONFIRMED=yes.

Twisted's reactor must be installed on, and driven by, the same asyncio event
loop as the rest of the process -- it cannot coexist with a plain
asyncio.run() call elsewhere. install_asyncio_reactor() must run before
anything (including this module) imports ctrader_open_api, since that
package binds Twisted's default reactor as a side effect of its own imports.
It also calls reactor.startRunning() explicitly -- without it, anything that
relies on Twisted's thread pool (e.g. resolving a hostname, which is how the
Client's "ssl:host:port" endpoint connects) hangs forever with no error,
because reactor.run() is what normally fires that startup trigger and here
we drive the loop ourselves instead. Wiring this into trading_agent.py later
means swapping its asyncio.run(run(prompt)) entry point for an explicit
loop.run_until_complete(...) on the same loop installed here.

connect(), account_summary(), get_price(), and get_candles() have been
live-tested against the demo endpoint (EURUSD, symbolId=1) and work,
including the delta-decoded OHLC math. place_market_order() has NOT been
tested -- confirm the volume scaling for the symbol you're trading (see its
docstring) before using it.
"""

import asyncio
import os
import sys
import time

from twisted.internet import asyncioreactor


def install_asyncio_reactor(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.AbstractEventLoop:
    """Installs Twisted's reactor on top of the given asyncio loop, then fires
    Twisted's startup triggers via startRunning() -- without this, things that rely on
    the reactor's thread pool (e.g. DNS resolution of hostnames, used by Client's
    "ssl:host:port" endpoint) hang forever with no error, because reactor.run() is what
    normally fires those triggers and we're driving the loop ourselves instead.
    installSignalHandlers=False because reactor.run() (which normally installs them) is
    never called -- and signal.signal() itself only works from the main thread, which
    this module is not always imported from (e.g. Streamlit runs each session's script
    in a worker thread).

    A reactor can only be installed once per process. Streamlit re-execs this
    module's top-level code (without a fresh process) on every script rerun, which
    would otherwise hit Twisted's ReactorAlreadyInstalledError on the second rerun.
    Checking sys.modules directly (the same check Twisted's own installReactor() uses)
    detects that case without itself triggering an auto-install of the default
    reactor, which a plain `from twisted.internet import reactor` probe would. When
    already installed, reuse its actual loop rather than the fresh, disconnected one
    `loop` would otherwise default to."""
    if "twisted.internet.reactor" in sys.modules:
        return sys.modules["twisted.internet.reactor"]._asyncioEventloop

    loop = loop or asyncio.new_event_loop()
    asyncioreactor.install(loop)
    from twisted.internet import reactor

    reactor.startRunning(installSignalHandlers=False)
    return loop


loop = install_asyncio_reactor()

from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol  # noqa: E402
from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model  # noqa: E402

PERIOD_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "H12": 720, "D1": 1440, "W1": 10080, "MN1": 43200,
}


class LiveTradingNotConfirmedError(Exception):
    pass


class CTraderBroker:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        ctid_trader_account_id: int | None = None,
        allow_live: bool = False,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.ctid_trader_account_id = ctid_trader_account_id

        if allow_live:
            if os.environ.get("CTRADER_LIVE_TRADING_CONFIRMED") != "yes":
                raise LiveTradingNotConfirmedError(
                    "allow_live=True but CTRADER_LIVE_TRADING_CONFIRMED env var is not "
                    "set to 'yes'. This is a real-money account. Set the env var "
                    "explicitly, in addition to allow_live=True, to confirm you intend "
                    "to trade live."
                )
            host = EndPoints.PROTOBUF_LIVE_HOST
            print("\n*** LIVE TRADING ENABLED -- ORDERS WILL USE REAL MONEY ***\n")
        else:
            host = EndPoints.PROTOBUF_DEMO_HOST

        self.environment = "live" if allow_live else "demo"
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self._spot_waiters: dict[int, asyncio.Future] = {}
        self._symbols_by_name: dict[str, int] | None = None
        self._symbol_names_by_id: dict[int, str] | None = None
        self._asset_names_by_id: dict[int, str] | None = None
        self.client.setMessageReceivedCallback(self._on_message)

    def _on_message(self, _client, message) -> None:
        if message.payloadType == Protobuf.get("SpotEvent").payloadType:
            event = Protobuf.extract(message)
            waiter = self._spot_waiters.get(event.symbolId)
            if waiter and not waiter.done():
                waiter.set_result(event)

    async def _send(self, payload_name: str, **params):
        deferred = self.client.send(payload_name, **params)
        response = await deferred.asFuture(asyncio.get_event_loop())
        return Protobuf.extract(response)

    async def connect(self) -> None:
        self.client.startService()
        await self._send("ApplicationAuthReq", clientId=self.client_id, clientSecret=self.client_secret)
        if self.ctid_trader_account_id is not None:
            await self._send(
                "AccountAuthReq",
                ctidTraderAccountId=self.ctid_trader_account_id,
                accessToken=self.access_token,
            )

    async def get_account_list(self) -> list:
        """Call after connect() (with ctid_trader_account_id left as None) to discover
        the ctidTraderAccountId(s) tied to this access token."""
        res = await self._send("GetAccountListByAccessTokenReq", accessToken=self.access_token)
        return list(res.ctidTraderAccount)

    async def account_summary(self) -> dict:
        res = await self._send("TraderReq", ctidTraderAccountId=self.ctid_trader_account_id)
        trader = res.trader
        scale = 10 ** trader.moneyDigits
        return {
            "balance": trader.balance / scale,
            "deposit_asset_id": trader.depositAssetId,
            "leverage": trader.leverageInCents / 100 if trader.leverageInCents else None,
        }

    async def get_deposit_currency(self) -> str:
        """Resolves account_summary()'s deposit_asset_id to a currency code (e.g.
        "USD"), fetching and caching the full asset list on first call. Verified
        live: the demo account's deposit asset is USD (assetId=16)."""
        if self._asset_names_by_id is None:
            res = await self._send(
                "ProtoOAAssetListReq", ctidTraderAccountId=self.ctid_trader_account_id
            )
            self._asset_names_by_id = {a.assetId: a.name for a in res.asset}
        summary = await self.account_summary()
        return self._asset_names_by_id.get(summary["deposit_asset_id"], "?")

    async def _ensure_symbols_loaded(self) -> None:
        if self._symbols_by_name is None:
            res = await self._send(
                "SymbolsListReq", ctidTraderAccountId=self.ctid_trader_account_id
            )
            self._symbols_by_name = {s.symbolName: s.symbolId for s in res.symbol}
            self._symbol_names_by_id = {s.symbolId: s.symbolName for s in res.symbol}

    async def get_symbol_id(self, symbol_name: str) -> int:
        """Resolves a ticker name (e.g. "EURUSD") to cTrader's numeric symbolId,
        fetching and caching the full symbol list on first call. Verified live:
        EURUSD == 1."""
        await self._ensure_symbols_loaded()
        try:
            return self._symbols_by_name[symbol_name]
        except KeyError:
            raise ValueError(f"Unknown cTrader symbol '{symbol_name}'") from None

    async def get_symbol_name(self, symbol_id: int) -> str:
        """Reverse of get_symbol_id -- resolves a numeric symbolId back to its
        ticker name (e.g. 1 -> "EURUSD"), for labeling deal history."""
        await self._ensure_symbols_loaded()
        return self._symbol_names_by_id.get(symbol_id, f"#{symbol_id}")

    async def get_price(self, symbol_id: int, timeout: float = 10.0) -> dict:
        """bid/ask are divided by the common 1e5 relative-price scale used for most FX
        symbols. Not yet verified against a real connection -- some symbols use a
        different digit count (see ProtoOASymbolByIdRes.digits) and would need rescaling."""
        loop = asyncio.get_event_loop()
        waiter = loop.create_future()
        self._spot_waiters[symbol_id] = waiter
        try:
            await self._send(
                "SubscribeSpotsReq",
                ctidTraderAccountId=self.ctid_trader_account_id,
                symbolId=[symbol_id],
            )
            event = await asyncio.wait_for(waiter, timeout=timeout)
            return {"bid": event.bid / 100000, "ask": event.ask / 100000}
        finally:
            self._spot_waiters.pop(symbol_id, None)
            await self._send(
                "UnsubscribeSpotsReq",
                ctidTraderAccountId=self.ctid_trader_account_id,
                symbolId=[symbol_id],
            )

    async def get_candles(self, symbol_id: int, period: str = "H1", count: int = 500) -> list:
        """OHLC is divided by the same common 1e5 relative-price scale as get_price --
        see that method's caveat about symbols with a different digit count."""
        period_enum = getattr(model, period)
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - PERIOD_MINUTES[period] * count * 60 * 1000

        res = await self._send(
            "GetTrendbarsReq",
            ctidTraderAccountId=self.ctid_trader_account_id,
            period=period_enum,
            symbolId=symbol_id,
            fromTimestamp=from_ms,
            toTimestamp=now_ms,
            count=count,
        )
        candles = []
        for bar in res.trendbar:
            candles.append({
                "timestamp_minutes": bar.utcTimestampInMinutes,
                "low": bar.low / 100000,
                "open": (bar.low + bar.deltaOpen) / 100000,
                "high": (bar.low + bar.deltaHigh) / 100000,
                "close": (bar.low + bar.deltaClose) / 100000,
                "volume": bar.volume,
            })
        return candles

    async def get_deal_history(self, days: int = 90, max_rows: int = 1000) -> list:
        """Closed and open deals (filled orders) over the last `days` days, oldest
        first. Live-verified shape: each deal has a tradeSide (BUY/SELL), volume
        (hundredths of a unit, same scale as place_market_order), executionPrice,
        and executionTimestamp (ms). Deals that closed a position additionally
        carry closePositionDetail.balance -- the running account balance
        immediately after that close, scaled by closePositionDetail.moneyDigits --
        that's what to plot for a balance-over-time chart. Deals that only opened
        a position have no closePositionDetail."""
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - days * 24 * 60 * 60 * 1000
        res = await self._send(
            "ProtoOADealListReq",
            ctidTraderAccountId=self.ctid_trader_account_id,
            fromTimestamp=from_ms,
            toTimestamp=now_ms,
            maxRows=max_rows,
        )
        deals = []
        for d in res.deal:
            scale = 10 ** d.moneyDigits
            deal = {
                "deal_id": d.dealId,
                "position_id": d.positionId,
                "symbol_id": d.symbolId,
                "symbol": await self.get_symbol_name(d.symbolId),
                "trade_side": model.ProtoOATradeSide.Name(d.tradeSide),
                "volume": d.volume,
                "execution_price": d.executionPrice,
                "execution_timestamp_ms": d.executionTimestamp,
                "commission": d.commission / scale,
                "status": model.ProtoOADealStatus.Name(d.dealStatus),
            }
            if d.HasField("closePositionDetail"):
                cpd = d.closePositionDetail
                cpd_scale = 10 ** cpd.moneyDigits
                deal["gross_profit"] = cpd.grossProfit / cpd_scale
                deal["swap"] = cpd.swap / cpd_scale
                deal["balance_after"] = cpd.balance / cpd_scale
            deals.append(deal)
        return deals

    async def place_market_order(
        self,
        symbol_id: int,
        volume: int,
        trade_side: str,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> dict:
        """trade_side: "BUY" or "SELL". volume is in hundredths of a unit of the base
        currency (confirmed live on the EURUSD demo: lotSize=10_000_000 == 100 lots ==
        10_000_000 / 100 units, minVolume=100_000 == 0.01 lot == 1_000 units) -- this
        ratio is broker/symbol-wide, not per-symbol, but check ProtoOASymbolByIdRes for
        a symbol's own minVolume/stepVolume before sizing an order. Live-tested with a
        100_000 (0.01 lot) market BUY on EURUSD demo -- filled immediately, returns a
        ProtoOAExecutionEvent with executionType ORDER_ACCEPTED on the initial response;
        the actual fill is a separate event delivered to the message-received callback,
        not captured by this method's return value -- call ReconcileReq after to confirm
        the resulting position."""
        params = {
            "ctidTraderAccountId": self.ctid_trader_account_id,
            "symbolId": symbol_id,
            "orderType": model.MARKET,
            "tradeSide": getattr(model, trade_side),
            "volume": volume,
            "timeInForce": model.IMMEDIATE_OR_CANCEL,
        }
        if stop_loss_price is not None:
            params["stopLoss"] = stop_loss_price
        if take_profit_price is not None:
            params["takeProfit"] = take_profit_price

        res = await self._send("NewOrderReq", **params)
        return res

    async def close_position(self, position_id: int, volume: int) -> dict:
        """volume is in the same hundredths-of-a-unit scale as place_market_order --
        pass the position's full tradeData.volume (from ReconcileReq) to close it
        entirely, or less to partially close. Live-tested on the EURUSD demo: opens
        as a closing MARKET order on the opposite side, returns the same
        ProtoOAExecutionEvent/ORDER_ACCEPTED shape as place_market_order (not the
        final fill) -- confirm via ReconcileReq that the position is actually gone."""
        res = await self._send(
            "ClosePositionReq",
            ctidTraderAccountId=self.ctid_trader_account_id,
            positionId=position_id,
            volume=volume,
        )
        return res


def from_env(allow_live: bool = False) -> CTraderBroker:
    client_id = os.environ.get("CTRADER_CLIENT_ID")
    client_secret = os.environ.get("CTRADER_CLIENT_SECRET")
    access_token = os.environ.get("CTRADER_ACCESS_TOKEN")
    ctid_trader_account_id = os.environ.get("CTRADER_CTID_TRADER_ACCOUNT_ID")
    if not client_id or not client_secret or not access_token:
        raise RuntimeError(
            "Set CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET and CTRADER_ACCESS_TOKEN "
            "(in .env) before using the broker. Run get_token.py first."
        )
    return CTraderBroker(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        ctid_trader_account_id=int(ctid_trader_account_id) if ctid_trader_account_id else None,
        allow_live=allow_live,
    )


async def _discover_account_id() -> None:
    """Run once, after get_token.py has saved a real access token, to find the
    ctidTraderAccountId to put in .env: uv run python3 ctrader_broker.py"""
    from dotenv import load_dotenv

    load_dotenv()
    broker = from_env()
    await broker.connect()
    accounts = await broker.get_account_list()
    for account in accounts:
        print(f"ctidTraderAccountId={account.ctidTraderAccountId} isLive={account.isLive} login={account.traderLogin}")


if __name__ == "__main__":
    loop.run_until_complete(_discover_account_id())
