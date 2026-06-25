"""OANDA v20 REST API wrapper.

Defaults hard to the practice (demo) environment. Live trading requires two
separate, explicit opt-ins -- a constructor flag AND an environment variable
-- so a stray default or copy-pasted call can never place a real order.

Setup (you do this yourself -- nothing here can create the account for you):
  1. Create a free practice account at https://www.oanda.com/demo-account/
  2. Generate a personal access token from the practice account's API page
  3. Put OANDA_API_TOKEN and OANDA_ACCOUNT_ID in your .env file
"""

import os

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.transactions as transactions


class LiveTradingNotConfirmedError(Exception):
    pass


class OandaBroker:
    def __init__(self, api_token: str, account_id: str, allow_live: bool = False):
        self.account_id = account_id

        if allow_live:
            if os.environ.get("OANDA_LIVE_TRADING_CONFIRMED") != "yes":
                raise LiveTradingNotConfirmedError(
                    "allow_live=True but OANDA_LIVE_TRADING_CONFIRMED env var is not set to "
                    "'yes'. This is a real-money account. Set the env var explicitly, in "
                    "addition to allow_live=True, to confirm you intend to trade live."
                )
            environment = "live"
            print("\n*** LIVE TRADING ENABLED -- ORDERS WILL USE REAL MONEY ***\n")
        else:
            environment = "practice"

        self.environment = environment
        self.api = oandapyV20.API(access_token=api_token, environment=environment)

    def account_summary(self) -> dict:
        r = accounts.AccountSummary(accountID=self.account_id)
        self.api.request(r)
        return r.response

    def get_price(self, instrument: str) -> dict:
        params = {"instruments": instrument}
        r = pricing.PricingInfo(accountID=self.account_id, params=params)
        self.api.request(r)
        return r.response["prices"][0]

    def get_candles(
        self, instrument: str, granularity: str = "H1", count: int = 500, to: str | None = None
    ) -> list:
        """to: RFC3339 timestamp -- fetches `count` candles ending at that time,
        for paginating past OANDA's 5000-candles-per-request cap."""
        params = {"granularity": granularity, "count": count, "price": "M"}
        if to is not None:
            params["to"] = to
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        self.api.request(r)
        return r.response["candles"]

    def get_transaction_history(self) -> list:
        """All transactions on the account, oldest first -- account creation,
        funding, config changes, and order fills. Balance-affecting types (e.g.
        TRANSFER_FUNDS, ORDER_FILL) carry an 'accountBalance' field, the running
        balance immediately after that transaction -- that's what to plot for a
        balance-over-time chart. Verified live: TransactionList's default date
        window can exclude an account's actual transactions, so this fetches the
        full range explicitly by ID instead."""
        last_id = int(self.account_summary()["lastTransactionID"])
        if last_id < 1:
            return []
        params = {"from": "1", "to": str(last_id)}
        r = transactions.TransactionIDRange(accountID=self.account_id, params=params)
        self.api.request(r)
        return r.response["transactions"]

    def get_open_trade(self, instrument: str) -> dict | None:
        """The single open trade on `instrument`, if any, else None. Assumes at
        most one open trade per instrument -- callers should only open a new
        position after confirming none is already open. For instruments that
        more than one independent strategy might trade, use get_open_trades
        and filter by clientExtensions tag instead."""
        r = trades.OpenTrades(accountID=self.account_id)
        self.api.request(r)
        for t in r.response["trades"]:
            if t["instrument"] == instrument:
                return t
        return None

    def get_open_trades(self, instrument: str | None = None) -> list:
        """All open trades, optionally filtered to one instrument. Unlike
        get_open_trade, doesn't assume at most one trade per instrument --
        needed once independent strategies can each hold a position on the
        same instrument at once, distinguished by their clientExtensions tag
        (see place_market_order's client_tag)."""
        r = trades.OpenTrades(accountID=self.account_id)
        self.api.request(r)
        out = r.response["trades"]
        if instrument is not None:
            out = [t for t in out if t["instrument"] == instrument]
        return out

    def close_position(self, instrument: str) -> dict:
        """Closes the entire open position on `instrument` (whichever side --
        long or short -- is actually open). Only safe when at most one
        strategy ever trades this instrument -- it closes every trade on the
        instrument together. Use close_trade for one specific trade."""
        trade = self.get_open_trade(instrument)
        if trade is None:
            raise RuntimeError(f"No open trade on {instrument} to close")
        side = "longUnits" if float(trade["currentUnits"]) > 0 else "shortUnits"
        r = positions.PositionClose(accountID=self.account_id, instrument=instrument, data={side: "ALL"})
        self.api.request(r)
        return r.response

    def close_trade(self, trade_id: str) -> dict:
        """Closes one specific trade by ID, leaving any other open trades on
        the same instrument (e.g. belonging to a different tagged strategy)
        untouched."""
        r = trades.TradeClose(accountID=self.account_id, tradeID=trade_id, data={"units": "ALL"})
        self.api.request(r)
        return r.response

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        client_tag: str | None = None,
    ) -> dict:
        """units: positive to buy, negative to sell. client_tag: written to
        the resulting Trade's clientExtensions.tag (via tradeClientExtensions
        on the order -- distinct from the order's own clientExtensions), so
        get_open_trades() can later attribute this trade to a specific
        strategy even when multiple strategies trade the same instrument."""
        order = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if stop_loss_price is not None:
            order["order"]["stopLossOnFill"] = {"price": f"{stop_loss_price:.5f}"}
        if take_profit_price is not None:
            order["order"]["takeProfitOnFill"] = {"price": f"{take_profit_price:.5f}"}
        if client_tag is not None:
            order["order"]["tradeClientExtensions"] = {"tag": client_tag}

        r = orders.OrderCreate(accountID=self.account_id, data=order)
        self.api.request(r)
        return r.response


def from_env(allow_live: bool = False) -> OandaBroker:
    token = os.environ.get("OANDA_API_TOKEN")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    if not token or not account_id:
        raise RuntimeError(
            "Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID (in .env) before using the broker."
        )
    return OandaBroker(api_token=token, account_id=account_id, allow_live=allow_live)
