import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


def ema(series, n):
    return pd.Series(series).ewm(span=n, adjust=False).mean()


def sma(series, n):
    return pd.Series(series).rolling(n).mean()


def rolling_std(series, n):
    return pd.Series(series).rolling(n).std(ddof=0)


def atr(high, low, close, n=14):
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(n).mean()


def rsi(series, n=14):
    series = pd.Series(series)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    s = pd.Series(series)
    macd_line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def adx(high, low, close, n=14):
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)
    up = high - prev_high
    dn = prev_low - low
    plus_dm  = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=low.index)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_s    = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean()  / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def _highest(series, n):
    return pd.Series(series).rolling(n).max()


def _lowest(series, n):
    return pd.Series(series).rolling(n).min()


class EmaCrossoverAtr(Strategy):
    """Trend-following EMA crossover with ATR-based stop-loss/take-profit.

    Baseline only -- expect a win rate well under 50%. Profitability (if
    any) comes from winners being larger than losers (tp_atr_mult > sl_atr_mult),
    not from being right most of the time.
    """

    fast = 20
    slow = 50
    atr_period = 14
    sl_atr_mult = 1.5
    tp_atr_mult = 2.5
    risk_pct = 0.01  # fraction of equity risked per trade (stop distance defines the risk)

    def init(self):
        self.ema_fast = self.I(ema, self.data.Close, self.fast)
        self.ema_slow = self.I(ema, self.data.Close, self.slow)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def _size_for_risk(self, stop_distance):
        risk_amount = self.equity * self.risk_pct
        units = int(risk_amount / stop_distance)
        return max(units, 1)

    def next(self):
        price = self.data.Close[-1]
        a = self.atr[-1]
        if a != a:  # NaN guard during warmup
            return

        stop_distance = self.sl_atr_mult * a
        size = self._size_for_risk(stop_distance)

        if crossover(self.ema_fast, self.ema_slow):
            self.position.close()
            self.buy(size=size, sl=price - stop_distance, tp=price + self.tp_atr_mult * a)
        elif crossover(self.ema_slow, self.ema_fast):
            self.position.close()
            self.sell(size=size, sl=price + stop_distance, tp=price - self.tp_atr_mult * a)


class _RiskSizedStrategy(Strategy):
    """Base helper: position sizing by fixed-fractional risk off the stop distance.

    Never sizes by raw leverage on full equity. The number of units is chosen so
    that hitting the protective stop loses approximately risk_pct of current equity.
    """

    risk_pct = 0.01

    def _size_for_risk(self, stop_distance):
        if stop_distance <= 0:
            return 0
        risk_amount = self.equity * self.risk_pct
        units = int(risk_amount / stop_distance)
        return max(units, 0)


class BollingerMeanReversion(_RiskSizedStrategy):
    """Mean reversion: fade stretched moves back toward a moving-average mean.

    Rationale: daily FX majors spend most of their time range-bound, so price
    extended far from its mean tends to snap back. We enter when price closes
    beyond a Bollinger band, target the middle band (the mean), and cap risk with
    an ATR stop. Expect a HIGH-ish win rate but small winners; the tail risk is a
    trend running against the position, which the ATR stop is there to contain.
    Win rate is NOT the objective -- profit factor and drawdown are.
    """

    bb_period = 20
    bb_k = 2.0
    atr_period = 14
    sl_atr_mult = 2.0
    risk_pct = 0.01
    max_hold = 20  # bars; time-stop so we don't marry a losing reversion

    def init(self):
        close = self.data.Close
        self.mid = self.I(sma, close, self.bb_period)
        self.sd = self.I(rolling_std, close, self.bb_period)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self._bars_held = 0

    def next(self):
        price = self.data.Close[-1]
        mid = self.mid[-1]
        sd = self.sd[-1]
        a = self.atr[-1]
        if mid != mid or sd != sd or a != a or sd == 0:
            return

        upper = mid + self.bb_k * sd
        lower = mid - self.bb_k * sd
        stop_distance = self.sl_atr_mult * a

        if self.position:
            self._bars_held += 1
            # Exit at the mean (take profit) or on time stop. ATR stop is on the order.
            if self.position.is_long and price >= mid:
                self.position.close()
            elif self.position.is_short and price <= mid:
                self.position.close()
            elif self._bars_held >= self.max_hold:
                self.position.close()
            return

        size = self._size_for_risk(stop_distance)
        if size <= 0:
            return

        # Fade extremes: long when stretched below lower band, short above upper.
        if price < lower:
            self._bars_held = 0
            self.buy(size=size, sl=price - stop_distance)
        elif price > upper:
            self._bars_held = 0
            self.sell(size=size, sl=price + stop_distance)


class BollingerMeanReversionTrendFilter(BollingerMeanReversion):
    """Mean reversion, but only fade in the direction of the longer-term trend.

    Adds a slow EMA regime filter: only take longs when price is above the slow
    EMA, only shorts when below. The idea is to avoid catching a falling knife in
    a strong trend -- fade pullbacks within a trend rather than fading the trend
    itself.
    """

    bb_k = 1.5  # a 1.5-std pullback within a trend is a more natural dip-buy level
    trend_period = 100

    def init(self):
        super().init()
        self.trend = self.I(ema, self.data.Close, self.trend_period)

    def next(self):
        price = self.data.Close[-1]
        mid = self.mid[-1]
        sd = self.sd[-1]
        a = self.atr[-1]
        t = self.trend[-1]
        if mid != mid or sd != sd or a != a or t != t or sd == 0:
            return

        upper = mid + self.bb_k * sd
        lower = mid - self.bb_k * sd
        stop_distance = self.sl_atr_mult * a

        if self.position:
            self._bars_held += 1
            if self.position.is_long and price >= mid:
                self.position.close()
            elif self.position.is_short and price <= mid:
                self.position.close()
            elif self._bars_held >= self.max_hold:
                self.position.close()
            return

        size = self._size_for_risk(stop_distance)
        if size <= 0:
            return

        if price < lower and price > t:  # dip within an uptrend
            self._bars_held = 0
            self.buy(size=size, sl=price - stop_distance)
        elif price > upper and price < t:  # pop within a downtrend
            self._bars_held = 0
            self.sell(size=size, sl=price + stop_distance)


class GoldScalperPro(_RiskSizedStrategy):
    """Python port of a user-supplied MQL5 EA (GoldScalperPro.mq5) for XAUUSD
    scalping on M5, with an H1 trend filter, RSI gate, ATR-based SL/TP,
    break-even + trailing stop, a London/NY session filter, and a daily
    drawdown kill-switch.

    Ported AS-IS, including a real bug in the EA's "Supertrend": genuine
    Supertrend carries a trailing band forward and only flips when price
    closes through the *active* band. This EA instead recomputes both bands
    fresh every bar from the H/L midpoint +/- multiplier*ATR of one bar back,
    with no persistence -- so it's actually a 1-bar-lagged ATR breakout test,
    not Supertrend, and will fire far more often / on more noise than real
    Supertrend would. Replicated faithfully rather than fixed, since the goal
    is to test what this EA actually does, not a corrected version of it.

    Two more deviations from the EA, both forced by what a single-timeframe,
    OHLC-bar backtest can represent -- not bugs, but worth knowing about:
      - No spread simulation. This project prices transaction costs via the
        `commission` param on Backtest, like every other strategy here, not
        a per-instrument spread -- so the EA's 30-point spread guard has no
        equivalent and is simply not applied.
      - Session-filter hours are matched against the data's UTC timestamps,
        not an MT5 broker's "server time" (commonly UTC+2/+3) -- treat
        session-filtered results as approximate, not exact.

    Needs an "HtfEmaPrev" column on the input data (see
    data.load_oanda_intraday_with_htf_trend) -- the H1 EMA(50) value as of
    the last fully-completed H1 bar before each row's timestamp. Comparing
    intrabar price to that value is algebraically equivalent to the EA's
    live "is the current, still-forming H1 EMA above the prior one"
    check (the EMA recursion reduces to exactly that comparison).
    """

    risk_pct = 0.01
    max_positions = 2
    rsi_period = 14
    rsi_buy_level = 55.0
    rsi_sell_level = 45.0
    st_period = 10  # kept for fidelity with the EA's input name; unused, see class docstring
    st_multiplier = 3.0
    atr_period = 14
    sl_atr_mult = 1.5
    tp_atr_mult = 2.5
    be_atr_mult = 1.0
    trail_atr_mult = 0.5
    min_atr = 0.50
    max_daily_dd_pct = 3.0
    london_open, london_close = 7, 12
    ny_open, ny_close = 13, 18

    def init(self):
        self.rsi = self.I(rsi, self.data.Close, self.rsi_period)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.htf_ema_prev = self.I(lambda x: pd.Series(x), self.data.HtfEmaPrev)
        self._day = None
        self._day_start_equity = None

    def _manage_open_trades(self, price, a):
        for trade in self.trades:
            if trade.is_long:
                be_level = trade.entry_price + a * self.be_atr_mult
                trail_level = price - a * self.trail_atr_mult
                if price >= be_level:
                    new_sl = max(trade.entry_price, trail_level)
                    if trade.sl is None or new_sl > trade.sl:
                        trade.sl = new_sl
            else:
                be_level = trade.entry_price - a * self.be_atr_mult
                trail_level = price + a * self.trail_atr_mult
                if price <= be_level:
                    new_sl = min(trade.entry_price, trail_level)
                    if trade.sl is None or new_sl < trade.sl:
                        trade.sl = new_sl

    def next(self):
        if len(self.data) < 3:
            return
        price = self.data.Close[-1]
        a = self.atr[-1]
        r = self.rsi[-1]
        htf = self.htf_ema_prev[-1]
        if a != a or r != r or htf != htf:
            return

        bar_date = self.data.index[-1].date()
        if self._day != bar_date:
            self._day = bar_date
            self._day_start_equity = self.equity
        dd_pct = (self._day_start_equity - self.equity) / self._day_start_equity * 100.0

        # Break-even + trailing always runs, regardless of the gates below --
        # matches the EA calling ManageOpenPositions() before any entry gate.
        self._manage_open_trades(price, a)

        if dd_pct >= self.max_daily_dd_pct:
            return
        hour = self.data.index[-1].hour
        in_session = (self.london_open <= hour < self.london_close) or (self.ny_open <= hour < self.ny_close)
        if not in_session:
            return
        if len(self.trades) >= self.max_positions:
            return
        if a < self.min_atr:
            return

        # The EA's actual "Supertrend": close[-1] vs a band built from bar[-2]'s H/L mid +/- mult*ATR.
        mid_prev = (self.data.High[-2] + self.data.Low[-2]) / 2.0
        atr_prev = self.atr[-2]
        if atr_prev != atr_prev:
            return
        upper_prev = mid_prev + self.st_multiplier * atr_prev
        lower_prev = mid_prev - self.st_multiplier * atr_prev
        st_bull = price > upper_prev
        st_bear = price < lower_prev
        st_signal = 1 if (st_bull and not st_bear) else (-1 if (st_bear and not st_bull) else 0)
        if st_signal == 0:
            return

        htf_bull = price > htf
        htf_bear = price < htf
        rsi_bull = r > self.rsi_buy_level
        rsi_bear = r < self.rsi_sell_level

        stop_distance = self.sl_atr_mult * a
        size = self._size_for_risk(stop_distance)
        if size <= 0:
            return

        if htf_bull and rsi_bull and st_signal == 1:
            self.buy(size=size, sl=price - stop_distance, tp=price + self.tp_atr_mult * a)
        elif htf_bear and rsi_bear and st_signal == -1:
            self.sell(size=size, sl=price + stop_distance, tp=price - self.tp_atr_mult * a)


class DonchianBreakout(_RiskSizedStrategy):
    """Trend-following breakout: enter on N-day channel breakouts, ride with an
    ATR trailing stop.

    Rationale: when range-bound markets do break, the move can run. This is the
    opposite archetype to the mean-reversion models, included so we can see which
    edge (if any) actually holds up across instruments rather than assuming one.
    Expect a LOW win rate; profitability depends entirely on the trailing stop
    letting winners run far past the average loser.
    """

    entry_period = 20
    exit_period = 10
    atr_period = 14
    sl_atr_mult = 2.0
    trail_atr_mult = 3.0
    risk_pct = 0.01

    def init(self):
        self.upper = self.I(_highest, self.data.High, self.entry_period)
        self.lower = self.I(_lowest, self.data.Low, self.entry_period)
        self.exit_high = self.I(_highest, self.data.High, self.exit_period)
        self.exit_low = self.I(_lowest, self.data.Low, self.exit_period)
        self.atr = self.I(atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self._trail = None

    def next(self):
        price = self.data.Close[-1]
        a = self.atr[-1]
        if a != a or self.upper[-1] != self.upper[-1]:
            return

        # Use prior-bar channel levels to avoid look-ahead on the breakout bar.
        upper = self.upper[-2]
        lower = self.lower[-2]
        stop_distance = self.sl_atr_mult * a

        if self.position:
            if self.position.is_long:
                self._trail = max(self._trail, price - self.trail_atr_mult * a)
                if price <= self._trail or price <= self.exit_low[-2]:
                    self.position.close()
            elif self.position.is_short:
                self._trail = min(self._trail, price + self.trail_atr_mult * a)
                if price >= self._trail or price >= self.exit_high[-2]:
                    self.position.close()
            return

        size = self._size_for_risk(stop_distance)
        if size <= 0:
            return

        if price > upper:
            self._trail = price - self.trail_atr_mult * a
            self.buy(size=size, sl=price - stop_distance)
        elif price < lower:
            self._trail = price + self.trail_atr_mult * a
            self.sell(size=size, sl=price + stop_distance)
