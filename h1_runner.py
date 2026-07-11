"""H1 sleeve runner -- fires every hour to check EMA crossover signals."""
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import broker
from live_runner import (
    H1_SLEEVES, MACD_H1_SLEEVES, PDHL_SLEEVES, ALLOCATION_FRACTION,
    _log, run_h1_sleeve, run_macd_h1_sleeve, run_pdhl_sleeve,
)

_AUTH_RETRY_DELAY = 180   # seconds between retries during OANDA maintenance window
_AUTH_MAX_RETRIES = 3


def _run_with_retry(fn, tag, *args):
    """Run a sleeve function, retrying on OANDA auth errors (daily maintenance window)."""
    for attempt in range(1, _AUTH_MAX_RETRIES + 1):
        try:
            fn(*args)
            return
        except Exception as e:
            msg = str(e)
            if "authorization" in msg.lower() or "Insufficient authorization" in msg:
                if attempt < _AUTH_MAX_RETRIES:
                    _log(f"{tag}: auth error (OANDA maintenance?), retry {attempt}/{_AUTH_MAX_RETRIES - 1} in {_AUTH_RETRY_DELAY}s -- {e}")
                    time.sleep(_AUTH_RETRY_DELAY)
                else:
                    _log(f"{tag}: auth error after {_AUTH_MAX_RETRIES} attempts, giving up -- {e}")
            else:
                _log(f"{tag}: ERROR -- {e}")
                return


def main() -> None:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        _log("[h1] weekend -- markets closed, skipping")
        return

    b = broker.from_env()
    account_equity = float(b.account_summary()["account"]["balance"])
    sleeve_equity = account_equity * ALLOCATION_FRACTION
    _log(f"[h1] equity={account_equity:.2f} per-sleeve={sleeve_equity:.2f}")
    for tag, instrument in H1_SLEEVES:
        _run_with_retry(run_h1_sleeve, tag, b, tag, instrument, sleeve_equity)
    for tag, instrument in MACD_H1_SLEEVES:
        _run_with_retry(run_macd_h1_sleeve, tag, b, tag, instrument, sleeve_equity)
    for tag, instrument in PDHL_SLEEVES:
        _run_with_retry(run_pdhl_sleeve, tag, b, tag, instrument, sleeve_equity)


if __name__ == "__main__":
    main()
