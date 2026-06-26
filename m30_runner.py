"""M30 sleeve runner -- fires every 30 minutes to check BbmrtM30 entry/exit signals.

The daily sleeves in live_runner.py run once at 22:30 BST after the daily candle
closes. This script runs the three M30 sleeves independently at 30-minute intervals
so RSI crossover signals are caught when they actually occur, not just once per day.
"""

from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

import broker  # noqa: E402
from live_runner import ALLOCATION_FRACTION, M30_SLEEVES, _log, run_bbmrt_m30_sleeve  # noqa: E402


def main() -> None:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        _log("[m30] weekend -- markets closed, skipping")
        return

    b = broker.from_env()
    account_equity = float(b.account_summary()["account"]["balance"])
    sleeve_equity = account_equity * ALLOCATION_FRACTION
    _log(f"[m30] equity={account_equity:.2f} per-sleeve={sleeve_equity:.2f}")
    for tag, instrument in M30_SLEEVES:
        try:
            run_bbmrt_m30_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")


if __name__ == "__main__":
    main()
