"""H1 sleeve runner -- fires every hour to check EMA crossover signals."""
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import broker
from live_runner import (
    H1_SLEEVES, MACD_H1_SLEEVES, PDHL_SLEEVES, ALLOCATION_FRACTION,
    _log, run_h1_sleeve, run_macd_h1_sleeve, run_pdhl_sleeve,
)


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
        try:
            run_h1_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in MACD_H1_SLEEVES:
        try:
            run_macd_h1_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")
    for tag, instrument in PDHL_SLEEVES:
        try:
            run_pdhl_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")


if __name__ == "__main__":
    main()
