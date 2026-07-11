"""
FVG session runner — fires every 30 min during London and NY opens.
Runs only the FVG_M30_SLEEVES, leaving the main bot untouched.

Cron to add (runs Mon-Fri at :00 and :30 of hours 7,8,9,13,14,15 UTC):
  */30 7-9,13-15 * * 1-5 cd /Users/bamznizzy/forex-bot && uv run python3 fvg_session_runner.py >> /Users/bamznizzy/forex-bot/logs/fvg.log 2>&1
"""
from dotenv import load_dotenv; load_dotenv()
import broker
from live_runner import (
    FVG_M30_SLEEVES,
    ALLOCATION_FRACTION,
    run_fvg_m30_sleeve,
    _log,
)


def main() -> None:
    b = broker.from_env()
    account_equity = float(b.account_summary()["account"]["balance"])
    sleeve_equity  = account_equity * ALLOCATION_FRACTION
    _log(f"FVG session run: equity={account_equity:.2f} sleeve={sleeve_equity:.2f}")
    for tag, instrument in FVG_M30_SLEEVES:
        try:
            run_fvg_m30_sleeve(b, tag, instrument, sleeve_equity)
        except Exception as e:
            _log(f"{tag}: ERROR -- {e}")


if __name__ == "__main__":
    main()
