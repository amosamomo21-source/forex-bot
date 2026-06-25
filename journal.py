"""Append-only trade journal for live_runner.py.

Records each fill (open) and close that live_runner.py's sleeves produce,
tagged by sleeve name -- independent of OANDA's own transaction history, so
per-sleeve performance (win rate, realized P/L) can be reviewed directly
without reconstructing it from raw OANDA transactions.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

JOURNAL_PATH = Path(__file__).parent / "trade_journal.csv"
FIELDS = [
    "timestamp",
    "tag",
    "instrument",
    "event",
    "direction",
    "units",
    "price",
    "sl",
    "tp",
    "realized_pl",
    "reason",
]


def _append(row: dict) -> None:
    is_new = not JOURNAL_PATH.exists()
    with JOURNAL_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def record_open(
    tag: str,
    instrument: str,
    direction: str,
    units: int,
    price: float,
    sl: float | None = None,
    tp: float | None = None,
) -> None:
    _append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tag": tag,
            "instrument": instrument,
            "event": "open",
            "direction": direction,
            "units": units,
            "price": price,
            "sl": sl if sl is not None else "",
            "tp": tp if tp is not None else "",
        }
    )


def record_close(tag: str, instrument: str, reason: str, realized_pl: float | None) -> None:
    _append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tag": tag,
            "instrument": instrument,
            "event": "close",
            "realized_pl": realized_pl if realized_pl is not None else "",
            "reason": reason,
        }
    )
