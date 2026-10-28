"""SQLite persistence for the big-model gate (local Rodaddy patch).

Two jobs, nothing else:

- `service_time` survives restarts, so a daemon bounce no longer resets the
  measured limits back to the over-generous fallback (the exact moment an
  abuser benefits most).
- `events` is the operational record: one row per gated request or rejection
  (when, who, which model, what happened, how long). Read it with sqlite3
  when deciding policy; nothing here interprets it.

Default path ~/ai-serving/omlx-gate.db (override: OMLX_GATE_DB; empty
string disables persistence and everything runs in memory).
"""

from __future__ import annotations

import logging
import os
import sqlite3

from omlx_gate.logging import logged

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS service_time (
    model TEXT PRIMARY KEY,
    ewma REAL NOT NULL,
    samples INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS events (
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    client TEXT NOT NULL,
    model TEXT NOT NULL,
    outcome TEXT NOT NULL,
    status INTEGER NOT NULL,
    elapsed REAL
);
"""

_RETAIN_DAYS = 14


class GateStore:
    """Tiny synchronous store; every write is one small WAL transaction."""

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            path = os.environ.get(
                "OMLX_GATE_DB", os.path.expanduser("~/ai-serving/omlx-gate.db")
            )
        self._conn = sqlite3.connect(path or ":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        pruned = self._conn.execute(
            "DELETE FROM events WHERE ts < datetime('now', ?)",
            (f"-{_RETAIN_DAYS} days",),
        ).rowcount
        self._conn.commit()
        logger.info(
            "GATE: store open path=%s pruned=%d old events", path or ":memory:", pruned
        )

    @logged
    def load_service_times(self) -> dict[str, tuple[float, int]]:
        rows = self._conn.execute("SELECT model, ewma, samples FROM service_time")
        return {model: (ewma, samples) for model, ewma, samples in rows}

    def save_service_time(self, model: str, ewma: float, samples: int) -> None:
        self._conn.execute(
            "INSERT INTO service_time (model, ewma, samples) VALUES (?, ?, ?)"
            " ON CONFLICT(model) DO UPDATE SET ewma=excluded.ewma,"
            " samples=excluded.samples, updated_at=datetime('now')",
            (model, ewma, samples),
        )
        self._conn.commit()

    def record_event(
        self,
        client: str,
        model: str,
        outcome: str,
        status: int,
        elapsed: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO events (client, model, outcome, status, elapsed)"
            " VALUES (?, ?, ?, ?, ?)",
            (client, model, outcome, status, elapsed),
        )
        self._conn.commit()
