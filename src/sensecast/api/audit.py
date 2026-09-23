"""Append-only audit log of human decisions on suggested orders (SQLite).

UPDATE and DELETE are blocked by triggers, so the log cannot be edited through the
application. Swap the connection string for Postgres in production.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    user_name     TEXT    NOT NULL,
    user_role     TEXT    NOT NULL,
    store_id      TEXT    NOT NULL,
    item_id       TEXT    NOT NULL,
    order_date    TEXT    NOT NULL,
    action        TEXT    NOT NULL CHECK (action IN ('approve', 'override', 'reject')),
    suggested_qty INTEGER NOT NULL,
    final_qty     INTEGER NOT NULL CHECK (final_qty >= 0),
    reason_code   TEXT,
    note          TEXT,
    model_version TEXT,
    request_id    TEXT
);
CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE INDEX IF NOT EXISTS ix_decisions_store ON decisions(store_id, order_date);
"""

COLUMNS = ["id", "ts", "user_name", "user_role", "store_id", "item_id", "order_date", "action",
           "suggested_qty", "final_qty", "reason_code", "note", "model_version", "request_id"]


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, **fields) -> int:
        fields["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
        cols = [c for c in COLUMNS if c != "id"]
        with self._lock, self._conn() as c:
            cur = c.execute(
                f"INSERT INTO decisions ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                [fields.get(k) for k in cols],
            )
            return int(cur.lastrowid)

    def query(self, store_id: str | None = None, limit: int = 200) -> list[dict]:
        sql = f"SELECT {', '.join(COLUMNS)} FROM decisions"
        args: list = []
        if store_id:
            sql += " WHERE store_id = ?"
            args.append(store_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            return [dict(zip(COLUMNS, row, strict=False)) for row in c.execute(sql, args).fetchall()]

    def latest_by_item(self, store_id: str | None, order_date: str) -> dict[tuple[str, str], dict]:
        rows = self.query(store_id, limit=100_000)
        out = {}
        for r in reversed(rows):  # oldest -> newest, newest wins
            if r["order_date"] == order_date:
                out[(r["store_id"], r["item_id"])] = r
        return out
