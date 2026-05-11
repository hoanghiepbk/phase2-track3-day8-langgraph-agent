"""Checkpointer factory for LangGraph state persistence.

Supports:
- ``none``     no checkpointer (one-shot run)
- ``memory``   in-process MemorySaver (default, no infrastructure)
- ``sqlite``   on-disk SqliteSaver with WAL mode (survives process restart)
- ``postgres`` PostgresSaver (requires extra dependency)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_SQLITE_PATH = "checkpoints.db"


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer for the requested backend."""
    kind = (kind or "memory").lower()

    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover - install-time error
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = Path(database_url or DEFAULT_SQLITE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import (  # type: ignore[import-not-found]
                PostgresSaver,
            )
        except ImportError as exc:  # pragma: no cover - install-time error
            raise RuntimeError(
                "Postgres checkpointer requires: pip install langgraph-checkpoint-postgres"
            ) from exc
        if not database_url:
            raise ValueError("postgres checkpointer requires database_url")
        return PostgresSaver.from_conn_string(database_url)

    raise ValueError(f"Unknown checkpointer kind: {kind!r}")
