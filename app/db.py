import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models import Chunk, DocumentOut, DocumentStatus, MessageOut, SourceOut

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT UNIQUE NOT NULL,
    pages INTEGER,
    chunks INTEGER,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id, idx);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_doc ON messages(doc_id, id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_document(self, doc_id: str, filename: str, sha256: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, sha256, status, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, filename, sha256, DocumentStatus.PROCESSING, _now()),
            )

    def update_document(self, doc_id: str, **fields: Any) -> None:
        allowed = {"pages", "chunks", "status", "progress", "error"}
        if not fields.keys() <= allowed:
            raise ValueError(f"Unknown fields: {fields.keys() - allowed}")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE documents SET {assignments} WHERE id = ?", (*fields.values(), doc_id)
            )

    def get_document(self, doc_id: str) -> DocumentOut | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _document(row) if row else None

    def find_by_hash(self, sha256: str) -> DocumentOut | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return _document(row) if row else None

    def list_documents(self) -> list[DocumentOut]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [_document(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def fail_interrupted(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE documents SET status = ?, error = ? WHERE status = ?",
                (
                    DocumentStatus.ERROR,
                    "Индексация была прервана перезапуском сервера. Загрузите файл снова.",
                    DocumentStatus.PROCESSING,
                ),
            )
            return cur.rowcount

    def save_chunks(self, chunks: list[Chunk]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chunks (id, doc_id, idx, text, page_start, page_end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(c.id, c.doc_id, c.index, c.text, c.page_start, c.page_end) for c in chunks],
            )

    def delete_chunks(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def list_chunks(self, doc_id: str) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY idx", (doc_id,)
            ).fetchall()
        return [_chunk(r) for r in rows]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
            ).fetchall()
        return {r["id"]: _chunk(r) for r in rows}

    def add_message(
        self, doc_id: str, role: str, content: str, sources: list[SourceOut] | None = None
    ) -> int:
        sources_json = (
            json.dumps([s.model_dump() for s in sources], ensure_ascii=False) if sources else None
        )
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (doc_id, role, content, sources_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, role, content, sources_json, _now()),
            )
            return int(cur.lastrowid or 0)

    def list_messages(self, doc_id: str, limit: int | None = None) -> list[MessageOut]:
        query = "SELECT * FROM messages WHERE doc_id = ? ORDER BY id DESC"
        params: tuple[Any, ...] = (doc_id,)
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_message(r) for r in reversed(rows)]

    def clear_messages(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE doc_id = ?", (doc_id,))


def _document(row: sqlite3.Row) -> DocumentOut:
    return DocumentOut(**{k: row[k] for k in DocumentOut.model_fields})


def _chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        index=row["idx"],
        text=row["text"],
        page_start=row["page_start"],
        page_end=row["page_end"],
    )


def _message(row: sqlite3.Row) -> MessageOut:
    sources = json.loads(row["sources_json"]) if row["sources_json"] else []
    return MessageOut(
        id=row["id"],
        role=row["role"],
        content=row["content"],
        sources=[SourceOut(**s) for s in sources],
        created_at=row["created_at"],
    )
