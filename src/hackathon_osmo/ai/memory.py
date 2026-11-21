"""Ultra-fast local SQLite + FTS5 persistent memory for Osmo (< 1 ms queries, zero cloud dependency).

Stores:
1. `memories`: durable facts, user preferences, tasks, and key notes read from desk sheets.
2. `episodes`: compact log of past conversations & desk readings for instant recall across restarts.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hackathon_osmo.paths import MEMORY_DB_FILE

VALID_CATEGORIES = ("preference", "fact", "note", "task")
_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "en", "au", "aux",
    "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "que", "qui", "quoi", "est", "sont",
    "c", "s", "n", "ne", "pas", "plus", "pour", "par", "sur", "dans", "avec", "sans", "tout",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for", "with", "and",
    "my", "your", "me", "you", "it", "this", "that", "what", "where", "how", "can", "osmo",
}


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _extract_fts_terms(text: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]{2,}", text.lower())
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= 8:
            break
    return out


class OsmoMemoryStore:
    """Thread-safe local SQLite + FTS5 memory store for Osmo."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else MEMORY_DB_FILE
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous = NORMAL;

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'voice',
                    pinned INTEGER NOT NULL DEFAULT 0
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    category,
                    content='memories',
                    content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, category)
                    VALUES (new.id, new.content, new.category);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, category)
                    VALUES ('delete', old.id, old.content, old.category);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, category)
                    VALUES ('delete', old.id, old.content, old.category);
                    INSERT INTO memories_fts(rowid, content, category)
                    VALUES (new.id, new.content, new.category);
                END;

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'face',
                    user_text TEXT NOT NULL,
                    reply_text TEXT NOT NULL,
                    tools_used TEXT NOT NULL DEFAULT ''
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    user_text,
                    reply_text,
                    content='episodes',
                    content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                    INSERT INTO episodes_fts(rowid, user_text, reply_text)
                    VALUES (new.id, new.user_text, new.reply_text);
                END;
                CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                    INSERT INTO episodes_fts(episodes_fts, rowid, user_text, reply_text)
                    VALUES ('delete', old.id, old.user_text, old.reply_text);
                END;
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def save_memory(
        self,
        content: str,
        *,
        category: str = "fact",
        source: str = "voice",
        pinned: bool = False,
    ) -> dict[str, Any]:
        clean = " ".join(str(content or "").split()).strip()
        if not clean:
            return {"ok": False, "error": "Contenu vide"}
        cat = category if category in VALID_CATEGORIES else "fact"
        now = _iso_now()

        with self._lock:
            # Avoid exact duplicates (case-insensitive)
            existing = self._conn.execute(
                "SELECT id, category, content, pinned FROM memories WHERE lower(content) = lower(?) LIMIT 1",
                (clean,),
            ).fetchone()
            if existing is not None:
                self._conn.execute(
                    "UPDATE memories SET updated_at = ?, category = ?, pinned = MAX(pinned, ?) WHERE id = ?",
                    (now, cat, int(pinned), existing["id"]),
                )
                self._conn.commit()
                return {
                    "ok": True,
                    "id": int(existing["id"]),
                    "category": cat,
                    "content": clean,
                    "deduplicated": True,
                }

            cur = self._conn.execute(
                "INSERT INTO memories(created_at, updated_at, category, content, source, pinned) VALUES (?, ?, ?, ?, ?, ?)",
                (now, now, cat, clean, source, int(pinned)),
            )
            self._conn.commit()
            return {
                "ok": True,
                "id": int(cur.lastrowid),
                "category": cat,
                "content": clean,
                "deduplicated": False,
            }

    def delete_memory(self, mem_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (int(mem_id),))
            self._conn.commit()
            return cur.rowcount > 0

    def forget(self, query: str) -> int:
        """Delete matching memories by numeric ID or keyword search."""
        q = str(query or "").strip()
        if not q:
            return 0
        if q.isdigit():
            return 1 if self.delete_memory(int(q)) else 0
        matches = self.search_memories(q, limit=5)
        removed = 0
        with self._lock:
            for m in matches:
                cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (int(m["id"]),))
                removed += cur.rowcount
            self._conn.commit()
        return removed

    def record_episode(
        self,
        user_text: str,
        reply_text: str,
        *,
        mode: str = "face",
        tools: list[str] | None = None,
    ) -> int:
        u = " ".join(str(user_text or "").split()).strip()
        r = " ".join(str(reply_text or "").split()).strip()
        if not u and not r:
            return 0
        tools_str = ", ".join(tools or [])
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO episodes(created_at, mode, user_text, reply_text, tools_used) VALUES (?, ?, ?, ?, ?)",
                (_iso_now(), mode, u[:500], r[:700], tools_str[:250]),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def search_memories(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        terms = _extract_fts_terms(query)
        if not terms:
            return []
        fts_expr = " OR ".join(f'"{t}"*' for t in terms)
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT m.id, m.category, m.content, m.updated_at, m.pinned
                    FROM memories_fts f
                    JOIN memories m ON m.id = f.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25(memories_fts), m.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_expr, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        return [dict(r) for r in rows]

    def search_episodes(self, query: str, *, limit: int = 3) -> list[dict[str, Any]]:
        terms = _extract_fts_terms(query)
        if not terms:
            return []
        fts_expr = " OR ".join(f'"{t}"*' for t in terms)
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT e.id, e.created_at, e.mode, e.user_text, e.reply_text
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.rowid
                    WHERE episodes_fts MATCH ?
                    ORDER BY bm25(episodes_fts), e.id DESC
                    LIMIT ?
                    """,
                    (fts_expr, int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        return [dict(r) for r in rows]

    def list_memories(self, *, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, created_at, updated_at, category, content, source, pinned
                FROM memories
                ORDER BY pinned DESC, updated_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def build_context_snippet(
        self,
        user_text: str,
        *,
        max_memories: int = 5,
        max_episodes: int = 2,
        max_chars: int = 520,
    ) -> str:
        """Build an ultra-compact (<120 tokens, <1 ms) memory block for the Vision Brain prompt."""
        t0 = time.monotonic()
        chosen_mem: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        # 1. Relevant FTS5 matches on current user utterance first
        for m in self.search_memories(user_text, limit=3):
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                chosen_mem.append(m)

        # 2. Fill remaining slots with pinned & most recent durable memories
        if len(chosen_mem) < max_memories:
            for m in self.list_memories(limit=max_memories):
                if m["id"] not in seen_ids:
                    seen_ids.add(m["id"])
                    chosen_mem.append(m)
                if len(chosen_mem) >= max_memories:
                    break

        parts: list[str] = []
        for m in chosen_mem:
            parts.append(f"[{m['category']}] {m['content']}")

        # 3. Include relevant past episodes (e.g. a card or notebook read 20 min ago or yesterday!)
        if user_text.strip():
            for ep in self.search_episodes(user_text, limit=max_episodes):
                reply = str(ep.get("reply_text") or "").strip()
                if reply:
                    short = reply if len(reply) <= 140 else reply[:137] + "…"
                    parts.append(f"[passé/{ep['mode']}] {short}")

        if not parts:
            return ""
        joined = " | ".join(parts)
        _ = (time.monotonic() - t0) * 1000.0
        return joined[:max_chars]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            m_count = int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
            e_count = int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        return {
            "memories_count": m_count,
            "episodes_count": e_count,
            "db_path": str(self.db_path),
        }
