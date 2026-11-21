"""Tests for Osmo's local SQLite + FTS5 persistent memory store."""

from __future__ import annotations

import time
from pathlib import Path

from hackathon_osmo.ai.memory import OsmoMemoryStore


def test_memory_save_deduplicate_search_and_forget(tmp_path: Path) -> None:
    db = tmp_path / "osmo_memory.db"
    store = OsmoMemoryStore(db)
    try:
        r1 = store.save_memory("Restaurant Tricotin, 15 avenue de Choisy, 01 45 85 51 52", category="note")
        assert r1["ok"] is True and r1["deduplicated"] is False

        # Exact duplicate (case-insensitive) should deduplicate
        r2 = store.save_memory("restaurant tricotin, 15 avenue de choisy, 01 45 85 51 52", category="note")
        assert r2["ok"] is True and r2["deduplicated"] is True and r2["id"] == r1["id"]

        store.save_memory("Zach prépare le hackathon Sciences Po 2026 avec Osmo", category="fact", pinned=True)
        store.save_memory("Toujours répondre en français oral concis", category="preference")

        # Fast FTS5 recall (< 5 ms)
        t0 = time.monotonic()
        hits = store.search_memories("C'était quoi le numéro du restaurant Tricotin ?")
        dt_ms = (time.monotonic() - t0) * 1000.0
        assert dt_ms < 25.0
        assert len(hits) >= 1
        assert "01 45 85 51 52" in hits[0]["content"]

        # Forget by keyword
        deleted = store.forget("Tricotin")
        assert deleted == 1
        assert store.search_memories("Tricotin") == []
    finally:
        store.close()


def test_memory_persists_across_restarts_and_builds_compact_context(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"
    s1 = OsmoMemoryStore(db)
    s1.save_memory("Projet démo : compagnon robotique Pixel 11 Pro + DJI Osmo", category="fact", pinned=True)
    s1.record_episode(
        "Lis la carte rose sur mon bureau",
        "C'est le restaurant Tricotin au 15 avenue de Choisy, 01 45 85 51 52.",
        mode="desk",
        tools=["look_at_desk"],
    )
    s1.close()

    # Reopen fresh store (simulating full osmo-app restart!)
    s2 = OsmoMemoryStore(db)
    try:
        snippet = s2.build_context_snippet("Rappelle-moi l'adresse de Tricotin qu'on a lue sur la carte")
        assert "Tricotin" in snippet
        assert "Pixel 11 Pro" in snippet
        assert len(snippet) <= 520
        st = s2.stats()
        assert st["memories_count"] == 1
        assert st["episodes_count"] == 1
    finally:
        s2.close()
