"""
Tests for Pengu memory system.
"""
import os
import tempfile

import pytest

from pengu.memory import (
    MemoryCategory,
    MemoryEntry,
    MemoryManager,
    MemoryType,
)


@pytest.fixture
async def memory():
    """Create a fresh memory manager with temp DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    mgr = MemoryManager(db_path=db_path)
    await mgr.initialize()
    yield mgr
    await mgr.close()
    os.unlink(db_path)


class TestMemoryManager:
    async def test_save_persistent(self, memory):
        entry = await memory.save(
            content="User prefers dark mode",
            category=MemoryCategory.PREFERENCE,
            memory_type=MemoryType.PERSISTENT,
        )
        assert entry.id.startswith("mem_")
        assert entry.content == "User prefers dark mode"
        assert entry.category == MemoryCategory.PREFERENCE

    async def test_save_session(self, memory):
        entry = await memory.save(
            content="Working on project X",
            category=MemoryCategory.TASK,
            memory_type=MemoryType.SESSION,
        )
        assert entry.id.startswith("mem_")
        assert len(memory._session_memory) == 1

    async def test_save_empty_rejected(self, memory):
        with pytest.raises(ValueError, match="empty"):
            await memory.save(content="", category=MemoryCategory.GENERAL)

    async def test_save_sensitive_rejected(self, memory):
        with pytest.raises(ValueError, match="sensitive"):
            await memory.save(
                content="my password is abc123",
                category=MemoryCategory.GENERAL,
            )

    async def test_search_persistent(self, memory):
        await memory.save(content="dark mode preference", category=MemoryCategory.PREFERENCE)
        await memory.save(content="light theme", category=MemoryCategory.PREFERENCE)
        results = await memory.search("dark", memory_type=MemoryType.PERSISTENT)
        assert len(results) == 1
        assert "dark mode" in results[0].content

    async def test_search_session(self, memory):
        await memory.save(
            content="Working on FoveaEdge",
            memory_type=MemoryType.SESSION,
        )
        results = await memory.search("FoveaEdge", memory_type=MemoryType.SESSION)
        assert len(results) == 1

    async def test_get_by_id(self, memory):
        entry = await memory.save(content="test entry", memory_type=MemoryType.PERSISTENT)
        fetched = await memory.get(entry.id)
        assert fetched is not None
        assert fetched.content == "test entry"

    async def test_delete(self, memory):
        entry = await memory.save(content="to delete", memory_type=MemoryType.PERSISTENT)
        deleted = await memory.delete(entry.id)
        assert deleted is True
        fetched = await memory.get(entry.id)
        assert fetched is None

    async def test_clear(self, memory):
        await memory.save(content="item 1", memory_type=MemoryType.PERSISTENT)
        await memory.save(content="item 2", memory_type=MemoryType.SESSION)
        count = await memory.clear()
        assert count >= 2

    async def test_stats(self, memory):
        stats = memory.get_stats()
        assert stats["initialized"] is True
        assert stats["session_count"] == 0


class TestMemoryPrivacy:
    async def test_reject_api_key(self, memory):
        with pytest.raises(ValueError, match="sensitive"):
            await memory.save(content="API key: sk-12345")

    async def test_reject_password(self, memory):
        with pytest.raises(ValueError, match="sensitive"):
            await memory.save(content="password is secret123")

    async def test_reject_token(self, memory):
        with pytest.raises(ValueError, match="sensitive"):
            await memory.save(content="auth_token: abc")

    async def test_reject_ssh_key(self, memory):
        with pytest.raises(ValueError, match="sensitive"):
            await memory.save(content="ssh_key: asdf")
