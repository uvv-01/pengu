"""
Memory system — SQLite-backed session and long-term memory.

Provides:
  - Session memory: temporary, cleared on restart
  - Long-term memory: persistent across restarts

Categories:
  - preference: user preferences
  - project: project context
  - task: task context
  - reminder: explicit reminders
  - summary: conversation/task summaries

Privacy:
  - No passwords, API keys, tokens, or secrets stored
  - User can inspect/delete all memories
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.memory")


class MemoryType(str, Enum):
    """Memory storage types."""
    SESSION = "session"
    PERSISTENT = "persistent"


class MemoryCategory(str, Enum):
    """Memory content categories."""
    PREFERENCE = "preference"
    PROJECT = "project"
    TASK = "task"
    REMINDER = "reminder"
    SUMMARY = "summary"
    GENERAL = "general"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    category: MemoryCategory
    memory_type: MemoryType
    created_at: float
    updated_at: float
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category.value,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
        }


class MemoryManager:
    """
    Manages session and persistent memory.

    Usage:
        memory = MemoryManager()
        await memory.initialize()

        # Save a memory
        entry = await memory.save(
            content="User prefers dark mode",
            category=MemoryCategory.PREFERENCE,
        )

        # Search memories
        results = await memory.search("dark mode")

        # Delete a memory
        await memory.delete(entry.id)

        # Clear all memories
        await memory.clear()
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path
        self._db = None
        self._initialized = False
        self._session_memory: list[MemoryEntry] = []

    async def initialize(self) -> None:
        """Initialize the SQLite database for persistent memory."""
        import aiosqlite

        if self._db_path is None:
            self._db_path = "data/pengu_memory.db"

        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_category ON memories(category)
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
        """)
        await self._db.commit()
        self._initialized = True
        logger.info("memory_initialized", db_path=self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def save(
        self,
        content: str,
        category: MemoryCategory = MemoryCategory.GENERAL,
        memory_type: MemoryType = MemoryType.PERSISTENT,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Save a memory entry.

        Args:
            content: The memory content
            category: Category of the memory
            memory_type: SESSION or PERSISTENT
            tags: Optional tags for searching
            metadata: Optional metadata

        Returns:
            The saved MemoryEntry
        """
        # Validate content
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")

        # Privacy check — reject sensitive content
        content_lower = content.lower()
        sensitive_patterns = [
            "password", "api_key", "api key", "token", "secret",
            "private_key", "private key", "ssh_key", "ssh key",
            "credential", "auth_token", "auth token",
        ]
        for pattern in sensitive_patterns:
            if pattern in content_lower:
                logger.warning("memory_rejected_sensitive", category=category.value)
                raise ValueError(f"Memory content appears to contain sensitive information: {pattern}")

        now = time.time()
        entry = MemoryEntry(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            content=content.strip(),
            category=category,
            memory_type=memory_type,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            metadata=metadata or {},
        )

        if memory_type == MemoryType.SESSION:
            self._session_memory.append(entry)
            logger.info("memory_saved_session", entry_id=entry.id, category=category.value)
        elif memory_type == MemoryType.PERSISTENT:
            if not self._initialized:
                await self.initialize()
            assert self._db is not None
            await self._db.execute(
                """INSERT INTO memories (id, content, category, memory_type, created_at, updated_at, tags, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.id,
                    entry.content,
                    entry.category.value,
                    entry.memory_type.value,
                    entry.created_at,
                    entry.updated_at,
                    json.dumps(entry.tags),
                    json.dumps(entry.metadata),
                ),
            )
            await self._db.commit()
            logger.info("memory_saved_persistent", entry_id=entry.id, category=category.value)

        return entry

    async def search(
        self,
        query: str,
        category: Optional[MemoryCategory] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """
        Search memories by content.

        Args:
            query: Search query
            category: Optional category filter
            memory_type: Optional type filter (SESSION/PERSISTENT)
            limit: Maximum results

        Returns:
            List of matching MemoryEntry objects
        """
        results: list[MemoryEntry] = []
        query_lower = query.lower()

        # Search session memory
        if memory_type is None or memory_type == MemoryType.SESSION:
            for entry in self._session_memory:
                if query_lower in entry.content.lower():
                    if category is None or entry.category == category:
                        results.append(entry)

        # Search persistent memory
        if memory_type is None or memory_type == MemoryType.PERSISTENT:
            if self._initialized and self._db:
                sql = "SELECT * FROM memories WHERE 1=1"
                params: list[Any] = []

                if category:
                    sql += " AND category = ?"
                    params.append(category.value)

                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type.value)

                sql += " ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)

                cursor = await self._db.execute(sql, params)
                rows = await cursor.fetchall()

                for row in rows:
                    entry = MemoryEntry(
                        id=row[0],
                        content=row[1],
                        category=MemoryCategory(row[2]),
                        memory_type=MemoryType(row[3]),
                        created_at=row[4],
                        updated_at=row[5],
                        tags=json.loads(row[6]) if row[6] else [],
                        metadata=json.loads(row[7]) if row[7] else {},
                    )
                    if query_lower in entry.content.lower():
                        results.append(entry)

        # Sort by relevance (newest first)
        results.sort(key=lambda e: e.updated_at, reverse=True)
        return results[:limit]

    async def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory entry by ID."""
        # Check session memory first
        for entry in self._session_memory:
            if entry.id == entry_id:
                return entry

        # Check persistent memory
        if self._initialized and self._db:
            cursor = await self._db.execute(
                "SELECT * FROM memories WHERE id = ?", (entry_id,)
            )
            row = await cursor.fetchone()
            if row:
                return MemoryEntry(
                    id=row[0],
                    content=row[1],
                    category=MemoryCategory(row[2]),
                    memory_type=MemoryType(row[3]),
                    created_at=row[4],
                    updated_at=row[5],
                    tags=json.loads(row[6]) if row[6] else [],
                    metadata=json.loads(row[7]) if row[7] else {},
                )

        return None

    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: The entry ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Check session memory
        for i, entry in enumerate(self._session_memory):
            if entry.id == entry_id:
                self._session_memory.pop(i)
                logger.info("memory_deleted_session", entry_id=entry_id)
                return True

        # Check persistent memory
        if self._initialized and self._db:
            cursor = await self._db.execute(
                "DELETE FROM memories WHERE id = ?", (entry_id,)
            )
            await self._db.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("memory_deleted_persistent", entry_id=entry_id)
            return deleted

        return False

    async def clear(
        self,
        memory_type: Optional[MemoryType] = None,
        category: Optional[MemoryCategory] = None,
    ) -> int:
        """
        Clear memories.

        Args:
            memory_type: Optional type filter (None = clear both)
            category: Optional category filter (None = clear all)

        Returns:
            Number of entries deleted
        """
        count = 0

        # Clear session memory
        if memory_type is None or memory_type == MemoryType.SESSION:
            if category:
                self._session_memory = [
                    e for e in self._session_memory if e.category != category
                ]
            else:
                count += len(self._session_memory)
                self._session_memory.clear()

        # Clear persistent memory
        if memory_type is None or memory_type == MemoryType.PERSISTENT:
            if self._initialized and self._db:
                sql = "DELETE FROM memories WHERE 1=1"
                params: list[Any] = []

                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type.value)

                if category:
                    sql += " AND category = ?"
                    params.append(category.value)

                cursor = await self._db.execute(sql, params)
                await self._db.commit()
                count += cursor.rowcount

        logger.info("memory_cleared", count=count)
        return count

    async def list_all(
        self,
        memory_type: Optional[MemoryType] = None,
        category: Optional[MemoryCategory] = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """List all memories with optional filters."""
        results: list[MemoryEntry] = []

        # Session memory
        if memory_type is None or memory_type == MemoryType.SESSION:
            for entry in self._session_memory:
                if category is None or entry.category == category:
                    results.append(entry)

        # Persistent memory
        if memory_type is None or memory_type == MemoryType.PERSISTENT:
            if self._initialized and self._db:
                sql = "SELECT * FROM memories WHERE 1=1"
                params: list[Any] = []

                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type.value)

                if category:
                    sql += " AND category = ?"
                    params.append(category.value)

                sql += " ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)

                cursor = await self._db.execute(sql, params)
                rows = await cursor.fetchall()

                for row in rows:
                    results.append(MemoryEntry(
                        id=row[0],
                        content=row[1],
                        category=MemoryCategory(row[2]),
                        memory_type=MemoryType(row[3]),
                        created_at=row[4],
                        updated_at=row[5],
                        tags=json.loads(row[6]) if row[6] else [],
                        metadata=json.loads(row[7]) if row[7] else {},
                    ))

        results.sort(key=lambda e: e.updated_at, reverse=True)
        return results[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "session_count": len(self._session_memory),
            "initialized": self._initialized,
            "db_path": self._db_path,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_memory: Optional[MemoryManager] = None


def get_memory(db_path: Optional[str] = None) -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory
    if _memory is None:
        _memory = MemoryManager(db_path)
    return _memory


def reset_memory() -> MemoryManager:
    """Reset the global memory manager (for testing)."""
    global _memory
    _memory = MemoryManager()
    return _memory
