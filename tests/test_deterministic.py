"""
Tests for deterministic tools — real filesystem, terminal, git operations.
"""

import os
import tempfile
import pytest
from pathlib import Path
from pengu.tools.deterministic import (
    fs_read_file,
    fs_write_file,
    fs_list_directory,
    fs_search_files,
    fs_grep,
    terminal_execute,
    app_open,
    git_execute,
    git_status,
)


class TestFileSystemReadFile:
    """Test filesystem.read_file tool."""

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = await fs_read_file(str(test_file))
        assert result.success is True
        assert result.output["content"] == "Hello, World!"
        assert result.output["lines"] == 1

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        result = await fs_read_file("/nonexistent/file.txt")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_directory_fails(self, tmp_path):
        result = await fs_read_file(str(tmp_path))
        assert result.success is False
        assert "not a file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_multiline_file(self, tmp_path):
        test_file = tmp_path / "multi.txt"
        test_file.write_text("line1\nline2\nline3")

        result = await fs_read_file(str(test_file))
        assert result.success is True
        assert result.output["lines"] == 3


class TestFileSystemWriteFile:
    """Test filesystem.write_file tool."""

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        test_file = tmp_path / "new.txt"
        result = await fs_write_file(str(test_file), "new content")
        assert result.success is True
        assert result.output["bytes_written"] > 0
        assert test_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, tmp_path):
        test_file = tmp_path / "sub" / "dir" / "file.txt"
        result = await fs_write_file(str(test_file), "nested content")
        assert result.success is True
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_write_overwrites_existing(self, tmp_path):
        test_file = tmp_path / "overwrite.txt"
        test_file.write_text("old content")
        result = await fs_write_file(str(test_file), "new content")
        assert result.success is True
        assert test_file.read_text() == "new content"


class TestFileSystemListDirectory:
    """Test filesystem.list_directory tool."""

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.py").write_text("b")
        (tmp_path / "subdir").mkdir()

        result = await fs_list_directory(str(tmp_path))
        assert result.success is True
        assert result.output["count"] == 3
        names = [e["name"] for e in result.output["entries"]]
        assert "file1.txt" in names
        assert "subdir" in names

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self):
        result = await fs_list_directory("/nonexistent/dir")
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tmp_path):
        result = await fs_list_directory(str(tmp_path))
        assert result.success is True
        assert result.output["count"] == 0


class TestFileSystemSearch:
    """Test filesystem.search_files tool."""

    @pytest.mark.asyncio
    async def test_search_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")

        result = await fs_search_files("*.py", str(tmp_path))
        assert result.success is True
        assert result.output["count"] == 2

    @pytest.mark.asyncio
    async def test_search_no_matches(self, tmp_path):
        (tmp_path / "a.txt").write_text("")

        result = await fs_search_files("*.xyz", str(tmp_path))
        assert result.success is True
        assert result.output["count"] == 0


class TestFileSystemGrep:
    """Test filesystem.grep tool."""

    @pytest.mark.asyncio
    async def test_grep_finds_match(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass")
        (tmp_path / "b.py").write_text("def world():\n    pass")

        result = await fs_grep("def hello", str(tmp_path))
        assert result.success is True
        assert result.output["count"] >= 1

    @pytest.mark.asyncio
    async def test_grep_no_match(self, tmp_path):
        (tmp_path / "a.py").write_text("hello world")

        result = await fs_grep("nonexistent", str(tmp_path))
        assert result.success is True
        assert result.output["count"] == 0

    @pytest.mark.asyncio
    async def test_grep_case_insensitive(self, tmp_path):
        (tmp_path / "a.py").write_text("HELLO world")

        result = await fs_grep("hello", str(tmp_path))
        assert result.success is True
        assert result.output["count"] >= 1


class TestTerminalExecute:
    """Test terminal.execute tool."""

    @pytest.mark.asyncio
    async def test_execute_simple_command(self):
        result = await terminal_execute("echo hello", shell="powershell")
        assert result.success is True
        assert "hello" in result.output["stdout"]

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(self):
        result = await terminal_execute("echo error >&2", shell="powershell")
        # Command may succeed but have stderr
        assert result.output["exit_code"] == 0 or result.output["stderr"] != ""

    @pytest.mark.asyncio
    async def test_execute_failing_command(self):
        result = await terminal_execute("exit 1", shell="powershell")
        assert result.success is False
        assert result.output["exit_code"] == 1

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        result = await terminal_execute(
            "Start-Sleep -Seconds 60", shell="powershell", timeout=2
        )
        assert result.success is False
        assert "timed out" in result.error.lower()


class TestApplicationOpen:
    """Test application.open tool."""

    @pytest.mark.asyncio
    async def test_open_notepad(self):
        result = await app_open("notepad")
        assert result.success is True
        assert result.output["status"] == "launched"

    @pytest.mark.asyncio
    async def test_open_unknown_application(self):
        result = await app_open("nonexistent_app_xyz_123")
        # On Windows, 'start' command may succeed even for unknown apps
        # Just verify it returns a ToolResult
        from pengu.tools.registry import ToolResult
        assert isinstance(result, ToolResult)
        # If it fails, it should say "not found"
        if not result.success:
            assert "not found" in result.error.lower()


class TestGitTools:
    """Test git tools."""

    @pytest.mark.asyncio
    async def test_git_status_in_repo(self, tmp_path):
        # Initialize a git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(tmp_path), capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(tmp_path), capture_output=True,
        )

        result = await git_status(cwd=str(tmp_path))
        assert result.success is True
        assert "exit_code" in result.output

    @pytest.mark.asyncio
    async def test_git_status_not_a_repo(self, tmp_path):
        result = await git_status(cwd=str(tmp_path))
        # git status returns error when not in a repo
        assert result.output["exit_code"] != 0 or result.success is False
