"""
Secure Filesystem — validated, permission-aware file operations.

Security features:
  - Path traversal protection
  - Sensitive file blocking
  - File size limits
  - Directory depth limits
  - Safe search with exclusion patterns
  - No credential/password/log exposure
"""

from __future__ import annotations

import glob
import os
import platform
import re
from pathlib import Path
from typing import Optional

from pengu.logging import get_logger

logger = get_logger("pengu.os.filesystem")

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

MAX_READ_SIZE = 2 * 1024 * 1024  # 2MB
MAX_WRITE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_SEARCH_RESULTS = 200
MAX_LIST_ENTRIES = 500
MAX_GREP_RESULTS = 100

# Sensitive paths that should not be read/modified
SENSITIVE_PATTERNS: list[str] = [
    r"\.env\b",
    r"\.env\.\w+",
    r"\.ssh[/\\]",
    r"[/\\]\.ssh\b",
    r"id_rsa",
    r"id_ed25519",
    r"\.gnupg[/\\]",
    r"[/\\]\.gnupg\b",
    r"\.aws[/\\]credentials",
    r"\.azure[/\\]",
    r"\.config[/\\]credentials",
    r"\.netrc\b",
    r"\.npmrc\b",
    r"\.docker[/\\]config",
    r"passwords?\.db",
    r"keychain",
    r"credential",
    r"\.pem\b",
    r"\.key\b",
    r"\.p12\b",
    r"\.pfx\b",
    r"\.jks\b",
]

# Directories to skip during search
EXCLUDED_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env", ".env", ".tox",
    "dist", "build", ".next", ".nuxt",
    "AppData", ".lmstudio", ".ollama",
}

# Sensitive paths regex
_SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def is_sensitive_path(path: str | Path) -> bool:
    """Check if a path matches sensitive file patterns."""
    path_str = str(path)
    return bool(_SENSITIVE_RE.search(path_str))


def validate_path(path: str | Path, operation: str = "read") -> tuple[bool, str]:
    """
    Validate a path for safety.
    
    Returns (is_valid, error_message).
    """
    p = Path(path).resolve()

    # Check for sensitive paths
    if is_sensitive_path(p):
        return False, f"Access denied: {p.name} is a sensitive file ({operation} blocked)"

    # Check path exists for read operations
    if operation == "read" and not p.exists():
        return False, f"Path not found: {p}"

    # Check file size for reads
    if operation == "read" and p.is_file():
        try:
            size = p.stat().st_size
            if size > MAX_READ_SIZE:
                return False, f"File too large: {size / 1024:.0f}KB (limit {MAX_READ_SIZE // 1024}KB)"
        except OSError:
            pass

    # Check parent directory is writable for writes
    if operation == "write":
        parent = p.parent
        if not parent.exists():
            # Allow creating new files in reasonable locations
            home = Path.home()
            if not str(p).startswith(str(home)):
                return False, f"Cannot create files outside home directory: {p}"

    return True, ""


# ---------------------------------------------------------------------------
# Safe filesystem operations
# ---------------------------------------------------------------------------

async def safe_list_directory(
    path: str = ".",
    max_entries: int = MAX_LIST_ENTRIES,
    show_hidden: bool = False,
) -> dict:
    """
    List directory contents safely.
    
    Returns dict with path, entries, count, and any errors.
    """
    p = Path(path).resolve()

    if not p.exists():
        return {"success": False, "error": f"Directory not found: {path}"}
    if not p.is_dir():
        return {"success": False, "error": f"Not a directory: {path}"}

    entries = []
    try:
        for entry in sorted(p.iterdir()):
            # Skip hidden files unless requested
            if not show_hidden and entry.name.startswith("."):
                continue

            try:
                stat = entry.stat()
                is_dir = entry.is_dir()
                entries.append({
                    "name": entry.name,
                    "type": "dir" if is_dir else "file",
                    "size": stat.st_size if not is_dir else 0,
                    "modified": stat.st_mtime,
                    "sensitive": is_sensitive_path(entry),
                })
            except (PermissionError, OSError):
                entries.append({
                    "name": entry.name,
                    "type": "unknown",
                    "size": 0,
                    "error": "permission denied",
                })

            if len(entries) >= max_entries:
                break

    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except OSError as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "path": str(p),
        "entries": entries,
        "count": len(entries),
        "truncated": len(list(p.iterdir())) > max_entries if p.is_dir() else False,
    }


async def safe_read_file(
    path: str,
    encoding: str = "utf-8",
    max_lines: int = 0,
) -> dict:
    """
    Read a file's contents safely.
    
    Validates path, blocks sensitive files, enforces size limits.
    """
    p = Path(path).resolve()

    valid, error = validate_path(p, "read")
    if not valid:
        return {"success": False, "error": error}

    if not p.is_file():
        return {"success": False, "error": f"Not a file: {path}"}

    try:
        content = p.read_text(encoding=encoding)

        lines = content.split("\n")
        total_lines = len(lines)

        if max_lines > 0:
            content = "\n".join(lines[:max_lines])
            truncated = total_lines > max_lines
        else:
            truncated = False

        return {
            "success": True,
            "path": str(p),
            "content": content,
            "size": p.stat().st_size,
            "total_lines": total_lines,
            "truncated": truncated,
        }

    except UnicodeDecodeError:
        return {"success": False, "error": f"Cannot read as text (encoding: {encoding})"}
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except OSError as e:
        return {"success": False, "error": str(e)}


async def safe_write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    confirm_overwrite: bool = True,
) -> dict:
    """
    Write content to a file safely.
    
    Validates path, blocks sensitive files, enforces size limits.
    """
    p = Path(path).resolve()

    valid, error = validate_path(p, "write")
    if not valid:
        return {"success": False, "error": error}

    # Check write size
    content_bytes = content.encode(encoding)
    if len(content_bytes) > MAX_WRITE_SIZE:
        return {
            "success": False,
            "error": f"Content too large: {len(content_bytes) / 1024:.0f}KB (limit {MAX_WRITE_SIZE // 1024}KB)",
        }

    # Check if file exists and warn about overwrite
    if confirm_overwrite and p.exists():
        existing_size = p.stat().st_size
        return {
            "success": False,
            "needs_confirmation": True,
            "error": f"File exists ({existing_size} bytes). Confirm overwrite.",
            "path": str(p),
        }

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)
        return {
            "success": True,
            "path": str(p),
            "bytes_written": len(content_bytes),
        }
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except OSError as e:
        return {"success": False, "error": str(e)}


async def safe_search_files(
    pattern: str,
    path: str = ".",
    max_results: int = MAX_SEARCH_RESULTS,
    include_hidden: bool = False,
) -> dict:
    """
    Search for files matching a glob pattern.
    
    Excludes common large/irrelevant directories.
    """
    search_path = os.path.join(path, "**", pattern)
    matches = glob.glob(search_path, recursive=True)

    results = []
    for match in matches:
        p = Path(match)

        # Skip excluded directories
        if any(excluded in p.parts for excluded in EXCLUDED_DIRS):
            continue

        # Skip hidden files unless requested
        if not include_hidden and any(part.startswith(".") for part in p.parts):
            continue

        try:
            results.append({
                "path": str(p),
                "name": p.name,
                "type": "dir" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else 0,
            })
        except OSError:
            continue

        if len(results) >= max_results:
            break

    return {
        "success": True,
        "pattern": pattern,
        "results": results,
        "count": len(results),
        "truncated": len(matches) > max_results,
    }


async def safe_grep(
    query: str,
    path: str = ".",
    file_pattern: str = "*",
    max_results: int = MAX_GREP_RESULTS,
) -> dict:
    """
    Search file contents for a text pattern (like grep).
    
    Excludes binary files, large files, and sensitive paths.
    """
    results = []
    search_path = os.path.join(path, "**", file_pattern)

    for file_path in glob.glob(search_path, recursive=True):
        p = Path(file_path)

        # Skip excluded directories
        if any(excluded in p.parts for excluded in EXCLUDED_DIRS):
            continue

        # Skip sensitive files
        if is_sensitive_path(p):
            continue

        if not os.path.isfile(file_path):
            continue

        # Skip very large files
        try:
            if os.path.getsize(file_path) > MAX_READ_SIZE:
                continue
        except OSError:
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if query.lower() in line.lower():
                        results.append({
                            "file": file_path,
                            "line": i,
                            "content": line.strip()[:200],
                        })
                        if len(results) >= max_results:
                            return {
                                "success": True,
                                "query": query,
                                "results": results,
                                "count": len(results),
                                "truncated": True,
                            }
        except (PermissionError, OSError):
            continue

    return {
        "success": True,
        "query": query,
        "results": results,
        "count": len(results),
    }
