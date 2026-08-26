"""
Process Manager — safe process inspection and management.

Provides process listing, info, and safe termination.
Only operates through typed interfaces — no arbitrary PID kills.

Security:
  - No system-critical process termination
  - No arbitrary PID termination from LLM output
  - Safe inspection only
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import psutil

from pengu.logging import get_logger

logger = get_logger("pengu.os.process_manager")

# Processes that should never be terminated
PROTECTED_PROCESSES: set[str] = {
    "system", "idle", "registry", "smss", "csrss", "wininit",
    "winlogon", "lsass", "services", "svchost", "dwm",
    "explorer", "searchhost", "taskhostw", "sihost",
    "fontdrvhost", "ctfmon", "RuntimeBroker",
}


@dataclass
class ProcessInfo:
    """Safe process information."""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_mb: float
    exe: str
    cmdline: str
    create_time: float
    num_threads: int

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "name": self.name,
            "status": self.status,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_mb": round(self.memory_mb, 1),
            "exe": self.exe,
            "cmdline": self.cmdline[:200] if self.cmdline else "",
            "num_threads": self.num_threads,
        }


class ProcessManager:
    """Safe process inspection and management."""

    def list_processes(
        self,
        name_filter: str = "",
        max_results: int = 50,
        min_memory_mb: float = 0,
    ) -> list[ProcessInfo]:
        """
        List running processes with optional filtering.
        
        Args:
            name_filter: filter by process name (case-insensitive substring)
            max_results: maximum number of results
            min_memory_mb: only return processes using more than this much RAM
        """
        processes = []
        for proc in psutil.process_iter(
            ["pid", "name", "status", "cpu_percent", "memory_info", "exe", "cmdline", "create_time", "num_threads"]
        ):
            try:
                info = proc.info
                name = info["name"] or ""

                if name_filter and name_filter.lower() not in name.lower():
                    continue

                mem = info.get("memory_info")
                memory_mb = (mem.rss / (1024 * 1024)) if mem else 0.0

                if memory_mb < min_memory_mb:
                    continue

                cmdline = " ".join(info.get("cmdline") or [])

                processes.append(ProcessInfo(
                    pid=info["pid"],
                    name=name,
                    status=info.get("status", "unknown"),
                    cpu_percent=info.get("cpu_percent", 0.0) or 0.0,
                    memory_mb=memory_mb,
                    exe=info.get("exe") or "",
                    cmdline=cmdline,
                    create_time=info.get("create_time", 0.0) or 0.0,
                    num_threads=info.get("num_threads", 0) or 0,
                ))

                if len(processes) >= max_results:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # Sort by memory usage descending
        processes.sort(key=lambda p: p.memory_mb, reverse=True)
        return processes

    def get_process(self, pid: int) -> Optional[ProcessInfo]:
        """Get detailed info for a specific process by PID."""
        try:
            proc = psutil.Process(pid)
            info = proc.as_dict(
                ["pid", "name", "status", "cpu_percent", "memory_info",
                 "exe", "cmdline", "create_time", "num_threads"]
            )
            mem = info.get("memory_info")
            memory_mb = (mem.rss / (1024 * 1024)) if mem else 0.0
            cmdline = " ".join(info.get("cmdline") or [])

            return ProcessInfo(
                pid=info["pid"],
                name=info["name"] or "",
                status=info.get("status", "unknown"),
                cpu_percent=info.get("cpu_percent", 0.0) or 0.0,
                memory_mb=memory_mb,
                exe=info.get("exe") or "",
                cmdline=cmdline,
                create_time=info.get("create_time", 0.0) or 0.0,
                num_threads=info.get("num_threads", 0) or 0,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning("process_not_found", pid=pid, error=str(e))
            return None

    def is_running(self, name: str) -> bool:
        """Check if a process with the given name is running."""
        name_lower = name.lower()
        for proc in psutil.process_iter(["name"]):
            try:
                proc_name = (proc.info["name"] or "").lower()
                if name_lower in proc_name:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def terminate(self, pid: int, force: bool = False) -> dict:
        """
        Safely terminate a process by PID.
        
        Refuses to terminate protected/system processes.
        """
        try:
            proc = psutil.Process(pid)
            name = proc.name()

            # Check protected processes
            if name.lower() in PROTECTED_PROCESSES:
                return {
                    "success": False,
                    "error": f"Refusing to terminate protected process: {name} (PID {pid})",
                }

            if force:
                proc.kill()
            else:
                proc.terminate()

            # Wait briefly for termination
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                if not force:
                    proc.kill()

            logger.info("process_terminated", pid=pid, name=name, force=force)
            return {
                "success": True,
                "pid": pid,
                "name": name,
                "action": "killed" if force else "terminated",
            }

        except psutil.NoSuchProcess:
            return {"success": False, "error": f"No process with PID {pid}"}
        except psutil.AccessDenied:
            return {"success": False, "error": f"Access denied for PID {pid}"}

    def get_top_cpu(self, n: int = 10) -> list[ProcessInfo]:
        """Get top N processes by CPU usage."""
        procs = self.list_processes(max_results=n)
        procs.sort(key=lambda p: p.cpu_percent, reverse=True)
        return procs[:n]

    def get_top_memory(self, n: int = 10) -> list[ProcessInfo]:
        """Get top N processes by memory usage."""
        return self.list_processes(max_results=n)  # already sorted by memory
