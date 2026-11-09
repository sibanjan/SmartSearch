"""One writer per index.

When the SmartSearch daemon (from smart-search-agents) runs on an index, it must be the only
process that writes to it: its job queue applies one change at a time, and a second writer
causes "database is locked" errors or lost updates. Reading (search, stats) is always safe.
The daemon announces itself with a pid file next to index.db.
"""

from __future__ import annotations

import os
from pathlib import Path

PID_FILE = "daemon.pid"


class IndexBusy(RuntimeError):
    """Another process (the daemon) owns writes to this index."""


def daemon_pid(index_dir: str | os.PathLike) -> int | None:
    """The pid of a live daemon for this index, or None."""
    try:
        pid = int((Path(index_dir).expanduser() / PID_FILE).read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid != os.getpid() and _alive(pid) else None


def ensure_can_write(index_dir: str | os.PathLike) -> None:
    pid = daemon_pid(index_dir)
    if pid is not None:
        raise IndexBusy(
            f"The SmartSearch daemon (pid {pid}) is running on this index and must be its only "
            "writer. Ask it to do this instead (`ssa jobs run index`, `ssa jobs run image_scan`, "
            "or Settings > Check files now), or stop it first with `ssa daemon stop`. "
            "Searching works while it runs."
        )


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
