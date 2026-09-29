"""Process and host memory snapshots for log lines."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("pyuvkin")


def process_rss_bytes() -> int | None:
    """Current resident set size of this process, or ``None`` if unknown."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        if sys.platform == "darwin":
            out = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True,
            )
            return int(out.strip()) * 1024
    except Exception:
        pass
    try:
        import resource
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux: kilobytes; macOS: bytes (peak, not current)
        return rss if sys.platform == "darwin" else rss * 1024
    except Exception:
        return None


def system_memory_bytes() -> tuple[int | None, int | None]:
    """``(available, total)`` host RAM in bytes, or ``(None, None)``."""
    try:
        if sys.platform.startswith("linux"):
            text = Path("/proc/meminfo").read_text()
            vals = {}
            for line in text.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    vals[parts[0][:-1]] = int(parts[1]) * 1024
            total = vals.get("MemTotal")
            available = vals.get("MemAvailable", vals.get("MemFree"))
            return available, total
        if sys.platform == "darwin":
            page = int(os.sysconf("SC_PAGE_SIZE"))
            total = int(os.sysconf("SC_PHYS_PAGES")) * page
            out = subprocess.check_output(["vm_stat"], text=True)
            counts = {}
            for line in out.splitlines():
                m = re.match(r"([^:]+):\s+(\d+)", line)
                if m:
                    counts[m.group(1).strip()] = int(m.group(2))
            # free + inactive + speculative + purgeable ≈ reclaimable
            pages = sum(
                counts.get(k, 0)
                for k in (
                    "Pages free", "Pages inactive",
                    "Pages speculative", "Pages purgeable",
                )
            )
            return pages * page, total
    except Exception:
        pass
    return None, None


def format_bytes(n: int | None) -> str:
    if n is None:
        return "?"
    gb = n / 1e9
    if gb >= 10:
        return f"{gb:.1f} GB"
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{n / 1e6:.0f} MB"


def snapshot() -> dict:
    rss = process_rss_bytes()
    available, total = system_memory_bytes()
    return {"rss_bytes": rss, "available_bytes": available, "total_bytes": total}


def format_snapshot(snap: dict | None = None) -> str:
    snap = snap or snapshot()
    return (
        f"process {format_bytes(snap['rss_bytes'])} RSS; "
        f"{format_bytes(snap['available_bytes'])} available "
        f"({format_bytes(snap['total_bytes'])} total)"
    )


def log_memory(label: str | None = None, *, level: int = logging.INFO) -> dict:
    """Log current process RSS and host available/total memory; return snapshot."""
    snap = snapshot()
    msg = format_snapshot(snap)
    if label:
        logger.log(level, "memory [%s]: %s", label, msg)
    else:
        logger.log(level, "memory: %s", msg)
    return snap
