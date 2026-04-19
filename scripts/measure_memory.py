"""
measure_memory.py  –  Monitor RSS of a given PID at a fixed interval.

Usage:
    python scripts/measure_memory.py --pid 12345 --interval 5
"""

from __future__ import annotations

import argparse
import sys
import time


def _read_rss_mb(pid: int) -> float | None:
    """Return RSS of *pid* in MB, or None if the process is gone."""
    try:
        import psutil

        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    except Exception:
        return None

    # Fallback for macOS / Linux without psutil
    try:
        from pathlib import Path

        statm = Path(f"/proc/{pid}/statm").read_text()
        pages = int(statm.split()[1])  # resident pages
        import resource

        page_size = resource.getpagesize()
        return pages * page_size / (1024 * 1024)
    except Exception:
        pass

    # macOS fallback: use ps
    try:
        import subprocess

        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
        ).strip()
        if out:
            return int(out) / 1024  # ps reports KB
    except Exception:
        pass

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor RSS of a PID and report peak usage"
    )
    parser.add_argument("--pid", type=int, required=True, help="Process ID to monitor")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds (default: 5)",
    )
    args = parser.parse_args()

    pid = args.pid
    interval = args.interval
    max_rss_mb: float = 0.0
    samples: int = 0

    print(f"Monitoring PID {pid} every {interval:.1f}s  (Ctrl+C to stop)")
    print()

    try:
        while True:
            rss = _read_rss_mb(pid)
            if rss is None:
                print(f"Process {pid} not found or exited.")
                break
            samples += 1
            if rss > max_rss_mb:
                max_rss_mb = rss
            print(f"  [{samples:>5}] RSS = {rss:>8.1f} MB  (peak {max_rss_mb:.1f} MB)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print()

    print()
    print(f"Samples collected : {samples}")
    print(f"Peak RSS          : {max_rss_mb:.1f} MB")


if __name__ == "__main__":
    sys.exit(main() or 0)
