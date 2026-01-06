"""Performance monitoring utilities for TTS synthesis."""

import time
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Track timing of synthesis stages for performance analysis.

    Usage:
        monitor = PerformanceMonitor()
        monitor.record("start")
        # ... do work ...
        monitor.record("inference_complete")
        # ... more work ...
        monitor.record("encoding_complete")

        report = monitor.report()
        # {'start': 0.0, 'inference_complete': 1.234, 'encoding_complete': 0.056, 'total': 1.290}
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize performance monitor.

        Args:
            name: Optional name for this monitoring session
        """
        self.name = name or "synthesis"
        self.start_time = time.perf_counter()
        self.events: List[Tuple[str, float]] = []

    def record(self, event: str) -> float:
        """
        Record a timing event.

        Args:
            event: Name of the event/stage

        Returns:
            Elapsed time since start
        """
        elapsed = time.perf_counter() - self.start_time
        self.events.append((event, elapsed))
        return elapsed

    def report(self) -> Dict[str, float]:
        """
        Generate timing report with per-stage durations.

        Returns:
            Dict mapping event names to durations (time since previous event)
        """
        result = {}
        prev_time = 0.0

        for event, timestamp in self.events:
            duration = timestamp - prev_time
            result[event] = round(duration, 4)
            prev_time = timestamp

        if self.events:
            result["total"] = round(self.events[-1][1], 4)
        else:
            result["total"] = 0.0

        return result

    def log_report(self, level: str = "info"):
        """
        Log the performance report.

        Args:
            level: Logging level (debug, info, warning, error)
        """
        report = self.report()
        log_fn = getattr(logger, level, logger.info)

        # Format as readable string
        stages = [f"{k}={v:.3f}s" for k, v in report.items() if k != "total"]
        total = report.get("total", 0)

        log_fn(f"[{self.name}] Performance: {', '.join(stages)} | Total: {total:.3f}s")

    def reset(self):
        """Reset the monitor for reuse."""
        self.start_time = time.perf_counter()
        self.events.clear()
