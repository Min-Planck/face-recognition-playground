"""
Module đo đạc tài nguyên hệ thống (CPU %, RAM MB, Latency, FPS).
Sử dụng psutil với background sampling thread trong quá trình thực thi hàm/pipeline.
"""

import os
import statistics as stats
import threading
import time
from typing import Any, Dict, List, Optional
import psutil


class ResourceMonitor:
    """
    Context manager theo dõi CPU % và RAM (MB) của tiến trình hiện tại.
    Lấy mẫu định kỳ trên background thread để đo chính xác mức sử dụng đỉnh (peak) và trung bình (avg).
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self._cpu_samples: List[float] = []
        self._ram_samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._elapsed_time: float = 0.0

    def _sample_loop(self):
        # Mồi bộ đếm CPU để tránh giá trị 0.0 ở chu kỳ đầu
        self.process.cpu_percent(interval=None)
        while self._running:
            time.sleep(self.interval)
            try:
                cpu = self.process.cpu_percent(interval=None)
                ram = self.process.memory_info().rss / (1024 * 1024)
                self._cpu_samples.append(cpu)
                self._ram_samples.append(ram)
            except Exception:
                pass

    def __enter__(self):
        self._cpu_samples.clear()
        self._ram_samples.clear()
        self._running = True
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        self._elapsed_time = time.perf_counter() - self._start_time
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def elapsed_seconds(self) -> float:
        return self._elapsed_time

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_time * 1000.0

    @property
    def fps(self) -> float:
        if self._elapsed_time > 0:
            return 1.0 / self._elapsed_time
        return 0.0

    @property
    def avg_cpu(self) -> float:
        return stats.mean(self._cpu_samples) if self._cpu_samples else 0.0

    @property
    def peak_ram(self) -> float:
        return max(self._ram_samples) if self._ram_samples else 0.0

    @property
    def avg_ram(self) -> float:
        return stats.mean(self._ram_samples) if self._ram_samples else 0.0

    def get_summary(self) -> Dict[str, float]:
        """Trả về dict chứa tất cả chỉ số đo đạc."""
        return {
            "elapsed_ms": round(self.elapsed_ms, 2),
            "fps": round(self.fps, 2),
            "avg_cpu_percent": round(self.avg_cpu, 2),
            "avg_ram_mb": round(self.avg_ram, 2),
            "peak_ram_mb": round(self.peak_ram, 2),
        }
