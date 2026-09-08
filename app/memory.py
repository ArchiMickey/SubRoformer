from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import psutil
import torch
from PySide6.QtCore import QThread, Signal

from inference import resolve_device


@dataclass
class MemoryUsage:
    requested: str
    device: str
    app_bytes: int | None = None
    used_percent: float | None = None
    app_label: str = "App memory"
    system_label: str = "Device-wide used"
    detail: str = ""

    def text(self):
        app = f"{self.app_bytes / 1_000_000_000:.2f} GB" if self.app_bytes is not None else "N/A"
        system = f"{self.used_percent:.1f}%" if self.used_percent is not None else "N/A"
        return f"{self.device.upper()} · {self.app_label}: {app} · {self.system_label}: {system}"


def read_memory(requested):
    usage = MemoryUsage(requested, requested)
    try:
        device = resolve_device(requested)
        usage.device = str(device)
        if device.type == "cpu":
            usage.app_label = "App RAM (RSS)"
            usage.system_label = "System RAM used"
            usage.app_bytes = psutil.Process().memory_info().rss
            usage.used_percent = psutil.virtual_memory().percent
            usage.detail = "Resident process RAM includes shared pages. System usage = (total − available) / total. GB = 10⁹ bytes."
        elif device.type == "mps":
            usage.app_label = "App Metal memory"
            usage.system_label = "System unified RAM used"
            usage.app_bytes = torch.mps.driver_allocated_memory()
            usage.used_percent = psutil.virtual_memory().percent
            usage.detail = "Metal driver allocations for this process, including caches. Percentage is whole-system unified RAM, not GPU-only usage. Do not add Metal allocations to process RSS. GB = 10⁹ bytes."
        else:
            index = device.index if device.index is not None else torch.cuda.current_device()
            usage.device = f"cuda:{index}"
            usage.app_label = "App PyTorch reserved"
            usage.system_label = "GPU-wide used"
            usage.app_bytes = torch.cuda.memory_reserved(index)
            free, total = torch.cuda.mem_get_info(index)
            usage.used_percent = (total - free) / total * 100
            usage.detail = "App value is PyTorch's reserved allocator memory (including cached blocks), not full process GPU memory: CUDA context and other driver allocations are excluded. Percentage includes all users of this GPU. GB = 10⁹ bytes."
    except Exception as error:
        usage.detail = f"Memory reading unavailable: {error}"
    return usage


class MemoryMonitor(QThread):
    updated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.requested = "auto"
        self.stopped = Event()

    def run(self):
        while not self.stopped.is_set():
            self.updated.emit(read_memory(self.requested))
            self.stopped.wait(1)

    def stop(self):
        self.stopped.set()
