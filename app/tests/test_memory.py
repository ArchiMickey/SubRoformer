from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("PySide6")
from app import memory


def test_cpu_process_gb_and_system_percentage(monkeypatch):
    monkeypatch.setattr(memory.psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=2_500_000_000)))
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(percent=75.4))
    result = memory.read_memory("cpu")
    assert result.app_bytes == 2_500_000_000
    assert result.used_percent == 75.4
    assert "2.50 GB" in result.text() and "75.4%" in result.text()


def test_indexed_cuda_uses_same_device_for_both_readings(monkeypatch):
    calls = []
    monkeypatch.setattr(memory.torch.cuda, "memory_reserved", lambda index: calls.append(index) or 3_000_000_000)
    monkeypatch.setattr(memory.torch.cuda, "mem_get_info", lambda index: calls.append(index) or (2_000_000_000, 8_000_000_000))
    result = memory.read_memory("cuda:2")
    assert calls == [2, 2]
    assert result.device == "cuda:2" and result.used_percent == 75
    assert result.app_bytes == 3_000_000_000
    assert "reserved" in result.app_label


def test_auto_resolves_and_unindexed_cuda_uses_current_device(monkeypatch):
    monkeypatch.setattr(memory, "resolve_device", lambda device: torch.device("cuda"))
    monkeypatch.setattr(memory.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(memory.torch.cuda, "memory_reserved", lambda index: 0)
    monkeypatch.setattr(memory.torch.cuda, "mem_get_info", lambda index: (10, 10))
    result = memory.read_memory("auto")
    assert result.requested == "auto" and result.device == "cuda:3"
    assert "0.00 GB" in result.text() and "0.0%" in result.text()


def test_mps_distinguishes_metal_from_unified_system_ram(monkeypatch):
    monkeypatch.setattr(memory, "resolve_device", lambda device: torch.device("mps"))
    monkeypatch.setattr(memory.torch.mps, "driver_allocated_memory", lambda: 1_000_000_000)
    monkeypatch.setattr(memory.psutil, "virtual_memory", lambda: SimpleNamespace(percent=60))
    result = memory.read_memory("mps")
    assert result.app_bytes == 1_000_000_000 and result.used_percent == 60
    assert "Metal" in result.app_label and "unified" in result.system_label


def test_bad_device_is_unavailable_not_zero():
    result = memory.read_memory("invalid-device")
    assert result.app_bytes is None and result.used_percent is None
    assert result.text().count("N/A") == 2
    assert "unavailable" in result.detail


def test_failed_global_query_preserves_valid_app_reading(monkeypatch):
    monkeypatch.setattr(memory.torch.cuda, "memory_reserved", lambda index: 100)

    def unavailable(index):
        raise RuntimeError("device lost")

    monkeypatch.setattr(memory.torch.cuda, "mem_get_info", unavailable)
    result = memory.read_memory("cuda:1")
    assert result.app_bytes == 100 and result.used_percent is None
    assert "device lost" in result.detail
