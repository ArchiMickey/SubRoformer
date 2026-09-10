from concurrent.futures import Future

import numpy as np
import pytest
import torch

from app import coreml_worker


class FakeExecutor:
    def __init__(self, **kwargs):
        assert kwargs["max_workers"] == 1
        assert kwargs["mp_context"].get_start_method() == "spawn"
        self.closed = False
        self.calls = []
        self.error = None

    def submit(self, function, *args):
        self.calls.append((function, args))
        result = Future()
        if self.error:
            result.set_exception(self.error)
        elif function is coreml_worker._load:
            result.set_result("config")
        else:
            result.set_result(args[0].copy())
        return result

    def shutdown(self, *, wait, cancel_futures):
        assert wait and cancel_futures
        self.closed = True


def test_process_preserves_pcm_and_closes(monkeypatch):
    monkeypatch.setattr(coreml_worker, "ProcessPoolExecutor", FakeExecutor)
    model = coreml_worker.CoreMLProcess("artifact", "mps")
    assert model.executor.calls[0] == (coreml_worker._load, ("artifact", "mps"))
    assert model.config == "config"
    assert next(model.parameters()).device.type == "cpu"
    audio = torch.randn(1, 2, 128)
    assert torch.equal(model(audio), audio)
    model.executor.error = RuntimeError("prediction failed")
    with pytest.raises(RuntimeError, match="prediction failed"):
        model(audio)
    model.close()
    assert model.executor.closed


def test_loading_failure_closes_process(monkeypatch):
    executor = FakeExecutor(max_workers=1, mp_context=coreml_worker.multiprocessing.get_context("spawn"))
    executor.error = RuntimeError("load failed")
    monkeypatch.setattr(coreml_worker, "ProcessPoolExecutor", lambda **kwargs: executor)
    with pytest.raises(RuntimeError, match="load failed"):
        coreml_worker.CoreMLProcess("artifact", "mps")
    assert executor.closed


def test_child_inference_has_no_grad_and_preserves_output(monkeypatch):
    def separator(audio):
        assert not torch.is_grad_enabled()
        return audio * 2

    monkeypatch.setattr(coreml_worker, "_separator", separator)
    audio = np.ones((1, 2, 128), dtype=np.float32)
    np.testing.assert_array_equal(coreml_worker._predict(audio), audio * 2)


def test_download_finishes_before_loading_phase(monkeypatch):
    events = []
    monkeypatch.setattr(coreml_worker, "resolve_coreml_artifact", lambda **kwargs: events.append("download") or "artifact")
    monkeypatch.setattr(coreml_worker, "ProcessPoolExecutor", FakeExecutor)
    model, config = coreml_worker.load_coreml_process(model_loading=lambda: events.append("loading"))
    try:
        assert events == ["download", "loading"]
        assert config == "config"
    finally:
        model.close()
