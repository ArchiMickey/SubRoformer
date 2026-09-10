from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import torch

from coreml_inference import load_local_coreml_separator, resolve_coreml_artifact


_separator = None


def _load(directory, dsp_device):
    global _separator
    torch.set_num_threads(4)
    _separator, config = load_local_coreml_separator(directory, dsp_device=dsp_device)
    return config


def _predict(audio):
    with torch.inference_mode():
        return _separator(torch.from_numpy(audio)).numpy()


class CoreMLProcess(torch.nn.Module):
    def __init__(self, directory, dsp_device):
        super().__init__()
        self.register_parameter("device_marker", torch.nn.Parameter(torch.empty(0), requires_grad=False))
        self.executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        try:
            self.config = self.executor.submit(_load, directory, dsp_device).result()
        except BaseException:
            self.close()
            raise

    def forward(self, audio):
        return torch.from_numpy(self.executor.submit(_predict, audio.numpy()).result())

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def load_coreml_process(*, local_files_only=False, download_progress=None, dsp_device="cpu", model_loading=None):
    directory = resolve_coreml_artifact(local_files_only=local_files_only, download_progress=download_progress)
    if model_loading is not None:
        model_loading()
    model = CoreMLProcess(directory, dsp_device).eval()
    return model, model.config
