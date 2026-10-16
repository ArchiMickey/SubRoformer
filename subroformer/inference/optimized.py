from __future__ import annotations

import copy

from ..frontend.subband.dsp.inference_optimized import OPTIMIZATION_STEPS, InferenceSubbandFilter


def optimize_for_inference(model, level: int = 3):
    """Return an independent eval copy for the sequential optimization experiments."""
    if level not in OPTIMIZATION_STEPS:
        raise ValueError(f"Unknown optimization level: {level}")
    if model.sb_filter is None:
        raise ValueError("These optimizations require the waveform-subband frontend")
    if getattr(model, "dsp_backend", "dsp") != "dsp":
        raise ValueError(f"These optimizations require dsp_backend='dsp'; use {model.dsp_backend} directly")
    optimized = copy.deepcopy(model)
    optimized.sb_filter = InferenceSubbandFilter(optimized.sb_filter, level)
    return optimized.eval()
