from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.coreml_worker import CoreMLProcess
from coreml_inference import load_local_coreml_separator
from inference import load_audio


def measure(model, audio):
    gaps = []
    durations = []
    stop = threading.Event()

    def heartbeat():
        previous = time.perf_counter()
        while not stop.wait(0.005):
            current = time.perf_counter()
            gaps.append(current - previous)
            previous = current

    with torch.inference_mode():
        for _ in range(2):
            model(audio)
        thread = threading.Thread(target=heartbeat)
        thread.start()
        try:
            for _ in range(10):
                start = time.perf_counter()
                output = model(audio)
                durations.append(time.perf_counter() - start)
        finally:
            stop.set()
            thread.join()
    return output, {"median_window_s": statistics.median(durations),
                    "heartbeat_max_ms": max(gaps) * 1000,
                    "heartbeat_p95_ms": float(np.percentile(gaps, 95)) * 1000,
                    "warm_seconds": durations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/coreml-fp16")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audio = torch.from_numpy(load_audio(args.audio, 48000)[:, :96000].copy())[None]
    reference, _ = load_local_coreml_separator(args.artifact, dsp_device="mps")
    expected, direct = measure(reference, audio)
    del reference
    model = CoreMLProcess(args.artifact, "mps")
    try:
        actual, isolated = measure(model, audio)
    finally:
        model.close()
    report = {"torch": torch.__version__, "warmup": 2, "repeats": 10,
              "direct": direct, "isolated": isolated,
              "bitwise_equal": torch.equal(expected, actual),
              "max_absolute_error": (expected - actual).abs().max().item()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["bitwise_equal"]:
        raise RuntimeError("Process isolation changed waveform output")


if __name__ == "__main__":
    main()
