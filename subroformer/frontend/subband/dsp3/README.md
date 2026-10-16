[← Frontend guide](../../README.md)

# DSP3 default and checkpoint compatibility

Run shell examples from the repository root.

The perceptual-checkpoint config, PyTorch waveform inference, and Core ML DSP
runtime now default to `dsp3`. Explicit `dsp` and `dsp2` selections are preserved;
`subroformer/configs/sbroformer_base_4stem_silu_dsp2.yaml` remains the DSP2 comparison config.
Training/evaluation configs without a backend retain their legacy `dsp` default.
Global-STFT frontends are unchanged and reject explicit DSP2/DSP3 selections.

```yaml
model:
  dsp_backend: dsp3
```

## What changed

Imported the local `../SBRoformer/dsp3` snapshot, with source SHA256 hashes in
`subroformer/frontend/subband/dsp3/source_manifest.json`. The source was untracked, so these hashes identify
the snapshot rather than its repository HEAD. FIR design, triangular banks,
FFT operations, decimation, and checkpoint tensor names remain unchanged.
The imported MIT license is retained in `subroformer/frontend/subband/dsp3/LICENSE`.
The manifest and historical benchmark JSONs retain their original `dsp3/...`
paths and hashes; the package move does not rewrite that provenance.

**This DSP3 snapshot already matches this repository's optimized DSP2 math.**
Wrapped 32-bit fixed-point modulation and native kernels had already been
ported here. DSP3 is an explicit synchronized backend, not a new long-audio
algorithm or a demonstrated speedup over the current DSP2. Wrapping avoids
large-angle FP32 trigonometric error; frequency quantization can still drift
over arbitrarily long recordings.

CPU uses a separately named lazy C++ extension, CUDA uses Triton, and MPS uses
the imported Metal shader. Unsupported inputs and unavailable native backends
retain the PyTorch reference fallback. Frequencies are fixed; signal gradients
and higher-order gradients remain supported. Native source files and provenance
ship in wheels. The desktop bundle includes DSP3 native sources.

The existing MPS FFT-cache wrapper was adapted to DSP3, including cache
invalidation and uncached training/gradient paths. This avoids losing the
current MPS optimization when switching backends. No checkpoint conversion or
Core ML neural-network re-export is required.

## Perceptual-checkpoint gate

The FP16 perceptual-finetuned 200k safetensors checkpoint loads strictly into
both backends. CUDA comparisons use FP32 model weights loaded from that file,
with FP32, FP16 autocast, and BF16 autocast; TF32 is disabled. Results include
the checkpoint hash, source hashes, runtime, and per-stem errors.

- All 50 MUSDB18-HQ test mixtures: a two-second excerpt starting at 10 seconds.
- One of those mixtures also uses a 30-second excerpt, exercising streaming
  overlap-add with the unchanged two-second neural-network window.
- At 48 kHz, hop ratio 4, batch size 1: **153 paired comparisons, 612 stem
  comparisons, all bit-identical** (maximum absolute and relative RMS errors 0).
- This is output parity, not full-song test-set SDR or a listening study.
  It establishes no quality improvement over DSP2.

Raw results: `benchmarks/results/dsp3-checkpoint-cuda.json`.
An additional CPU FP32 check on the first two-second excerpt is also
bit-identical for all four stems: `benchmarks/results/dsp3-checkpoint-cpu.json`.

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m benchmarks.validate_dsp3_checkpoint \
  --dataset /datasets/musdb18hq/test \
  --output benchmarks/results/dsp3-checkpoint-cuda.json
```

## Validation and limitations

Regression coverage includes CPU/CUDA output and gradient parity, long-audio
phase wrapping, strict state loading, finite model backward/optimizer steps,
fallbacks, inference defaults, explicit legacy overrides, and streaming memory.
Native kernel benchmarks compare against the unfused PyTorch reference, not
against DSP2, and distinguish modulation-only from the full two-second bank.

On this Linux host (PyTorch 2.9.1+cu128, RTX 4090), the measured full two-second
analysis+synthesis bank took 214.35 ms on CPU with eight threads and 3.33 ms on
CUDA. Relative to the unfused reference, these runs measured 1.10x and 1.46x;
30-second synthesis modulation alone measured 1.77x and 2.73x. CPU timings vary
between trials. These are not end-to-end model speedups or DSP2 comparisons.
Maximum full-bank absolute errors versus the reference were 1.43e-6 (CPU) and
1.19e-6 (CUDA). Raw timings and all trials are in
`benchmarks/results/dsp3-cpu8.json` and `benchmarks/results/dsp3-cuda.json`.

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -o addopts= -q
.venv/bin/python -m benchmarks.benchmark_dsp3_kernels --device cpu --threads 8 \
  --output benchmarks/results/dsp3-cpu8.json
.venv/bin/python -m benchmarks.benchmark_dsp3_kernels --device cuda --threads 1 \
  --output benchmarks/results/dsp3-cuda.json
```

MPS/Metal source and cache tests are included, but MPS hardware and Core ML
execution are unavailable on this Linux host. Those device-specific tests are
skipped; no new Apple-runtime or MPS-performance validation is claimed. Run the
DSP3 MPS kernel/cache tests and benchmark on a supported Mac before claiming
MPS parity or speed.

Final Linux validation: `pytest tests app/tests -o addopts= -q` passed 885 tests
with 236 skips (unavailable hardware/optional dependencies). Two- and 30-second
stereo analysis/synthesis matched DSP2 bit-for-bit on CPU and CUDA. Both project
sdists/wheels built successfully; `twine check` passed (SubRoformer retains its
pre-existing missing long-description metadata warnings). Installed-wheel
native CPU/CUDA modulation and gradients also matched the standalone DSP3
package exactly over 30-second phase spans, with both extensions coexisting.
