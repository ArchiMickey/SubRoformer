# Optional fixed-point DSP backend

Set this field in your existing YAML config to use `dsp2` for both subband
analysis and synthesis:

```yaml
model:
  dsp_backend: dsp2
```

The default perceptual-checkpoint config, `sbroformer_base_4stem_silu.yaml`,
explicitly selects `dsp2`. Training and evaluation use `dsp`, including its
level-3 inference filter caching, when the config omits `dsp_backend`.
PyTorch and Core ML inference default to `dsp2` when the config omits `dsp_backend`;
set it explicitly to `dsp` to retain legacy inference.
`dsp2` supports only the `waveform_subband` frontend. It uses the same FIR
filters and checkpoint keys, so existing PyTorch checkpoints load strictly
without conversion. Keep the rest of the checkpoint's model config unchanged.

Fixed-point phase accumulation wraps before conversion to radians, reducing
long-signal modulation error without FP64 oscillator tensors for FP32 inputs.
Full MUSDB18-HQ evaluation with the perceptual checkpoint measured 11.4023 dB
mean SDR versus 11.3971 dB with legacy DSP; this small difference does not
establish perceptual improvement.
The original filtering, padding, and boundary limitations are unchanged.

Complex CPU inputs use a lazily compiled C++ modulation kernel (requires a
C++ compiler and Ninja on PATH); complex CUDA inputs use an optional Triton
kernel. Complex64 MPS inputs use a lazily compiled Metal kernel through
[`torch.mps.compile_shader`](https://docs.pytorch.org/docs/stable/generated/torch.mps.compile_shader.html).
All three fuse wrapped phase generation, sine/cosine, and multiplication
without allocating a full oscillator tensor. Float32 phase is retained for
complex64 inputs, and complex128 uses float64 phase. Signal gradients use the
conjugate oscillator; frequencies remain non-learnable. Two- and three-dimensional
inputs with contiguous sample rows are read directly, including padded filterbank
outputs and expanded batch views. Other noncontiguous inputs are copied before
kernel execution.

On supported x86 CPUs, native compilation enables AVX2 vector trigonometry.
Linux OpenMP builds use PyTorch's configured thread count. Phase/trigonometry
is computed in 1024-sample stack tiles and reused across batch/channel rows.
CUDA fuses frequency quantization into the modulation launch, avoiding separate
PyTorch phase-step kernels. Complex64 uses bounded-phase fast sine/cosine;
complex128 retains double-precision libdevice trigonometry. None of these paths uses
recursive oscillators, so approximation error does not accumulate over time.

MPS quantizes frequencies once per band in a small Metal launch, then uses a
three-dimensional sample/band/batch grid for modulation. This avoids repeated
64-bit division per sample and retains only a small phase-step tensor, not a
full oscillator. Its complex multiplication preserves PyTorch MPS's fused
arithmetic rather than disabling contraction, which could amplify rounding
differences through the FP16 neural network. MPS does not support complex128.

If the CPU extension cannot load, Triton is absent, or the MPS shader API is
unavailable, modulation falls back
to the original tiled PyTorch DSP2 implementation. Real inputs, other devices,
and broadcast band shapes also retain that implementation. CUDA/MPS kernel
compilation and runtime errors are not silently suppressed. Native source is
included in Python wheels and the desktop app bundle.

Run kernel/reference output, gradient, and filterbank integration parity tests:

```bash
python -m pytest tests/test_dsp2_cpu_kernel.py tests/test_dsp2_cuda_kernel.py tests/test_dsp2_mps_kernel.py tests/test_dsp2_mps_inference.py tests/test_dsp2_kernel_dispatch.py tests/test_frequency_shift.py tests/test_dsp_backend.py
```

Parity is numerical, not bitwise: different native trigonometric implementations
can differ at floating-point rounding precision. The MPS probes additionally
require exact forward/first-gradient equality on the tested runtime. CUDA tests
skip without CUDA and Triton; MPS tests skip without MPS; native CPU tests require
the compiler toolchain.

## Reproducible speed and parity check

```bash
python benchmarks/benchmark_dsp2_kernels.py --device cpu --threads 8 --require-speedup --output /tmp/dsp2-cpu.json
python benchmarks/benchmark_dsp2_kernels.py --device cuda --threads 1 --require-speedup --output /tmp/dsp2-cuda.json
python benchmarks/benchmark_dsp2_kernels.py --device mps --threads 1 --require-speedup --output /tmp/dsp2-mps.json
```

The benchmark compares public dispatch with the retained original DSP2 path,
including phase preparation and allocation. It checks output parity, warms both
paths, alternates their timing order over three trials, and reports median wall
time with explicit CUDA/MPS synchronization via PyTorch's benchmark timer. Compilation is
excluded. It covers stereo complex64 modulation chunks (16 bands, 2 and 30 seconds)
and complete 118-band analysis/synthesis (2 seconds, 48 kHz). It does not measure
full-model separation or imply a universal speedup for every tensor shape/device.
For less CPU timing variation on multisocket machines, bind both paths to the
same physical cores with `taskset` (for example, `taskset -c 8-15` for eight
threads); choose cores available on your machine. JSON reports record affinity,
individual trials, numerical errors, and kernel source hashes.

On the tested Xeon Platinum 8480C (eight pinned physical cores) and RTX 4090,
warmed complex64 medians versus original DSP2 were:

| Workload | CPU speedup | CUDA speedup |
| --- | ---: | ---: |
| Decimated analysis modulation chunk | 6.00x | 3.76x |
| 2-second synthesis modulation chunk | 3.72x | 5.49x |
| 30-second synthesis modulation chunk | 2.12x | 2.72x |
| Complete 2-second analysis + synthesis | 1.08x | 1.45x |

Maximum absolute error across these modulation probes was `5.97e-7`; complete
filterbank error was at most `1.44e-6`, with relative RMS error below `2.65e-7`.
These are measured numerical differences, not bitwise equality or a dataset-level
quality evaluation. Full CPU filterbank gains are smaller and more timing-sensitive
than isolated modulation gains because FFT/filtering work remains unchanged.
Recorded trials are in `benchmarks/results/dsp2-cpu8.json` and
`benchmarks/results/dsp2-cuda.json` (paths relative to the repository root).

## Model and app defaults

The model's MPS evaluation path also reuses the existing level-3 filter FFT
cache, with DSP2 modulation on both cached and uncached paths. This retains the
app's previous FFT-reuse optimization when switching its Core ML DSP to DSP2.
Training, gradient-enabled calls, inference-created buffers, and CPU/CUDA DSP2
use the uncached filterbank. Cache entries are bounded to the current input
shape and invalidated on filter mutation/reload, device/dtype changes, and
training. No modulation tables are cached. Checkpoint keys are unchanged.
The experimental legacy `optimize_for_inference` helper still rejects DSP2
rather than silently reverting to legacy modulation.

The app keeps Core ML as the default neural backend on macOS 15+, selects MPS
when available, and now defaults its DSP to DSP2. The PyTorch app path also uses
DSP2 by default. The compiled Core ML mask network and checkpoint are unchanged;
no re-export or download is required for this DSP change. Explicit
`dsp_backend: dsp` configurations remain respected, and CPU remains selectable.

Run full-model benchmarks against the retained **uncached DSP2 reference**:

```bash
python benchmarks/benchmark_dsp2_mps_model.py --backend torch --audio song.wav --require-speedup --output /tmp/dsp2-mps-torch.json
python benchmarks/benchmark_dsp2_mps_model.py --backend coreml --audio song.wav --require-speedup --output /tmp/dsp2-mps-coreml.json
```

These use local checkpoints/artifacts, require exact output equality for noise,
silence, quiet input, and optional music, and time complete 2-second, batch-one
forwards including CPU output transfer. Core ML additionally times the prior
legacy cached DSP default; its output need not equal fixed-point DSP2. The
`--require-speedup` gate checks every reported timing baseline. Full-model gains
include FFT caching as well as the Metal kernel; isolated kernel gains do not.

On Apple M5 Pro, macOS 26.6.2, PyTorch 2.14.0, warmed median results were:

| Workload | Reference ms | Optimized ms | Speedup |
| --- | ---: | ---: | ---: |
| Decimated analysis modulation | 0.0832 | 0.0166 | 5.03x |
| 2-second synthesis modulation | 1.0328 | 0.2205 | 4.68x |
| 30-second synthesis modulation | 15.6543 | 4.7009 | 3.33x |
| Uncached 2-second analysis + synthesis | 74.9503 | 67.8294 | 1.105x |
| Complete PyTorch FP16 model, 2 seconds | 667.0133 | 627.4589 | 1.063x |
| Complete Core ML model, 2 seconds | 267.6522 | 224.0322 | 1.195x |

All measured MPS outputs matched the original DSP2 reference exactly, including
full-model noise, silence, quiet input, and music probes. The prior Core ML
**legacy DSP** default measured 226.4814 ms versus 224.0322 ms with optimized
DSP2, only a 1.011x gain; the 1.195x claim is specifically versus uncached DSP2.
These are local forward-latency measurements, not full-track/GUI throughput or
dataset quality claims. The filter FFT cache retains 148,258,080 bytes (about
141.39 MiB) for this model/shape, matching the legacy cache policy.

Reports with source/input hashes and all timing trials are
`benchmarks/results/dsp2-mps.json`, `benchmarks/results/dsp2-mps-torch.json`, and
`benchmarks/results/dsp2-mps-coreml.json`. The model benchmarks use four CPU
threads; isolated DSP benchmarks use one. Other devices and runtime versions
require their own validation.
