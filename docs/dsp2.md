# Optional fixed-point DSP backend

Set this field in your existing YAML config to use `dsp2` for both subband
analysis and synthesis:

```yaml
model:
  dsp_backend: dsp2
```

Training and evaluation default to `dsp`, including its level-3 inference filter
caching. PyTorch inference defaults to `dsp2` when the config omits `dsp_backend`;
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
kernel. Both fuse wrapped phase generation, sine/cosine, and multiplication
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
complex128 retains double-precision libdevice trigonometry. Neither path uses
recursive oscillators, so approximation error does not accumulate over time.

If the CPU extension cannot load or Triton is absent, modulation falls back
to the original tiled PyTorch DSP2 implementation. Real inputs, other devices,
and broadcast band shapes also retain that implementation. CUDA kernel runtime
errors are not silently suppressed. Native source is included in Python wheels.

Run kernel/reference output, gradient, and filterbank integration parity tests:

```bash
python -m pytest tests/test_dsp2_cpu_kernel.py tests/test_dsp2_cuda_kernel.py tests/test_dsp2_kernel_dispatch.py tests/test_frequency_shift.py tests/test_dsp_backend.py
```

Parity is numerical, not bitwise: different native trigonometric implementations
can differ at floating-point rounding precision. CUDA tests skip without CUDA
and Triton; native CPU tests require the compiler toolchain.

## Reproducible speed and parity check

```bash
python benchmarks/benchmark_dsp2_kernels.py --device cpu --threads 8 --require-speedup --output /tmp/dsp2-cpu.json
python benchmarks/benchmark_dsp2_kernels.py --device cuda --threads 1 --require-speedup --output /tmp/dsp2-cuda.json
```

The benchmark compares public dispatch with the retained original DSP2 path,
including phase preparation and allocation. It checks output parity, warms both
paths, alternates their timing order over three trials, and reports median wall
time with CUDA synchronization via PyTorch's benchmark timer. Compilation is
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

`dsp2` bypasses the legacy inference-cache wrapper; the experimental
`optimize_for_inference` helper rejects it rather than reverting to legacy
modulation. This option applies to PyTorch execution, not existing Core ML
artifacts. MPS compatibility and performance have not been validated.
