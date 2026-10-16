# Inference optimization experiments

## Integrated default (September 8, 2026)

Step 3 is now constructed automatically by `SBRoformer` for the waveform-subband frontend. Normal `load_separator`, `build_model`, and direct `SBRoformer` calls require no extra optimization wrapper. The shared implementation lives in `subroformer/frontend/subband/dsp/inference_optimized.py`, which is included by the existing DSP package configuration. The macOS app and DMG were subsequently rebuilt on September 8, 2026, with the integrated module verified inside the bundle.

Eval-mode forwards with gradients disabled use the cached path on CPU and MPS. Training and gradient-enabled execution retain the legacy DSP path; returning to training also releases inference caches. Checkpoint keys/values, transformer arithmetic, and the legacy `subroformer/frontend/subband/dsp/subband_fast_triangle.py` implementation are unchanged. Global-STFT frontends are unaffected. Modulation caching remains experimental and is not enabled by default.

The tables below retain the original sequential experiments. Their source hashes describe the pre-merge files, not today's integrated model. `benchmarks/reference_model.py` rebuilds the unchanged legacy filter for continued comparisons, so the benchmark does not accidentally compare two already-optimized models.

The merged model was checked with the trained checkpoint using the same two-second input, two warmups and seven interleaved measurements:

| Device / precision | Legacy median | Merged median | Latency reduction | Bitwise equal |
| --- | ---: | ---: | ---: | --- |
| CPU / FP32 | 3.14602 s | 3.06738 s | 2.50% | Yes |
| MPS / FP16 autocast | 0.68690 s | 0.65478 s | 4.68% | Yes |

Every measured forward and full-checkpoint silence/tone/variable-length probe had zero maximum absolute error. The raw [merged-model result](../benchmarks/results/inference_20260907/merged_step_3.json) contains timings and provenance; compare within this run, not across runs with different clock/thermal conditions.

Merge validation: `pytest tests app/tests/test_memory.py app/tests/test_streaming.py` passed **445 tests**, with **3 skipped**. CPU training-gradient comparisons remain exact. MPS backward repeated legacy runs differed by up to approximately `3.3e-7`, as did legacy-versus-merged runs; the MPS gradient test uses `rtol=1e-5, atol=1e-6`, while inference checks remain strictly bitwise. Returning to training clears the cache. Source hashing confirms `subroformer/model.py` differs from the pre-merge version only by the import and filter-construction lines; all previously existing DSP/inference files remain intact.

## Scope and preservation

The initial experiments started on September 7 and completed on September 8, 2026. Their reference was the existing working-tree model, including the user's pre-existing edits, not a clean Git checkout. During those experiments, `model.py`, `inference.py`, `model_utils.py`, and every existing `dsp/*.py` file remained unchanged from the pre-experiment SHA-256 snapshot. Full hashes are recorded in each result JSON. These historical paths remain in the recorded provenance; the implementation now lives under `subroformer/`. The subsequently authorized merge changed only filter construction in the model.

The initial `subroformer/inference/optimized.py` entry point remains available for experiment compatibility. It produces an independent deep copy with a selected optimization level, without globally patching modules or changing the input instance. It is no longer needed for ordinary model use.

## Protocol

- Hardware: Apple M5 Pro; PyTorch 2.14.0; CPU restricted to four PyTorch threads.
- Configuration: `subroformer/configs/sbroformer_base_4stem.yaml` (128 realized bands, four outputs).
- Weights: local trained `subroformer_base_4stem_1400000_ema.safetensors`, loaded strictly as FP32 on both devices. The artifact hash is in the result files.
- Input: seeded synthetic noise at amplitude scale 0.02, shape `[1, 2, 96000]`, corresponding to two seconds at 48 kHz. This is a forward benchmark, not a music-quality benchmark.
- CPU: FP32. MPS: FP16 autocast with FP32 stored weights. Both use evaluation and inference modes.
- Each experiment uses two warmup forwards per variant and seven measured forwards per variant. A final confirmation uses eleven measurements per variant.
- Original, previous stage, and current stage are interleaved with rotating order; step 1 compares only original/current. MPS is synchronized before and after each timed forward. No other tests or benchmark jobs were run concurrently by this task.
- Timings exclude construction, loading, device transfer of input, output transfer to CPU, and equality checks. Warmup/cold observations and every individual timing are retained in JSON. Cache allocation happens during warmup.
- Equality is checked against the original on the **same device and precision**, including cold and warm execution. Baseline repeatability is checked separately. This does not claim CPU and MPS are bitwise equal to one another.
- No FFT-length changes, reordered sums, normalization/dtype changes, projection packing, compiler transformations, or approximate arithmetic were introduced.

## Sequential results

Each row includes all preceding stages. Positive percentages mean lower latency. Overall reduction is `100 * (1 - median(current) / median(original))` using that row's interleaved baseline. Absolute timings from different runs should not be compared as if environmental conditions were constant.

| Cumulative stage | CPU original → stage (s) | CPU reduction | MPS original → stage (s) | MPS reduction | Persistent DSP cache | Bitwise equal on both |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1. Validate fixed filters once | 3.300 → 3.289 | 0.34% | 0.927 → 0.957 | -3.25% | 0 MiB | Yes |
| 2. Reuse analysis input FFT | 3.280 → 3.291 | -0.34% | 0.926 → 0.915 | 1.15% | 0 MiB | Yes |
| 3. Cache fixed filter FFTs | 3.225 → 3.110 | 3.56% | 0.926 → 0.876 | 5.35% | 141.39 MiB | Yes |
| 4. Cache modulation tables | 3.160 → 3.063 | 3.06% | 0.930 → 0.883 | 5.02% | 236.70 MiB | Yes |

The marginal effect of each stage is measured against the immediately preceding stage in the same run, using the median of paired percentage reductions. This estimator is different from the ratio of medians in the table above.

| Added stage | CPU marginal reduction | MPS marginal reduction | Interpretation |
| --- | ---: | ---: | --- |
| 1 | -0.13% | 0.27% | No demonstrated standalone gain; MPS observations varied substantially. |
| 2 | -1.96% | 0.35% | No clear standalone gain; retained as shared-FFT groundwork. |
| 3 | 3.07% | 3.52% | Measured benefit on both devices. |
| 4 | 0.20% | 0.57% | Very small marginal change for another 95.31 MiB; not the default. |

Step 3 is the default. These short measurements are not confidence intervals or a guarantee of speedup on other CPUs, GPUs, segment sizes, batch sizes, or complete separation workloads.

Raw results: [step 1](../benchmarks/results/inference_20260907/step_1.json), [step 2](../benchmarks/results/inference_20260907/step_2.json), [step 3](../benchmarks/results/inference_20260907/step_3.json), [step 4](../benchmarks/results/inference_20260907/step_4.json).

## Pre-merge default confirmation

The final implementation defaults to step 3. A separate eleven-sample confirmation, after all code changes and tests, produced:

| Device / precision | Original median | Optimized median | Latency reduction | Speedup | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU / FP32 | 3.25184 s | 3.16143 s | 2.78% | 1.0286x | 0 |
| MPS / FP16 autocast | 0.93610 s | 0.89176 s | 4.74% | 1.0497x | 0 |

The retained cache is 141.39 MiB on either device. CPU observed ranges were 3.11928–3.30310 s original and 3.00578–3.19746 s optimized; MPS ranges were 0.91257–1.00239 s original and 0.88442–1.02163 s optimized. These overlapping ranges reinforce that the table reports sample medians, not a guaranteed per-call gain or statistical significance.

All original repeats, optimized forwards, and the silence/tone/variable-length probes were finite and bitwise-equal within each backend. The SHA-256 integrity check passed. The tested pre-merge source and benchmark hashes are recorded in [the confirmation artifact](../benchmarks/results/inference_20260907/recommended_step_3.json).

## Verification and cache behavior

- All sequential experiments produced finite outputs, bitwise-equal to the original; maximum absolute and RMS output errors were zero. Repeated original forwards were also bitwise equal.
- Full-checkpoint step-4 probes additionally cover silence `[1, 2, 96000]`, two-tone audio `[2, 2, 12347]`, and new noise `[1, 2, 48000]` on CPU and MPS. All are bitwise equal with zero error.
- The targeted suite passed **126 tests**, covering the new implementation, original model/inference paths, MPS precision and linear compatibility. New tests include single/four-stem models, batches 1/2, changing/odd lengths, silence, repeated calls, MPS FP32/FP16 execution, FP16 stored weights on MPS, checkpoint reload, buffer mutation, cache bounds, device transfer, and unchanged gradients through the fallback path.
- Cached spectra/modulation tensors are not in the state dict. The cache retains one signal length/device/dtype/filter-version combination, not every length encountered. Shape or relevant filter changes invalidate it; device/dtype conversion and checkpoint reload clear it. Reported cache sizes are retained tensor bytes, **not peak process/device memory**.
- Training or gradient-enabled calls fall back to the original DSP methods. Buffers constructed under inference mode lack version counters; those models also use the safe original DSP path rather than risk a stale cache. Construct/load/place the model outside inference mode to enable caching, and enter inference mode for its forwards.
- Use ordinary in-place buffer operations or checkpoint loading for filter changes, not untracked `.data` mutation. Call `optimized.sb_filter.clear_cache()` to explicitly release the retained cache.

## Usage

Use the existing model-loading API; no optimization wrapper is required:

```python
import torch

from subroformer.inference.engine import load_separator

model, config = load_separator(
    "subroformer/configs/sbroformer_base_4stem.yaml",
    "checkpoints/subroformer_base_4stem_1400000_ema.safetensors",
    device="cpu",
)
audio = torch.randn(1, 2, config.audio.chunk_size, device=next(model.parameters()).device)

with torch.inference_mode():
    estimate = model(audio)
```

For MPS, load on `device="mps"` and use `torch.autocast("mps", dtype=torch.float16)` around the forward. The existing `separate_audio` function already manages eval/inference mode. To reproduce non-default experiments, the compatibility helper still accepts explicit `level=1`, `2`, or `4`.

## Reproduce

Choose a new output path; the runner refuses to overwrite existing results:

```sh
.venv/bin/python -m benchmarks.benchmark_inference_optimized \
  --level 3 --merged --repeats 11 --verify-probes \
  --output benchmarks/results/inference_rerun.json

.venv/bin/python -m pytest tests/test_inference_optimized.py \
  tests/test_inference.py tests/test_mps_inference.py \
  tests/test_model.py tests/test_mps_linear.py
```

The JSON records source/config/checkpoint hashes, precision, input shape, raw samples, cold timings, comparisons, hardware, and runtime settings. The final confirmation also records UTC timestamps.
