[← Package guide](../README.md)

# Subband frontends

The waveform-subband frontend analyzes audio into frequency-range signals and
synthesizes the separated waveforms. Its implementation versions live under
`subroformer/frontend/subband/`; choose one through `model.dsp_backend`:

| Backend | Implementation and guide |
| --- | --- |
| `dsp` | Legacy PyTorch analysis/synthesis under [dsp/](subband/dsp/), with inference filter FFT caching |
| `dsp2` | Wrapped fixed-point phase modulation with native CPU/CUDA/MPS kernels; [DSP2 guide](subband/dsp2/README.md) |
| `dsp3` | Imported snapshot matching the optimized DSP2 math, with its own native kernel identity and provenance; [DSP3 guide](subband/dsp3/README.md) |

`subroformer/frontend/subband_v2/` provides a separate complex FIR filterbank
with an explicit band count through `SubbandFilterV2`. Its complex subband
representation differs from the V1 representation used by existing SBRoformer
checkpoints, so it is not selected by `model.dsp_backend`.

DSP2 and DSP3 require `frontend_type: waveform_subband`. Global-STFT frontends
reject those explicit backend selections.

## Selection and defaults

Set the backend explicitly in a matching model config:

```yaml
model:
  dsp_backend: dsp3
```

The perceptual-finetuned config selects DSP3. PyTorch waveform inference and
Core ML DSP also select DSP3 when their config omits the backend.
Direct `SBRoformer` construction and training/evaluation retain the legacy
`dsp` default when the field is absent. Explicit `dsp` and `dsp2` settings remain
supported.

DSP selection preserves checkpoint tensor names; keep the remaining model
settings matched to the checkpoint. DSP3 does not establish a speed or quality
improvement over this repository's optimized DSP2. Backend-specific parity,
native compilation requirements, fallbacks, and measurements are documented in
the linked guides.

## Legacy DSP inference cache

The legacy `dsp` frontend automatically uses level-3 filter FFT caching for
evaluation with gradients disabled. Training and gradient-enabled execution
use the original path; returning to training releases the cache.
Models whose filter buffers were created inside inference mode also use the
original path.

The cache retains one signal length/device/dtype/filter-version combination.
Relevant shape or filter changes invalidate it; device/dtype conversion and
checkpoint reload clear it. Cached tensors are excluded from checkpoints.
Call `model.sb_filter.clear_cache()` to release the cache explicitly.
Modulation-table caching remains experimental and is not enabled by default.

See the [optimization experiments](../../docs/inference_optimization.md) for
dated measurements, reproduction commands, and historical source hashes.
Run those shell examples, and the backend guides' examples, from the repository
root.
