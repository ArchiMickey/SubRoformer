[← SubRoformer](../../README.md)

# Inference

First complete [source installation](../../docs/installation.md). Run commands from the
repository root. For app usage, see the [app guide](../../app/README.md);
for Core ML, see the [Core ML guide](README.md#core-ml-inference).

## Streaming outputs

CLI and in-app file inference stream finalized float32 WAV samples using a
bounded overlap-add buffer. Input decoding still holds the input waveform in
memory. The Python API returns full arrays by default; callers can instead use
`separate_audio(..., on_chunk=callback, collect_output=False)` to consume named
chunks without retaining full outputs. The callback receives
`(start_sample, {stem: channels_by_samples_array})`.

## CLI and model resolution

Run inference with automatic Hugging Face fallback:

```bash
uv run python -m subroformer.inference --input song.wav --output_dir outputs/song \
  --device cuda --precision bf16
```

The default model is the 200k perceptual-finetuned EMA with SiLU fusion.
Without `--config`, inference uses the installed package's
`sbroformer_base_4stem_silu.yaml`, independently of the working directory.
The checkpoint defaults to
`checkpoints/subroformer_base_4stem_1400000_ema_perceptual_finetuned_200000_fp16.safetensors`
relative to the working directory. Missing checkpoints and missing explicit
config paths are downloaded separately from
[`archimickey/SubRoformer`](https://huggingface.co/archimickey/SubRoformer)
into the Hugging Face cache. Explicit `--config` / `--checkpoint` paths also
take precedence when they exist; missing explicit paths fall back to the
canonical perceptual-finetuned artifacts, not similarly named remote files. Existing invalid
files raise a load error rather than being replaced.

Use `--cache_dir` to choose the cache, `--revision` to select a commit/tag,
and `--local_files_only` for offline local/cache-only loading. The repository
is public; complete anonymous config/checkpoint downloads were verified on
2026-09-06 without `HF_TOKEN` or a cached login. Private/gated repositories require
an authorized token via `hf auth login` or `HF_TOKEN`. The original FP32 Base artifacts are available at revision
`23497a6972e2b880552a42e10b99893cb3ff205b`; this older revision does not
contain the default perceptual-finetuned artifacts.

The locally supplied Base four-stem EMA checkpoint has 1,400,000 optimizer
updates. It is intentionally ignored by Git; see [checkpoint details](../../docs/checkpoints.md).

```bash
uv run python -m subroformer.inference \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --checkpoint checkpoints/subroformer_base_4stem_1400000_ema.safetensors \
  --input song.wav --output_dir outputs/song \
  --device cuda --precision bf16
```

## macOS / Apple Silicon (Metal)

Use a native arm64 Python environment on Apple Silicon. Metal inference is
verified with PyTorch 2.14.0; older PyTorch releases may lack the complex/FFT
operations this model needs. No FlashAttention or CPU-fallback environment
variable is needed with the verified version.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[test]' 'torch==2.14.0'
.venv/bin/python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
.venv/bin/python -m subroformer.inference --input song.wav --output_dir outputs/song \
  --device mps --precision auto
```

`--device auto` (the CLI default) prefers CUDA, then Metal (`mps`), then CPU.
`--precision auto` (the default) uses FP16 on Metal/MPS, prefers BF16 when
supported on CUDA (otherwise FP16), and retains FP32 on CPU.
PyTorch is the default backend; Core ML requires `--backend coreml`.
Explicit Metal BF16 requires macOS 15 or newer.
Explicit `fp32`, `bf16`, and `fp16` overrides remain available. Inference uses
autocast rather than converting all model weights or complex DSP to half precision.
For serving, use `load_separator(device="mps")` or `device="auto"`; its default
remains CPU. Start with inference batch size 1. The neural network and DSP run
on Metal; audio decoding, resampling, and overlap-add use CPU arrays.

To prepare the ignored local checkpoint for offline inference:

```bash
.venv/bin/hf download archimickey/SubRoformer \
  checkpoints/subroformer_base_4stem_1400000_ema.safetensors \
  --revision 23497a6972e2b880552a42e10b99893cb3ff205b --local-dir .
```

Use `--local_files_only` after downloading. Authentication may be required as
described above. FFmpeg is not needed for WAV inference, but the evaluation
dependencies imported by the broader test suite need `ffmpeg` and `ffprobe`
on PATH (`brew install ffmpeg` on macOS).

For CPU, use `--device cpu --precision fp32`. Output is four floating-point
WAVs: `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav`, at 48 kHz.
Input is resampled to the model rate; mono input is duplicated to stereo.
Output duration follows the resampled input, including short clips. Existing
output files are not overwritten. The default uses two-second windows with
fourfold overlap. File inference keeps the decoded input in RAM and streams finalized output
samples to disk; it is not live microphone processing.

## Python API

For serving, load the model once and return named stem arrays without writing
files or computing SDR:

```python
from subroformer.inference.engine import load_audio, load_separator, separate_audio

model, config = load_separator(
    device="cpu",
)
audio = load_audio("song.wav", int(config.audio.sample_rate))
stems = separate_audio(model, audio, config)
vocals = stems["vocals"]
```

Each value is a CPU float32 array shaped `[channels, samples]`; inputs to
`separate_audio` must already use the configured sample rate. CUDA and Metal callers
can pass `amp_dtype=resolve_amp_dtype(next(model.parameters()).device)` after
importing `resolve_amp_dtype` from `subroformer.inference.engine` to use the same automatic policy.
Omitting `amp_dtype` retains FP32 in this low-level API. The shared `separate_overlap_add` core
lives in `subroformer/inference/engine.py`; evaluation imports it and adds reference loading and
SDR scoring. Importing inference does not load evaluation, museval, or Accelerate.
`load_separator()` uses the same packaged-config and local-first checkpoint resolution and accepts
`repo_id`, `revision`, `cache_dir`, and `local_files_only` keyword arguments.

## Checkpoint export

Both inference and evaluation accept safetensors and legacy PyTorch weights.
`subroformer.checkpoint_utils.load_model_checkpoint(model, path)` always loads strictly.
To export another checkpoint without overwriting an existing file:

```bash
uv run python -m subroformer.checkpoint_utils --checkpoint model.pt --output model.safetensors
```

Export preserves tensor values and dtypes; optimizer state is excluded. Keep
the matching YAML config with the weights. Safetensors exports are ignored by Git.

Add `--fp16` to convert floating tensors to FP16 while preserving non-floating
tensors and provenance metadata. The default download is the FP16 perceptual-finetuned copy;
the original FP32 checkpoint remains available by explicit path. The desktop
app preserves checkpoint precision (`load_separator(..., preserve_dtype=True)`)
and has no precision override. The CLI retains its independent `--precision`
autocast policy and FP32 model loading.

## Core ML inference

The desktop app defaults to Core ML on macOS 15+ and PyTorch elsewhere, with
MPS selected when available. CLI/API defaults remain PyTorch, including for
`--backend auto`; CLI automatic precision uses FP16 on MPS. Core ML is selected
explicitly in the CLI/API on macOS 15+ via `--backend coreml`
or `load_separator(device="auto", backend="coreml")`. It downloads the compiled
200k perceptual-finetuned Base mask network from `archimickey/SubRoformer`, under
`coreml/base-4stem-perceptual-200k-fp16-v1/`. Its matching config is
`subroformer/configs/sbroformer_base_4stem_silu.yaml`: SiLU fusion is required, not GEGLU.
Both PyTorch and Core ML default to this finetuned model; backend selection
is unchanged. Custom config/checkpoint pairs require PyTorch.

This is a hybrid backend: the learned mask network runs in Core ML at FP16
with RMSNorm, RMSNorm2d, GRN, and the small TimeMix blocks kept in FP32
for numerical stability, including quiet tails;
complex-valued subband analysis, STFT, and synthesis run in PyTorch, on CPU by
default. With PyTorch 2.14+, explicitly select `--device mps` to run this DSP on
the GPU as well. Rolling overlap-add and public audio tensors remain on CPU.
Exact DSP buffers from the source checkpoint are bundled
in `dsp.safetensors`. No full PyTorch checkpoint is downloaded for this backend.
The runtime defaults to fixed-point DSP3 when the artifact config omits
`dsp_backend`. On MPS this uses the native Metal modulation kernel and retains
the inference filter FFT cache. Explicit `dsp_backend: dsp` remains supported.
This changes only the runtime DSP, not the compiled neural network; see
[DSP3 parity and validation limits](../frontend/dsp3/README.md).
Core ML uses `CPU_AND_GPU`; this is not an Apple Neural Engine deployment, nor
a claim of improved speed. Core ML schedules supported operations itself.
The earlier Base artifacts remain under `coreml/base-4stem-v1/` (FP32) and
`coreml/base-4stem-fp16-v1/` (FP16); neither is the default or downloaded by
the current backend. DSP and public audio tensors
remain FP32; only the compiled neural network uses FP16. For the CLI, use
`--precision auto` or `--precision fp16`; requesting FP32/BF16 requires PyTorch.

The compiled shape is one 2-second stereo window at 48 kHz. Existing overlap-add
handles arbitrary file lengths and mono duplication. Batch requests are processed
one window at a time by Core ML. Custom contexts and weights require a new export;
they are not silently substituted with the Base artifact.

```sh
python -m subroformer.inference --input song.wav --output_dir stems --backend coreml
python -m subroformer.inference --input song.wav --output_dir stems --backend coreml --device mps
python -m subroformer.inference --input song.wav --output_dir stems --backend torch --device mps
```

The MPS DSP path uses the **same compiled checkpoint**, without re-export or
reduced precision: DSP remains FP32. Features and masks still cross the CPU
Core ML API boundary. `--device auto` retains CPU DSP for compatibility;
PyTorch 2.7 fails the MPS inverse-STFT path. Explicit MPS requests require
PyTorch 2.14+ and available MPS hardware and are never silently retried on CPU.
The API equivalent is `load_separator(backend="coreml", device="mps")`, or
`load_coreml_separator(dsp_device="mps")`. In the desktop app, select **Core ML**
as the backend and **mps** as the device. The app's Auto device also selects MPS
when available. The app defaults to Core ML on macOS 15+; PyTorch can still
be selected manually, including for custom config/checkpoint pairs.
The app isolates Core ML and its DSP in a spawned process to prevent native
prediction calls from blocking Python playback callbacks. CLI/API inference
continues in-process. This changes scheduling, not model weights or precision.

Reproduce the CPU-DSP versus MPS-DSP comparison with a local artifact and cached
PyTorch checkpoint (use the application runtime, not the older converter venv):

```sh
.venv/bin/python -m benchmarks.benchmark_coreml_dsp \
  --validation-audio /path/to/stereo-music.wav \
  --artifact artifacts/coreml-perceptual-200k-fp16 \
  --output artifacts/coreml-dsp-benchmark.json
```

The benchmark records load/first-forward and warmed timings and checks noise,
silence, and beginning/middle/end music waveforms against FP32 PyTorch and the
unchanged CPU-DSP Core ML reference. MPS results are not bitwise identical;
the existing 0.1% relative RMS / 1e-6 absolute-floor threshold still applies.

HF downloads use its normal cache; `--local_files_only` and the app's offline
checkbox prohibit network access. Missing/corrupt artifacts and Core ML loading
errors are reported, not silently retried through another inference engine.
The app displays reconstructed artifact bytes, percentage, and an estimated
download ETA. File counts and Xet network-transfer bytes are not treated as model
bytes; ETA starts estimating after byte progress is observed. Model loading is
shown as a separate phase after the download completes.
If a compiled artifact cannot load on a particular Mac, use `--backend torch`
in the CLI, or select PyTorch in the desktop app.

### Reproduce the export

The tested converter environment is Python 3.11, PyTorch 2.7.0, NumPy 2.2.6,
and coremltools 9.0. Keep it separate from training/runtime dependencies:

```sh
uv venv --python 3.11 .venv-coreml
uv pip install --python .venv-coreml/bin/python -e . 'torch==2.7.0' 'numpy==2.2.6' 'coremltools==9.0'
.venv-coreml/bin/python -m subroformer.inference.export_coreml --validation-audio /path/to/stereo-music.wav
```

The exporter emits `mask.mlpackage`, `mask.mlmodelc`, `config.yaml`,
`dsp.safetensors`, and `conversion.json` under `artifacts/coreml-perceptual-200k-fp16/`.
Validation compares the compiled mask and complete waveforms against FP32
PyTorch loaded from the published FP16 checkpoint, using noise, silence, and
beginning/middle/end music chunks. The report includes checkpoint SHA-256 and numerical errors;
it is not a full MUSDB SDR evaluation or cross-machine compatibility guarantee.
Waveform validation permits at most 0.1% RMS error, with a 1e-6 absolute RMS
floor (-120 dBFS) for near-silent outputs. Both relative and absolute errors
are recorded, including quiet tails where relative error alone is misleading.
The trace fixes shape-dependent branches intentionally for this input shape.

Pass `--upload` to publish only after validation succeeds. It uploads to the
default `archimickey/SubRoformer` repository using the existing HF login.
Publish the compiled directory,
config, DSP buffers, and validation report together in one HF commit. The
portable `.mlpackage` may also be published for future recompilation; the app
downloads only the compiled runtime files. Build the app on macOS with
`python app/build.py` after installing `app/requirements-build.txt`.

See Apple's [compiled model prediction documentation](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html).
