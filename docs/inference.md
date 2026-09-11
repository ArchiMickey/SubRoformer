[← SubRoformer](../README.md)

# Inference

First complete [source installation](installation.md). Run commands from the
repository root. For app usage, see the [app guide](../app/README.md);
for Core ML, see the [Core ML guide](coreml.md).

CLI and in-app file inference stream finalized float32 WAV samples using a
bounded overlap-add buffer. Input decoding still holds the input waveform in
memory. The Python API returns full arrays by default; callers can instead use
`separate_audio(..., on_chunk=callback, collect_output=False)` to consume named
chunks without retaining full outputs. The callback receives
`(start_sample, {stem: channels_by_samples_array})`.

Run inference with automatic Hugging Face fallback:

```bash
uv run python inference.py --input song.wav --output_dir outputs/song \
  --device cuda --precision bf16
```

The default model is the 200k perceptual-finetuned EMA with SiLU fusion.
The script uses existing `configs/sbroformer_base_4stem_silu.yaml` and
`checkpoints/subroformer_base_4stem_1400000_ema_perceptual_finetuned_200000_fp16.safetensors` relative to the
working directory first. Each missing artifact is downloaded separately from
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
updates. It is intentionally ignored by Git; see [checkpoint details](checkpoints.md).

```bash
uv run python inference.py \
  --config configs/sbroformer_base_4stem.yaml \
  --checkpoint checkpoints/subroformer_base_4stem_1400000_ema.safetensors \
  --input song.wav --output_dir outputs/song \
  --device cuda --precision bf16
```

### macOS / Apple Silicon (Metal)

Use a native arm64 Python environment on Apple Silicon. Metal inference is
verified with PyTorch 2.14.0; older PyTorch releases may lack the complex/FFT
operations this model needs. No FlashAttention or CPU-fallback environment
variable is needed with the verified version.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[test]' 'torch==2.14.0'
.venv/bin/python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
.venv/bin/python inference.py --input song.wav --output_dir outputs/song \
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

For serving, load the model once and return named stem arrays without writing
files or computing SDR:

```python
from inference import load_audio, load_separator, separate_audio

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
importing `resolve_amp_dtype` from `inference` to use the same automatic policy.
Omitting `amp_dtype` retains FP32 in this low-level API. The shared `separate_overlap_add` core
lives in `inference.py`; evaluation imports it and adds reference loading and
SDR scoring. Importing inference does not load evaluation, museval, or Accelerate.
`load_separator()` uses the same local-first download fallback and accepts
`repo_id`, `revision`, `cache_dir`, and `local_files_only` keyword arguments.

Both inference and evaluation accept safetensors and legacy PyTorch weights.
`checkpoint_utils.load_model_checkpoint(model, path)` always loads strictly.
To export another checkpoint without overwriting an existing file:

```bash
uv run python -m checkpoint_utils --checkpoint model.pt --output model.safetensors
```

Export preserves tensor values and dtypes; optimizer state is excluded. Keep
the matching YAML config with the weights. Safetensors exports are ignored by Git.

Add `--fp16` to convert floating tensors to FP16 while preserving non-floating
tensors and provenance metadata. The default download is the FP16 perceptual-finetuned copy;
the original FP32 checkpoint remains available by explicit path. The desktop
app preserves checkpoint precision (`load_separator(..., preserve_dtype=True)`)
and has no precision override. The CLI retains its independent `--precision`
autocast policy and FP32 model loading.
