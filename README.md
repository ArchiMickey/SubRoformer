# SubRoformer

Standalone training, evaluation, and file inference for subband music source
separation. Base configs support joint four-stem output and separate single-stem
models. Existing `SBRoformer` class names and config stems are retained for
checkpoint compatibility.

## Install

Use Python 3.11 and a PyTorch installation appropriate for your CPU, CUDA, or Metal
environment. A source checkout is recommended for access to configs and tests.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[test]'
```

Torch attention is the default; FlashAttention is optional. No sibling
repository, private data adapter, or local dataset mount is required.

## Inference

Run inference with automatic Hugging Face fallback:

```bash
uv run python inference.py --input song.wav --output_dir outputs/song \
  --device cuda --precision bf16
```

The script uses existing `configs/sbroformer_base_4stem.yaml` and
`checkpoints/subroformer_base_4stem_1400000_ema.safetensors` relative to the
working directory first. Each missing artifact is downloaded separately from
[`archimickey/SubRoformer`](https://huggingface.co/archimickey/SubRoformer)
into the Hugging Face cache. Explicit `--config` / `--checkpoint` paths also
take precedence when they exist; missing explicit paths fall back to the
canonical Base artifacts, not similarly named remote files. Existing invalid
files raise a load error rather than being replaced.

Use `--cache_dir` to choose the cache, `--revision` to select a commit/tag,
and `--local_files_only` for offline local/cache-only loading. The repository
is currently private, so authenticate with `hf auth login` or set `HF_TOKEN`
to an authorized token. The weights and matching config are available at
revision `23497a6972e2b880552a42e10b99893cb3ff205b`; use `--revision` with
that commit for a pinned download.

The locally supplied Base four-stem EMA checkpoint has 1,400,000 optimizer
updates. It is intentionally ignored by Git; see [checkpoint details](docs/checkpoints.md).

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
`--precision auto` (the default) prefers BF16 when supported, otherwise FP16
on GPU; CPU retains FP32. Metal BF16 requires macOS 15 or newer.
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
fourfold overlap. File inference currently holds the decoded audio and window
outputs in RAM; it is not a streaming interface.

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

## Training

Provide your own aligned stereo audio in this layout:

```text
data/musdb18hq/train/<song>/{mixture,vocals,bass,drums,other}.wav
data/musdb18hq/test/<song>/{mixture,vocals,bass,drums,other}.wav
```

The generic loader also accepts FLAC. Training files must already match
`audio.sample_rate` (48 kHz for supplied configs). Resample all stems consistently
before training; do not merely change the header rate. The four targets must
be time-aligned. Random mixing constructs its mixture from the selected stems.
Dataset files are not supplied.

```bash
uv run accelerate launch --num_processes 1 --gpu_ids 0 --mixed_precision bf16 train.py \
  --config configs/sbroformer_base_4stem.yaml \
  --data_path data/musdb18hq/train
```

For two GPUs:

```bash
uv run accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 --mixed_precision bf16 train.py \
  --config configs/sbroformer_base_4stem.yaml \
  --data_path data/musdb18hq/train
```

Use `configs/sbroformer_base_4stem.yaml` for joint separation. For a single
target, use `configs/sbroformer_base_vocals.yaml`, `configs/sbroformer_base_bass.yaml`,
`configs/sbroformer_base_drums.yaml`, or `configs/sbroformer_base_other.yaml`.
All five configs are self-contained. Single-stem configs preserve the original
batch size of 4 and disable random mixing; joint training uses batch size 2
and enables random mixing. Single-stem training still needs all four source
stems to construct mixtures. `--max_updates 2` limits a smoke run.

Single-stem inference uses the same CLI with its matching config and separately
trained single-stem checkpoint. The supplied 1.4M-update joint checkpoint is
not compatible with single-stem configs; no single-stem weights are included.

Each epoch consists of configured synthetic training steps, not necessarily one
pass over the songs. The loader already shards distributed samples; it must not
be sharded again by `accelerator.prepare`.

Periodic evaluation is disabled by default. To enable the inherited train/test
subset monitor, specify `--eval_every_updates 20000 --eval_root data/musdb18hq`.
Do not use repeated test-subset measurements as a held-out checkpoint-selection
protocol. Use a separate validation split for model selection.

Training writes raw and EMA weights at completion and resumable Accelerate state
directories at `training.save_every_updates`. `--resume` requires such a complete
state directory; the supplied inference-only EMA file cannot resume optimization.

## Evaluation

```bash
uv run python evaluate.py \
  --config configs/sbroformer_base_4stem.yaml \
  --checkpoint checkpoints/subroformer_base_4stem_1400000_ema.pt \
  --root data/musdb18hq/test --device cuda --precision bf16 \
  --hop_ratio 4 --batch_size 1 --output_json eval_results/full_test.json
```

The evaluator sums the resampled reference stems as model input, matching the
SBRoformer evaluation protocol; it does not read `mixture.wav`. It computes
`museval` SDR at 44.1 kHz. Each stem score is the median across songs of per-song
median window SDR; the joint summary averages the four stem medians. Distributed
evaluation shards whole songs and aggregates all rows before taking medians.

## Tests

```bash
uv run pytest
uv run torchrun --standalone --nproc-per-node=2 tests/ddp_data_smoke.py
RANDOM_MIXING=0 uv run torchrun --standalone --nproc-per-node=2 tests/ddp_data_smoke.py
uv run torchrun --standalone --nproc-per-node=2 tests/ddp_evaluate_smoke.py
```

See [preparation validation](docs/validation.md) for the checks performed.
Tests use synthetic audio. This public recipe is not a reproduction of the
supplied checkpoint's extra-data training run.

## Acknowledgments

Thanks to Roman Solovyev (ZFTurbo) for
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
from which this project's dataset, DataLoader, dataset-type support, and
augmentation schema are derived.
