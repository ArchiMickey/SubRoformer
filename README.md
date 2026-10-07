![SubRoformer — Music Source Separation, with the app icon on the left](app/assets/subroformer-banner.png)

# SubRoformer

This is the official repository for **SubRoformer: Subband Time-Domain
Representation for Music Source Separation**.

Music source separation into **vocals, bass, drums, and other**, with a local
desktop app and tools for SubRoformer's training, evaluation, and inference.

The perceptual-checkpoint config and waveform inference now default to DSP3,
with native CPU/CUDA/MPS modulation and the existing MPS FFT cache. Explicit
DSP/DSP2 configs remain supported. See [DSP3 validation](subroformer/frontend/dsp3/README.md).

## Contributions

- **Subband time-domain representation:** exposes spectral structure while
  preserving fine temporal and phase information.
- **Efficient subband analysis–synthesis front end:** reduces redundant full-rate
  computation and shortens the sequences presented to the separator.
- **Multiscale axial RoFormer:** captures long-range temporal and cross-band
  dependencies while retaining fine-grained information for mask estimation.

## The app
Demo on [Netrum & Halvorsen - Phoenix](https://www.youtube.com/watch?v=yH88qRmgkGI&list=RDyH88qRmgkGI&start_radio=1):

<div align="center">

https://github.com/user-attachments/assets/45b3fa51-0370-4635-a93b-738856479956

</div>

**SubStemStudio** lets you drop in a song, separate four stems, listen
while separation progresses, and export 48 kHz float32 WAVs. It includes
synchronized spectrograms and per-stem playback controls.

The source app supports macOS, Windows, and Linux. Core ML is the default on
macOS 15+; PyTorch supports CUDA, Apple Silicon (MPS), and CPU.
See the [app installation and usage guide](app/README.md) to get started.

## Checkpoints and test SDR

Both checkpoints are available on
[Hugging Face](https://huggingface.co/archimickey/SubRoformer).

| Checkpoint | Vocals | Bass | Drums | Other | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Base, 1.4M updates (EMA) | 12.6863 | 11.4961 | 12.2267 | 9.9445 | **11.5884** |
| Perceptual finetuning, +200k updates (EMA, default) | 12.6424 | 11.1357 | 12.0518 | 9.7584 | **11.3971** |

MUSDB18-HQ full-test `museval` SDR in dB (50 songs). Each stem is the median of
per-song median SDRs; Mean averages the four stem medians. For these extra-data
checkpoint results, see the
[evaluation protocol](subroformer/training/README.md#published-test-results).
They are not separate benchmarks of the FP16 or Core ML exports.

The perceptual-finetuned checkpoint sacrifices some SDR (**0.1913 dB** mean)
to prioritize perceptual quality. SDR alone does not establish a perceptual
improvement; that requires matched listening validation. It adds 200k updates
to the 1.4M Base and uses the matching **SiLU** config, not the original
**GEGLU** config. See [checkpoint files and configs](docs/checkpoints.md).

## Quick start

Run these commands from the repository root with Python 3.11 and a PyTorch
installation suitable for your hardware:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e .
```

See [source installation](docs/installation.md) for environment details.

### Train

Prepare aligned 48 kHz stereo stems for each song:

```text
data/musdb18hq/train/<song>/{mixture,vocals,bass,drums,other}.wav
```

Train the Base four-stem model on one CUDA GPU with BF16 support:

```bash
uv run accelerate launch --num_processes 1 --gpu_ids 0 --mixed_precision bf16 \
  --module subroformer.training.train \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --data_path data/musdb18hq/train \
  --output_dir checkpoints/my_run
```

Add `--max_updates 2` for a short smoke run. At completion, training saves
`model.pt` and `model_ema.pt` in the output directory; use `model_ema.pt` for
inference with the same YAML config. See the [training guide](subroformer/training/README.md)
for multiple GPUs and resuming training.

### Inference from the CLI

Use the released perceptual-finetuned checkpoint (downloaded automatically
if absent locally):

```bash
uv run python -m subroformer.inference \
  --input song.wav --output_dir outputs/song
```

Or use the checkpoint from your training run:

```bash
uv run python -m subroformer.inference \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --checkpoint checkpoints/my_run/model_ema.pt \
  --input song.wav --output_dir outputs/my_run
```

The CLI selects CUDA, then MPS, then CPU automatically. Override with
`--device cuda`, `--device mps`, or `--device cpu`. It writes `vocals.wav`,
`bass.wav`, `drums.wav`, and `other.wav` at 48 kHz, resampling input as needed.

### Inference in Python

Load the released checkpoint once and reuse it for multiple songs:

```python
from subroformer.inference.engine import (
    load_audio, load_separator, resolve_amp_dtype, separate_audio,
)

model, config = load_separator(device="auto")
audio = load_audio("song.wav", int(config.audio.sample_rate))
stems = separate_audio(
    model, audio, config,
    amp_dtype=resolve_amp_dtype(next(model.parameters()).device),
)
vocals = stems["vocals"]  # float32 NumPy array: [channels, samples]
```

For your own weights, pass
`config_path="subroformer/configs/sbroformer_base_4stem.yaml"` and
`checkpoint_path="checkpoints/my_run/model_ema.pt"` to `load_separator`.
Keep each checkpoint paired with the config used to train it. See the
[inference guide](subroformer/inference/README.md) for cache, offline, and
Core ML options.

## Documentation

| Task | Guide |
| --- | --- |
| App installation, usage, and native builds | [app/README.md](app/README.md) |
| Python/source installation | [docs/installation.md](docs/installation.md) |
| Package modules and imports | [subroformer/README.md](subroformer/README.md) |
| Model configs and inheritance | [subroformer/configs/README.md](subroformer/configs/README.md) |
| Training and dataset preparation | [Training guide](subroformer/training/README.md) |
| Evaluation and SDR protocol | [Evaluation guide](subroformer/training/README.md#evaluation) |
| CLI and Python inference | [Inference guide](subroformer/inference/README.md) |
| DSP backends and defaults | [Frontend guide](subroformer/frontend/README.md) |
| Checkpoint downloads and provenance | [docs/checkpoints.md](docs/checkpoints.md) |
| Core ML setup and export | [Core ML guide](subroformer/inference/README.md#core-ml-inference) |
| Tests and validation | [docs/testing.md](docs/testing.md) |
| Historical inference optimization experiments | [docs/inference_optimization.md](docs/inference_optimization.md) |

## Acknowledgments

Thanks to Roman Solovyev (ZFTurbo) for
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
from which this project's dataset and DataLoader code are derived.
