![SubRoformer — Music Source Separation, with the app icon on the left](app/assets/subroformer-banner.png)

# SubRoformer

This is the official repository for SubRoformer: Subband Time-Domain Representation for Music Source Separation. SubRoformer separates music mixtures into four stems: vocals, bass, drums, and other instruments.

![Subformer](app/assets/subroformer-framework.png)

## Results and Checkpoints

| Checkpoint | Vocals | Bass | Drums | Other | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Base, 1.4M updates (EMA) | 12.6863 | 11.4961 | 12.2267 | 9.9445 | **11.5884** |
| Perceptual finetuning, +200k updates (EMA, default) | 12.6424 | 11.1357 | 12.0518 | 9.7584 | **11.3971** |

The model is trained on **100 songs from MUSDB18-HQ** and **500 additional songs from an internal dataset**. Evaluation is conducted on the **MUSDB18-HQ test set (50 songs)** using `museval` SDR (dB). The SDR for each stem is computed as the median of the per-song median SDRs, while the Mean SDR is the average across the four stems. For details on the evaluation of checkpoints trained with additional data, see the [evaluation protocol](subroformer/training/README.md#published-test-results). 

The **perceptually fine-tuned checkpoint** prioritizes perceptual quality at the cost of a slight reduction in Mean SDR (**0.1913 dB**). It is trained for an additional **200k updates** starting from the Base checkpoint (1.4M updates) and requires the corresponding **SiLU** configuration rather than the original **GEGLU** configuration. See [checkpoint files and configs](docs/checkpoints.md) for details. Both checkpoints are available on [Hugging Face](https://huggingface.co/archimickey/SubRoformer).

## Quick start

Run these commands from the repository root with Python 3.11 and a PyTorch
installation suitable for your hardware:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e .
```

See [source installation](docs/installation.md) for environment details.

### Inference

Use the released perceptual-finetuned checkpoint (downloaded automatically):

```bash
uv run python -m subroformer.inference \
  --input demo/song.wav \
  --output_dir outputs/song
```


### Train

Prepare aligned 48 kHz stereo stems for each song:

```text
data/musdb18hq/train/<song>/{mixture,vocals,bass,drums,other}.wav
```

Train the Base four-stem model on one CUDA GPU with BF16 support:

```bash
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    --module subroformer.training.train \
    --config subroformer/configs/sbroformer_base_4stem.yaml \
    --data_path data/musdb18hq/train
```


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



See the [training guide](subroformer/training/README.md) for multiple GPUs and resuming training.



Or use the checkpoint from your training run:

```bash
uv run python -m subroformer.inference \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --checkpoint checkpoints/my_run/model_ema.pt \
  --input demo/song.wav \
  --output_dir outputs/my_run
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
audio = load_audio("demo/song.wav", int(config.audio.sample_rate))
stems = separate_audio(
    model, audio, config,
    amp_dtype=resolve_amp_dtype(next(model.parameters()).device),
)
vocals = stems["vocals"]  # float32 NumPy array: [channels, samples]
```

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
