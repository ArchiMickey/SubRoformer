![SubRoformer — Music Source Separation](app/assets/subroformer-banner.png)

# SubRoformer

This is the official repository for **SubRoformer: Subband Time-Domain Representation for Music Source Separation**. SubRoformer separates music mixtures into four stems: **vocals, bass, drums, and other instruments**.

![SubRoformer Framework](app/assets/subroformer-framework.png)

## Results

The following table reports the source separation performance of the released checkpoints on MUSDB18-HQ.

| Checkpoint | Vocals | Bass | Drums | Other | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base (1.4M updates, EMA) | 12.6863 | 11.4961 | 12.2267 | 9.9445 | **11.5884** |
| Perceptually fine-tuned (+200k updates, EMA, default) | 12.6424 | 11.1357 | 12.0518 | 9.7584 | **11.3971** |

The models are trained on **100 songs from MUSDB18-HQ** and **500 additional songs from an internal dataset**. Evaluation is conducted on the **MUSDB18-HQ test set (50 songs)** using `museval` SDR (dB). The SDR for each stem is computed as the median of the per-song median SDRs, and the Mean SDR is the average of the four stem medians. For further details, see the [evaluation protocol](subroformer/training/README.md#published-test-results).

The **perceptually fine-tuned checkpoint** prioritizes perceptual quality with a slight reduction in Mean SDR (**0.1913 dB**). It is fine-tuned for an additional **200k updates** from the Base checkpoint (1.4M updates) and uses the corresponding **SiLU** configuration instead of the original **GEGLU** configuration. See [checkpoint files and configs](docs/checkpoints.md) for details.

Both checkpoints are available on [Hugging Face](https://huggingface.co/archimickey/SubRoformer).

## 1. Installation

Run the following commands from the repository root using Python 3.11 and a PyTorch installation compatible with your hardware:

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e .
```

See the [installation guide](docs/installation.md) for detailed environment setup instructions.

## 2. Inference

### 2.1 Command-Line Inference (CLI)

Run source separation using the released perceptually fine-tuned checkpoint, which is automatically downloaded if not available locally:

```bash
uv run python -m subroformer.inference \
  --input demo/song.wav \
  --output_dir outputs/song
```

The CLI automatically selects the available device in the following order: **CUDA → MPS → CPU**. You can manually specify a device using `--device cuda`, `--device mps`, or `--device cpu`.

The separated stems are saved as `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav` at **48 kHz**, with automatic resampling of the input audio when necessary.

### 2.2 Python Inference

Load the released checkpoint once and reuse the model to separate multiple songs:

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

### 2.3 SubStemStudio App

Demo using [Netrum & Halvorsen - Phoenix](https://www.youtube.com/watch?v=yH88qRmgkGI&list=RDyH88qRmgkGI&start_radio=1):

<div align="center">

https://github.com/user-attachments/assets/45b3fa51-0370-4635-a93b-738856479956

</div>

**SubStemStudio** provides a graphical interface for music source separation. Users can import a song, separate it into four stems, listen to the results as separation progresses, and export the stems as **48 kHz float32 WAV files**. The app also provides synchronized spectrogram visualization and individual playback controls for each stem.

The app supports **macOS, Windows, and Linux**. Core ML is used by default on macOS 15+, while the PyTorch backend supports CUDA, Apple Silicon (MPS), and CPU.

See the [SubStemStudio installation and usage guide](app/README.md) for details.

## 3. Training

This section describes how to train SubRoformer from scratch using the MUSDB18-HQ dataset.

### 3.1 Download the Dataset

Download and extract the **MUSDB18-HQ** dataset using:

```bash
bash scripts/download_musdb18hq.sh
```

After extraction, the dataset is organized as follows:

```text
data/
└── musdb18hq/ (~30 GB)
    ├── train/ (100 songs)
    │   ├── A Classic Education - NightOwl/
    │   │   ├── bass.wav
    │   │   ├── drums.wav
    │   │   ├── mixture.wav
    │   │   ├── other.wav
    │   │   └── vocals.wav
    │   └── ...
    └── test/ (50 songs)
        ├── Al James - Schoolboy Facination/
        │   ├── bass.wav
        │   ├── drums.wav
        │   ├── mixture.wav
        │   ├── other.wav
        │   └── vocals.wav
        └── ...
```

### 3.2 Train the Model

To train SubRoformer on a single GPU, run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --module subroformer.training.train \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --data_path data/musdb18hq/train
```

For multi-GPU training, checkpoint management, and resuming interrupted training, see the [training guide](subroformer/training/README.md).

### 3.3 Inference with a Trained Checkpoint

To run inference using a checkpoint from your own training run:

```bash
uv run python -m subroformer.inference \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --checkpoint checkpoints/my_run/model_ema.pt \
  --input demo/song.wav \
  --output_dir outputs/my_run
```

## Documentation

| Topic | Guide |
| --- | --- |
| App installation, usage, and native builds | [App Guide](app/README.md) |
| Python/source installation | [Installation Guide](docs/installation.md) |
| Package modules and imports | [Package Overview](subroformer/README.md) |
| Model configurations and inheritance | [Configuration Guide](subroformer/configs/README.md) |
| Training and dataset preparation | [Training Guide](subroformer/training/README.md) |
| Evaluation and SDR protocol | [Evaluation Guide](subroformer/training/README.md#evaluation) |
| CLI and Python inference | [Inference Guide](subroformer/inference/README.md) |
| DSP backends and defaults | [Frontend Guide](subroformer/frontend/README.md) |
| Checkpoint downloads and provenance | [Checkpoint Guide](docs/checkpoints.md) |
| Core ML setup and export | [Core ML Guide](subroformer/inference/README.md#core-ml-inference) |
| Tests and validation | [Testing Guide](docs/testing.md) |
| Historical inference optimization experiments | [Inference Optimization](docs/inference_optimization.md) |

## Acknowledgments

We thank **Roman Solovyev (ZFTurbo)** for developing [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training), from which parts of the dataset handling and DataLoader code in this repository are derived.