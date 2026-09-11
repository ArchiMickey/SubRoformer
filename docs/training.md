[← SubRoformer](../README.md)

# Training

First complete [source installation](installation.md). Run commands from the
repository root. The supplied checkpoint's extra-data training provenance is
recorded in [checkpoint details](checkpoints.md).

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

## Acknowledgments

Thanks to Roman Solovyev (ZFTurbo) for
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
from which this project's dataset, DataLoader, dataset-type support, and
augmentation schema are derived.
