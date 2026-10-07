[← SubRoformer](../../README.md)

# Training and evaluation

First complete [source installation](../../docs/installation.md). Run commands from the
repository root. The supplied checkpoint's extra-data training provenance is
recorded in [checkpoint details](../../docs/checkpoints.md).

## Dataset preparation

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

## Training

```bash
uv run accelerate launch --num_processes 1 --gpu_ids 0 --mixed_precision bf16 --module subroformer.training.train \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --data_path data/musdb18hq/train
```

For two GPUs:

```bash
uv run accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 --mixed_precision bf16 --module subroformer.training.train \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --data_path data/musdb18hq/train
```

Use `subroformer/configs/sbroformer_base_4stem.yaml` for joint separation. For a single
target, use `subroformer/configs/sbroformer_base_vocals.yaml`, `subroformer/configs/sbroformer_base_bass.yaml`,
`subroformer/configs/sbroformer_base_drums.yaml`, or `subroformer/configs/sbroformer_base_other.yaml`.
All five configs are self-contained. Single-stem configs preserve the original
batch size of 4 and disable random mixing; joint training uses batch size 2
and enables random mixing. Single-stem training still needs all four source
stems to construct mixtures. `--max_updates 2` limits a smoke run.

See the [config inventory](../configs/README.md#inventory) for the complete
list, including the perceptual-finetuned SiLU model and DSP2 comparison config.

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

## Resume

Training writes raw and EMA weights at completion and resumable Accelerate state
directories at `training.save_every_updates`. `--resume` requires such a complete
state directory; the supplied inference-only EMA file cannot resume optimization.

## Evaluation

Prepare aligned audio as described in [dataset preparation](#dataset-preparation).
Use the matching config/checkpoint pair listed in
[checkpoint details](../../docs/checkpoints.md); the example below evaluates the original Base model.

```bash
uv run python -m subroformer.training.evaluate \
  --config subroformer/configs/sbroformer_base_4stem.yaml \
  --checkpoint checkpoints/subroformer_base_4stem_1400000_ema.pt \
  --root data/musdb18hq/test --device cuda --precision bf16 \
  --hop_ratio 4 --batch_size 1 --output_json eval_results/full_test.json
```

The evaluator sums the resampled reference stems as model input, matching the
SBRoformer evaluation protocol; it does not read `mixture.wav`. It computes
`museval` SDR at 44.1 kHz. Each stem score is the median across songs of per-song
median window SDR; the joint summary averages the four stem medians. Distributed
evaluation shards whole songs and aggregates all rows before taking medians.

### Published test results

The [README checkpoint table](../../README.md#checkpoints-and-test-sdr) records the
two released EMA results. The original 1.4M EMA was evaluated on 2026-09-06; the perceptual
+200k EMA on 2026-09-11. Both use all 50 MUSDB18-HQ test songs, eight-process
whole-song sharding, BF16 inference, two-second segments at 48 kHz, Hamming
overlap-add, hop ratio 4, and batch size 1, with museval scoring at 44.1 kHz.
These are source EMA evaluations, not new evaluations of the FP16 exports or
Core ML artifacts. The Base model used MUSDB18-HQ plus 500 extra songs; this is not
a MUSDB-only result or a controlled architectural ablation.

## Acknowledgments

Thanks to Roman Solovyev (ZFTurbo) for
[Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
from which this project's dataset, DataLoader, dataset-type support, and
augmentation schema are derived.
