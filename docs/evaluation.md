[← SubRoformer](../README.md)

# Evaluation

First complete [source installation](installation.md) and prepare aligned audio
as described in the [training guide](training.md). Run commands from the
repository root. Use the matching config/checkpoint pair listed in
[checkpoint details](checkpoints.md); the example below evaluates the original Base model.

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

## Published test results

The [README checkpoint table](../README.md#checkpoints-and-test-sdr) records the
two released EMA results. The original 1.4M EMA was evaluated on 2026-09-06; the perceptual
+200k EMA on 2026-09-11. Both use all 50 MUSDB18-HQ test songs, eight-process
whole-song sharding, BF16 inference, two-second segments at 48 kHz, Hamming
overlap-add, hop ratio 4, and batch size 1, with museval scoring at 44.1 kHz.
These are source EMA evaluations, not new evaluations of the FP16 exports or
Core ML artifacts. The Base model used MUSDB18-HQ plus 500 extra songs; this is not
a MUSDB-only result or a controlled architectural ablation.
