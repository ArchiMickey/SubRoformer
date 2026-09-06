# Base four-stem checkpoint: 1,400,000 updates

- File: `checkpoints/subroformer_base_4stem_1400000_ema.pt`
- Config: `configs/sbroformer_base_4stem.yaml`
- Weights: EMA, decay 0.999; original wrapper contains `decay` and `model`.
- Saved training state: `global_update=1400000`, `epoch=1399`.
- SHA-256: `4085bca04659063402b2e2f0af3c497e0071d6bdfa84e672c00e34bcbea0d4f9`.
- Output order: vocals, bass, drums, other; stereo, 48 kHz.

## Safetensors inference copy

- File: `checkpoints/subroformer_base_4stem_1400000_ema.safetensors`
- SHA-256: `c75764944269195db7ff435bfddc3472f569a454ad7e0d0b1f0537983944d228`.
- All 404 tensor names, dtypes, shapes, and values match the original EMA file exactly.
- Strict loading with the four-stem Base config succeeds; a CPU float32 forward
  on 4,800 stereo samples returns finite `[1, 4, 2, 4800]` output.
- Metadata records the source SHA-256, EMA decay, config path, and provenance
  update/epoch counters. The source EMA wrapper itself only has `decay` and
  `model`; the counters come from the saved-training-state provenance above.

Use the same config, sample rate, and output order as the PyTorch copy. This is
an inference-weight export, not resumable optimizer state or a new SDR evaluation.

## Original artifact provenance

This is a byte-for-byte copy of the evaluated EMA artifact, renamed for the
standalone release. It is not the non-EMA model or a complete resumable training
state. Optimizer state, RNG states, private training configs, manifests, and
dataset implementations are excluded.

Training used MUSDB18-HQ plus 500 extra songs. Training on the public
generic-folder recipe alone is not an exact reproduction of this extra-data
checkpoint.

The original full-test audit on 2026-09-06 recorded 50 unique MUSDB18-HQ test
songs and 200 finite stem scores. With EMA weights, bf16, two-second segments,
hop ratio 4, and inference batch size 1, per-stem median museval SDR was:

| Vocals | Bass | Drums | Other | Mean of stem medians |
| ---: | ---: | ---: | ---: | ---: |
| 12.6863 | 11.4961 | 12.2267 | 9.9445 | 11.5884 |

These are previously measured results, not a full-test rerun of this export.
The checkpoint's effective model config is checked against the source config;
public training paths and evaluation defaults deliberately differ.

Weights are locally present but ignored by Git. The configured HF destination
is [archimickey/SubRoformer](https://huggingface.co/archimickey/SubRoformer),
currently private. The safetensors copy and matching YAML were uploaded together
at revision `23497a6972e2b880552a42e10b99893cb3ff205b`. The remote checkpoint
is 414,655,864 bytes and its LFS SHA-256 matches the safetensors checksum above.
Inference supports local-first downloading from this repository using an
authorized HF login or `HF_TOKEN`; repository visibility was not changed.

Fresh-cache downloads were verified against both local files (config byte
equality and checkpoint SHA-256). The inference CLI then ran from an empty
working directory with no config/checkpoint arguments and offline cache-only
loading, producing four finite 48 kHz stereo stems of 4,800 samples each.

- [Safetensors checkpoint](https://huggingface.co/archimickey/SubRoformer/blob/23497a6972e2b880552a42e10b99893cb3ff205b/checkpoints/subroformer_base_4stem_1400000_ema.safetensors)
- [Matching YAML config](https://huggingface.co/archimickey/SubRoformer/blob/23497a6972e2b880552a42e10b99893cb3ff205b/configs/sbroformer_base_4stem.yaml)
