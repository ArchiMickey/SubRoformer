# Standalone preparation validation

## Apple Silicon / Metal inference

Verified on 2026-09-06 on arm64 macOS 26.6.2 with a fresh project `.venv`,
Python 3.11.15, and PyTorch 2.14.0. `torch.backends.mps.is_available()` is true.
FFmpeg 9.0.1 was installed for the broader evaluation/test imports.

- Downloaded the Base four-stem safetensors checkpoint at pinned HF revision
  `23497a6972e2b880552a42e10b99893cb3ff205b` into ignored `checkpoints/`.
  SHA-256 matches `c75764944269195db7ff435bfddc3472f569a454ad7e0d0b1f0537983944d228`.
- Strict checkpoint loading succeeds with 97,234,748 parameters on `mps:0`.
- Native MPS FP32 inference uses the unchanged model/DSP without enabling
  `PYTORCH_ENABLE_MPS_FALLBACK`. A real-checkpoint serving call processes a
  short stereo noise clip using the default 96,000-sample padded window and
  returns four finite float32 arrays shaped `[2, 4800]`.
- Offline CLI inference on a short mono tone produces four finite stereo
  FLOAT WAVs, each 4,800 samples at 48 kHz, under `outputs/mps-smoke/stems/`.
- Single- and four-stem tiny-model tests compare MPS and CPU output with
  `rtol=1e-3, atol=1e-5`, including batched overlapping windows and file output.
  Device-priority and unsupported-precision tests also pass.
- Full suite: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest`
  reports 103 passed, with one third-party audiomentations deprecation warning.
  `uv pip check --python .venv/bin/python` and `git diff --check` pass.

These are inference smoke checks, not quality/SDR or performance benchmarks.
MPS training and older PyTorch releases are not validated.

### Mixed-precision follow-up

The inference CLI and `separate_file` now default to `precision="auto"`:
BF16 when supported, otherwise FP16 on GPU; CPU retains FP32. Both MPS
BF16 and FP16 pass tiny-model forward checks and real-checkpoint two-second
smoke checks. Tests observe reduced-precision activations inside the network;
returned waveforms remain float32. The full suite reports 108 passed.

Full-track MPS BF16 inference on the downloaded MUSDB18-HQ test mixture
`PR - Oh No` (76.1945 seconds) finishes in 82.30 seconds including loading.
The existing FP32 results are preserved separately. Scoring the BF16 outputs
against checksum-verified reference stems with `training_utils.museval_sdr`
gives vocals -0.0017, bass 0.0004, drums 7.7214, other -1.6113 dB;
the four-stem mean is 1.5272 dB. This uses the original mixture file, not the
evaluator's sum of reference stems, and is not a full-test-set benchmark.
Changing precision does not resolve the low single-track SDR.
Local outputs and score report: `outputs/musdb18hq-pr-oh-no-bf16/`.

## Original preparation

Initial four-stem-only snapshot validated on 2026-09-06 using Python 3.11.14 and the existing machine's installed
ML dependencies through a separate virtual environment. This was not a fresh
download-and-install test of every dependency.

| Check | Result |
| --- | --- |
| Public source unit/integration suite | 60 passed |
| Extracted source-distribution suite | 60 passed |
| Two-rank CPU/Gloo random-mixing loader | Passed; distinct rank samples and correct four-stem tensors |
| Two-rank CPU/Gloo ordinary distributed loader | Passed |
| Two-rank distributed evaluation aggregation | Passed |
| Synthetic training CLI | One optimizer update, finite loss/gradient norm, changed raw versus EMA weights, saved update state |
| Synthetic file-inference tests | Mono resampling, short clips, raw/EMA checkpoint loading, safe output handling |
| Actual Base EMA checkpoint | Strict state-dict load: all keys matched; 97,234,748 parameters |
| Actual checkpoint CUDA/bf16 inference | Four finite stereo WAVs, 120,000 samples each at 48 kHz |
| Checkpoint copy | Byte-for-byte comparison and documented SHA-256 matched |
| Source model/audio config compatibility | Exact effective-config match with the checkpoint's training config |
| Config inventory | Only the self-contained `sbroformer_base_4stem.yaml` |
| Wheel installation | All runtime modules imported from a separate wheel installation outside source checkouts |
| Distribution content audit | No private dataset implementation, machine data paths, weights, or audio |

The model implementation and shared training utilities are unchanged from the
source snapshot. Public changes remove private dataset dispatch/cache handling,
external comparison-model loading, and unpublished configuration dependencies;
they add standalone inference and focused tests.

No full MUSDB evaluation was rerun for this export. No Base-model training run
was launched. No commit, push, visibility change, license grant, or checkpoint
upload was performed. Before release, review adapted-code provenance, applicable
license terms, checkpoint distribution terms, and distribution contents.

## Local safetensors CPU/GPU inference check

Verified on 2026-09-06 with the actual 1.4M-update Base four-stem EMA
safetensors checkpoint, PyTorch 2.9.1+cu128, and GPU 0 (RTX 4090).
The inference CLI used local default config/weights with
`HF_HUB_OFFLINE=1 --local_files_only`, batch size 1, and hop ratio 4.
Input was 2.25 seconds of synthetic stereo tones/noise at 48 kHz, exercising
multiple overlapping two-second windows and tail padding.

| Device / precision | CLI wall time, including model loading | Result |
| --- | ---: | --- |
| CPU / fp32, 4 OMP/MKL threads | 15.358 s | Passed |
| CUDA:0 / fp32 | 4.913 s | Passed |
| CUDA:0 / bf16 | 4.916 s | Passed |

All runs emitted four finite, nonzero stereo FLOAT WAVs with exactly 108,000
samples at 48 kHz. Maximum absolute GPU-fp32 versus CPU-fp32 difference across
stems was 2.384e-5; bf16 versus CPU-fp32 was 6.056e-3. These are synthetic-input
smoke checks, not SDR measurements or warmed-up throughput benchmarks.
GPU memory returned to its idle baseline after completion.

Local evidence: `outputs/inference_device_check_20260906_225245/report.json`,
with per-device logs and output WAVs alongside it. No HF download was needed.

## MUSDB18-HQ full-song precision check

Verified on 2026-09-06 using the complete test song `AM Contra - Heart Peripheral`
(210.106 seconds), the local 1.4M-update EMA safetensors checkpoint, and the
Base four-stem config. The checkpoint SHA-256 matches `docs/checkpoints.md`.
Each precision ran the actual `inference.py` CLI on `mixture.wav`, on GPU 0
(RTX 4090), with two-second windows, hop ratio 4, batch size 1, and offline loading.
Saved WAVs were reloaded and scored with `training_utils.museval_sdr` against
reference stems: resampling from 48 kHz to 44.1 kHz, median window SDR per stem,
then arithmetic mean across the four stems.

| Precision | Vocals | Bass | Drums | Other | Mean SDR (dB) | Inference wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 13.6907 | -2.1905 | 11.0031 | 0.5059 | 5.7523 | 41.142 s |
| bf16 | 13.6818 | -2.6582 | 11.0020 | 0.5136 | 5.6348 | 27.883 s |
| fp16 | 13.6921 | -2.2106 | 11.0045 | 0.5058 | 5.7480 | 27.897 s |

All three CLI processes exited successfully and emitted four finite stereo
FLOAT WAVs at 48 kHz, each with 10,085,077 samples per channel. Times include
loading and writing, but exclude SDR scoring; these are not warmed-up benchmarks.
Both mixed-precision modes passed this song-level runtime check. Bass SDR is
low even in fp32; bf16 also lowers bass SDR by 0.4676 dB relative to fp32.
This is one test song, not the 50-song test-set median or proof of precision
equivalence across the dataset. GPU memory returned to its idle baseline.

Evidence and reproducible runner: `outputs/musdb_precision_check_20260906/`;
`report.json` records commands, hashes, timings, SDR, shapes, and differences
against fp32. Precision subdirectories contain the separated WAVs.

## Full MUSDB18-HQ test evaluation with mixture.wav

On 2026-09-06, reran all 50 test songs using the local 1.4M-update EMA
safetensors checkpoint, Base four-stem config, and `mixture.wav` as input.
The initial reference-sum run was stopped and its partial results were not used.
That run used an evaluator that read the actual mixture file; a regression test deliberately
makes it differ from the reference sum. All 15 evaluation tests and the two-rank
distributed aggregation smoke test passed before relaunch.

The completed run used eight RTX 4090 GPUs (0-7), bf16, two-second windows,
hop ratio 4, and inference batch size 1. Whole songs were sharded across ranks;
all per-song rows were gathered before computing global per-stem medians.
SDR uses the existing `training_utils.museval_sdr` 44.1-kHz scoring protocol.

| Vocals | Bass | Drums | Other | Mean of stem medians |
| ---: | ---: | ---: | ---: | ---: |
| 12.6902 | 11.5072 | 11.9227 | 9.9364 | 11.5141 |

Completion audit verified 50 unique expected song names, 200 finite stem scores,
exact config/checkpoint/data paths, matching recomputed medians, and exit code 0.
The first song's scores exactly match the earlier saved-WAV bf16 check.
Elapsed time was 582 seconds; no evaluation processes remained and all eight
GPUs returned to idle memory. NCCL printed a process-group cleanup warning on
exit, but all workers exited successfully and the complete result was verified.

Artifacts: `outputs/musdb_full_ddp_mixture_20260906/full_test.json`, `audit.json`,
`run.log`, and the exact launch command in `run.sh` in the same directory.
These are newly measured mixture-file results, not the historical reference-sum
checkpoint-provenance scores in `docs/checkpoints.md`.

On 2026-09-07, evaluation was switched back to the sum of resampled reference
stems to match SBRoformer. Result JSONs identify this as
`mixture_source: reference_sum`. The full test set was not rerun after this
switch; the mixture-file results above remain historical measurements, not
results for the current default. File inference still uses the supplied audio.
