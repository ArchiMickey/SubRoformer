# Core ML inference

The desktop app defaults to Core ML on macOS 15+ and PyTorch elsewhere, with
MPS selected when available. CLI/API defaults remain PyTorch, including for
`--backend auto`; CLI automatic precision uses FP16 on MPS. Core ML is selected
explicitly in the CLI/API on macOS 15+ via `--backend coreml`
or `load_separator(device="auto", backend="coreml")`. It downloads the compiled
200k perceptual-finetuned Base mask network from `archimickey/SubRoformer`, under
`coreml/base-4stem-perceptual-200k-fp16-v1/`. Its matching config is
`configs/sbroformer_base_4stem_silu.yaml`: SiLU fusion is required, not GEGLU.
Both PyTorch and Core ML default to this finetuned model; backend selection
is unchanged. Custom config/checkpoint pairs require PyTorch.

This is a hybrid backend: the learned mask network runs in Core ML at FP16
with RMSNorm, RMSNorm2d, GRN, and the small TimeMix blocks kept in FP32
for numerical stability, including quiet tails;
complex-valued subband analysis, STFT, and synthesis run in PyTorch, on CPU by
default. With PyTorch 2.14+, explicitly select `--device mps` to run this DSP on
the GPU as well. Rolling overlap-add and public audio tensors remain on CPU.
Exact DSP buffers from the source checkpoint are bundled
in `dsp.safetensors`. No full PyTorch checkpoint is downloaded for this backend.
Core ML uses `CPU_AND_GPU`; this is not an Apple Neural Engine deployment, nor
a claim of improved speed. Core ML schedules supported operations itself.
The earlier Base artifacts remain under `coreml/base-4stem-v1/` (FP32) and
`coreml/base-4stem-fp16-v1/` (FP16); neither is the default or downloaded by
the current backend. DSP and public audio tensors
remain FP32; only the compiled neural network uses FP16. For the CLI, use
`--precision auto` or `--precision fp16`; requesting FP32/BF16 requires PyTorch.

The compiled shape is one 2-second stereo window at 48 kHz. Existing overlap-add
handles arbitrary file lengths and mono duplication. Batch requests are processed
one window at a time by Core ML. Custom contexts and weights require a new export;
they are not silently substituted with the Base artifact.

```sh
python inference.py --input song.wav --output_dir stems --backend coreml
python inference.py --input song.wav --output_dir stems --backend coreml --device mps
python inference.py --input song.wav --output_dir stems --backend torch --device mps
```

The MPS DSP path uses the **same compiled checkpoint**, without re-export or
reduced precision: DSP remains FP32. Features and masks still cross the CPU
Core ML API boundary. `--device auto` retains CPU DSP for compatibility;
PyTorch 2.7 fails the MPS inverse-STFT path. Explicit MPS requests require
PyTorch 2.14+ and available MPS hardware and are never silently retried on CPU.
The API equivalent is `load_separator(backend="coreml", device="mps")`, or
`load_coreml_separator(dsp_device="mps")`. In the desktop app, select **Core ML**
as the backend and **mps** as the device. The app's Auto device also selects MPS
when available. The app defaults to Core ML on macOS 15+; PyTorch can still
be selected manually, including for custom config/checkpoint pairs.
The app isolates Core ML and its DSP in a spawned process to prevent native
prediction calls from blocking Python playback callbacks. CLI/API inference
continues in-process. This changes scheduling, not model weights or precision.

Reproduce the CPU-DSP versus MPS-DSP comparison with a local artifact and cached
PyTorch checkpoint (use the application runtime, not the older converter venv):

```sh
.venv/bin/python benchmarks/benchmark_coreml_dsp.py \
  --validation-audio /path/to/stereo-music.wav \
  --artifact artifacts/coreml-perceptual-200k-fp16 \
  --output artifacts/coreml-dsp-benchmark.json
```

The benchmark records load/first-forward and warmed timings and checks noise,
silence, and beginning/middle/end music waveforms against FP32 PyTorch and the
unchanged CPU-DSP Core ML reference. MPS results are not bitwise identical;
the existing 0.1% relative RMS / 1e-6 absolute-floor threshold still applies.

HF downloads use its normal cache; `--local_files_only` and the app's offline
checkbox prohibit network access. Missing/corrupt artifacts and Core ML loading
errors are reported, not silently retried through another inference engine.
The app displays reconstructed artifact bytes, percentage, and an estimated
download ETA. File counts and Xet network-transfer bytes are not treated as model
bytes; ETA starts estimating after byte progress is observed. Model loading is
shown as a separate phase after the download completes.
If a compiled artifact cannot load on a particular Mac, use `--backend torch`
in the CLI, or select PyTorch in the desktop app.

## Reproduce the export

The tested converter environment is Python 3.11, PyTorch 2.7.0, NumPy 2.2.6,
and coremltools 9.0. Keep it separate from training/runtime dependencies:

```sh
uv venv --python 3.11 .venv-coreml
uv pip install --python .venv-coreml/bin/python -e . 'torch==2.7.0' 'numpy==2.2.6' 'coremltools==9.0'
.venv-coreml/bin/python export_coreml.py --validation-audio /path/to/stereo-music.wav
```

The exporter emits `mask.mlpackage`, `mask.mlmodelc`, `config.yaml`,
`dsp.safetensors`, and `conversion.json` under `artifacts/coreml-perceptual-200k-fp16/`.
Validation compares the compiled mask and complete waveforms against FP32
PyTorch loaded from the published FP16 checkpoint, using noise, silence, and
beginning/middle/end music chunks. The report includes checkpoint SHA-256 and numerical errors;
it is not a full MUSDB SDR evaluation or cross-machine compatibility guarantee.
Waveform validation permits at most 0.1% RMS error, with a 1e-6 absolute RMS
floor (-120 dBFS) for near-silent outputs. Both relative and absolute errors
are recorded, including quiet tails where relative error alone is misleading.
The trace fixes shape-dependent branches intentionally for this input shape.

Pass `--upload` to publish only after validation succeeds. It uploads to the
default `archimickey/SubRoformer` repository using the existing HF login.
Publish the compiled directory,
config, DSP buffers, and validation report together in one HF commit. The
portable `.mlpackage` may also be published for future recompilation; the app
downloads only the compiled runtime files. Build the app on macOS with
`python app/build.py` after installing `app/requirements-build.txt`.

See Apple's [compiled model prediction documentation](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html).
