# Core ML inference

PyTorch is the default backend, including for `--backend auto`. On supported
Macs, the app starts with MPS and the FP16 checkpoint; CLI automatic precision
uses FP16 on MPS. Core ML remains opt-in on macOS 15+ via `--backend coreml`
or `load_separator(device="auto", backend="coreml")`. It downloads the compiled
Base mask network from `archimickey/SubRoformer`, under
`coreml/base-4stem-fp16-v1/`. Custom config/checkpoint pairs require PyTorch.

This is a hybrid backend: the learned mask network runs in Core ML at FP16
with RMSNorm, RMSNorm2d, and GRN kept in FP32 for numerical stability;
complex-valued subband analysis, STFT, synthesis, and rolling overlap-add remain
in PyTorch/NumPy on CPU. Exact DSP buffers from the source checkpoint are bundled
in `dsp.safetensors`. No full PyTorch checkpoint is downloaded for this backend.
Core ML uses `CPU_AND_GPU`; this is not an Apple Neural Engine deployment, nor
a claim of improved speed. Core ML schedules supported operations itself.
The earlier FP32 artifact remains under `coreml/base-4stem-v1/`; it is not the
default and is not downloaded by the FP16 backend. DSP and public audio tensors
remain FP32; only the compiled neural network uses FP16. For the CLI, use
`--precision auto` or `--precision fp16`; requesting FP32/BF16 requires PyTorch.

The compiled shape is one 2-second stereo window at 48 kHz. Existing overlap-add
handles arbitrary file lengths and mono duplication. Batch requests are processed
one window at a time by Core ML. Custom contexts and weights require a new export;
they are not silently substituted with the Base artifact.

```sh
python inference.py --input song.wav --output_dir stems --backend coreml
python inference.py --input song.wav --output_dir stems --backend torch --device mps
```

HF downloads use its normal cache; `--local_files_only` and the app's offline
checkbox prohibit network access. Missing/corrupt artifacts and Core ML loading
errors are reported, not silently retried through another inference engine.
If a compiled artifact cannot load on a particular Mac, use `--backend torch`
in the CLI. The desktop app uses PyTorch by default.

## Reproduce the export

The tested converter environment is Python 3.11, PyTorch 2.7.0, NumPy 2.2.6,
and coremltools 9.0. Keep it separate from training/runtime dependencies:

```sh
uv venv --python 3.11 .venv-coreml
uv pip install --python .venv-coreml/bin/python -e . 'torch==2.7.0' 'numpy==2.2.6' 'coremltools==9.0'
.venv-coreml/bin/python export_coreml.py --validation-audio /path/to/stereo-music.wav
```

The exporter emits `mask.mlpackage`, `mask.mlmodelc`, `config.yaml`,
`dsp.safetensors`, and `conversion.json` under `artifacts/coreml-fp16/`.
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
