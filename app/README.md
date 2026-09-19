# SubStemStudio

A native, local desktop GUI using PySide6 and pyqtgraph. App-specific source,
dependencies, and tests live in `app/`. The app and CLI share the inference
engine, model loading, and device helpers.

SubStemStudio is the app name; the SubRoformer model, Python package, and
Hugging Face repository names are unchanged. The default export parent folder
is `~/Music/SubStemStudio`; existing exports are not moved.

### Memory and streaming

The app and CLI accumulate predictions in a rolling overlap-add buffer and
write finalized float32 samples directly to WAV files. They do not retain
full-track stem arrays or all overlapping predictions. The input waveform is
still decoded/resampled in memory; model and batch memory are unchanged.

Playback uses temporary disk-backed PCM and bounded range reads. Temporary playback files close on reset or exit.
The GUI acknowledges each audio delivery before the worker continues, avoiding
an unbounded queue when the interface is slower than inference. Export and
temporary-storage disks must have enough free space for the audio.

WAVs are written during separation; an interrupted or failed run can leave an
incomplete export folder. Only successful finalization reports completion.
The Python array-returning API remains available; use
`separate_audio(..., on_chunk=callback, collect_output=False)` for bounded
output storage. Overlap weights, accumulation order, and output dtype are
unchanged.

## Install and launch

From the repository root, with Python 3.11 or 3.12 and an appropriate PyTorch
installation for your platform:

```sh
python -m pip install -e .
python -m pip install -r app/requirements.txt
python -m app
```

With this repository's uv environment, use:

```sh
uv pip install --python .venv/bin/python -r app/requirements.txt
uv run --no-sync python -m app
```

The same source targets macOS, Windows, and Linux (with a graphical desktop).

The packaged Mac app requires **macOS 15 (Sequoia) or later on Apple Silicon**,
including in CPU mode, because of its bundled runtime dependencies.
Use the **Help** button for keyboard shortcuts and MPS troubleshooting. If you
see `ValueError: MPS is unavailable`, check **Apple menu → About This Mac** and
install available macOS updates. MPS also requires supported GPU hardware and
an MPS-enabled PyTorch build. Select **cpu** in Device to run without MPS
(CLI: `--device cpu`); this does not lower the packaged app's macOS minimum.
CPU is available on all three; CUDA requires a compatible NVIDIA/PyTorch setup;
MPS requires a supported Mac. A native binary build recipe is included below;
no prebuilt binaries are provided. Windows/Linux and full-size GPU inference require
platform-specific validation. Linux may need system Qt/XCB libraries.

## Use

1. Browse for a mixture, or drop one local audio file anywhere in the window
   to fill the Mixture path and preview its spectrogram immediately. Drops accept
   WAV, FLAC, AIFF/AIF, OGG, and MP3 files and are disabled while a preview or
   separation is running. File decoding
   uses SoundFile; supported formats depend on its libsndfile installation.
2. Choose an export parent folder, device (including `cuda:1`),
   batch size, and overlap ratio. Model chunk size stays in the model config.
   The app defaults to Core ML on macOS 15+ and PyTorch elsewhere, and initially
   selects CUDA → MPS → CPU, so supported Macs start with MPS. DSP2 uses its
   optimized Metal modulation and filter FFT cache on MPS by default for both
   neural backends. On the PyTorch path,
   precision is fixed by the checkpoint: the default weights stay FP16.
   Select an FP32 checkpoint for FP32 inference. The info pane's PRECISION card
   shows the loaded backend and precision throughout separation and after completion.
   Autocast follows the weight dtype; RMSNorm
   statistics, complex DSP, and audio accumulation use FP32 where needed.
   MPS linear layers add bias explicitly because virtualized Metal GPUs can omit
   fused linear biases. CPU/CUDA retain the standard linear implementation;
   no fallback environment variable or FP32 conversion of all weights is required.
3. Optionally select a matching config/checkpoint pair. Blank fields use the
   200k perceptual-finetuned FP16 checkpoint
   (`subroformer_base_4stem_1400000_ema_perceptual_finetuned_200000_fp16.safetensors`)
   with `sbroformer_base_4stem_silu.yaml`. Core ML uses the matching validated
   `coreml/base-4stem-perceptual-200k-fp16-v1/` artifact.
   Missing artifacts may download from Hugging Face. The following
   config/checkpoint fallback rules apply to the PyTorch path.
   Missing config and checkpoint files fall back independently to the published
   Base artifacts, including missing explicit paths. Existing files are preserved;
   directories and corrupt files fail rather than being replaced. Keep custom
   configs/checkpoints compatible with the Base artifact when only one is missing.
   The current repo is public: anonymous full downloads were verified on 2026-09-06.
   No `HF_TOKEN` or cached login is required; private/gated repos still need access.
   Offline mode uses local files or the HF cache and never starts a download.
4. Click **Separate stems**. Stems start with a gray hatched unprocessed
   region, distinct from genuine silence. A pale yellow boundary
   advances as finalized spectrogram columns arrive; unfinished regions remain
   masked after cancellation or failure. The mixture is already available.
   Larger batches reduce update frequency and consume more memory.
5. Open the exported stems after completion. Each run creates a unique child
   directory and writes numbered float32 WAVs without overwriting prior runs.

Before model loading, the progress bar shows each missing artifact's percentage
and completed/total MB. Cache lookup and connection setup use an indeterminate
bar; cached artifacts complete immediately. Xet progress measures reconstructed
file bytes (not compressed network bytes), so it can advance in large chunks.
After downloads, model loading is indeterminate, then separation starts at 0%.

The separation progress bar shows the percentage of audio finalized. Elapsed
separation time and ETA update after each inference batch. ETA uses average
time per completed overlap-add window, including the shorter final batch;
it stays “estimating” until the first batch completes. Model loading/downloads,
input preparation, and WAV export are separate phases and excluded from ETA.
Warm-up and variable device load can make early estimates inaccurate. The bar
reaches 100% when separation finishes; **Saving stems** remains visible until
export completes. Cancel/failure clears the ETA while retaining progress.

The status pane also shows **RTF (real-time factor)**: elapsed separation time
divided by finalized audio duration. **RTF ≤ 1** means real-time speed or faster;
**RTF > 1** means slower than real-time. It updates after each batch, counts
audio duration once (not once per stem), and excludes loading and WAV export.
Until the first batch it shows `RTF —`; incomplete runs are labeled “so far”.
This measures progressive file-separation throughput, not live audio latency.

### Listen while separating

The **Mixture, Vocals, Bass, Drums, Others** tabs control playback.
**Mixture** turns all stems on. Clicking a stem toggles only that stem; disabling one also clears the
Mixture indicator. Playback sums only enabled stems when a subset is on. All-on plays the original
mixture directly, without double-counting it, and works before separation finishes.
Unavailable stems are disabled for single-stem or nonstandard models.
**Shift-click a stem** or press **Shift+2–5** (Vocals, Bass, Drums, Others) to
listen to only that stem, retaining the current timestamp and pause/play state.
**Mute all** turns every stem off without pausing: silent audio
keeps the timeline advancing to the end of the mixture. Enable stems or Mixture
to hear audio again at the current timestamp. Muting while paused stays paused.

Click any spectrogram to seek and play at that timestamp **without changing the
enabled stems**. Tab changes retain the current timestamp and pause/play state;
turning everything off keeps playback running silently. If a newly enabled stem is not finalized
at the current position, playback stops rather than jumping backward.
Dragging pans, scrolling zooms, and teal playheads share the playback position.
**Space** pauses/resumes; **1–5** toggle the corresponding tabs (Mixture selects
all); **backtick (`)** mutes all stems. The separate **Invert (~)** button or
**Tilde (~)** shortcut inverts the enabled stems,
so Mixture becomes all muted, and all muted becomes Mixture.
Stem buttons show their keys. Text and numeric fields keep normal keyboard entry. Holding a shortcut
does not repeatedly toggle it.

Wheel and trackpad gestures over configuration fields or the divider beside them
scroll the configuration pane without changing field values.

Only finalized waveform samples are playable. Playback can consume new chunks
as they arrive, but stops when it catches the latest available audio (or the
track end). It does not automatically restart after reaching that edge: press
Resume once more audio arrives, or click an earlier point. Clicking an
unseparated region places the playhead at the available edge without playing
uncomputed silence. New input/run and window close stop the previous audio.
After cancellation, already finalized audio remains playable.

Playback uses Qt Multimedia and the default system output device, with no new
dependency. If the source format is unsupported, playback uses the device's
preferred format, with polyphase resampling, mono averaging or channel repetition,
and float32/uint8/int16/int32 PCM conversion as needed. Missing or unusable devices show a playback-only error and do
not stop separation. Stems retain unclipped float32 audio; only the final
playback sum is clipped to [-1, 1]. Exported WAVs are unchanged. Playback retains
finalized stem and mixture PCM in temporary files.

### Theme and spectrogram colors

The interface uses [Catppuccin Macchiato](https://catppuccin.com/palette/#macchiato)
with matching button and spectrogram-frame accents: Mixture is mauve, Vocals pink,
Bass blue, Drums peach, and Others green. Mute all is red and Invert is yellow,
with separate tooltips for their actions. Enabled audio buttons fill with their
accent when selected. Spectrogram titles and selected frame borders match their
buttons; the spectrogram images retain the Magma colormap and fixed −90 to 0 dBFS
range during separation.
A teal line marks playback.

Drag the glass-style vertical divider to resize the configuration pane, or the
horizontal divider above playback to resize the status pane. Both panes retain
minimum usable sizes; resizing preserves the plots and playback position.
The dividers use translucent gradients and hover highlights, not native OS blur.

### Spectrogram layouts

The **Spectrogram layout** selector changes the arrangement without resetting
playback, zoom, or separation progress:

- **Stacked:** full-width plots with vertical scrolling.
- **Grid:** two columns with scrolling when needed.
- **Fit all:** automatically sized grid showing the mixture and every stem
  together, without vertical scrolling. Cards resize with the window; use a
  larger window for more spectral detail.

These are layout options, not alpha-blended spectrogram layers. Click-to-play
works in every layout; the selected plot keeps its playhead.

The UI runs inference, decoding, and spectral calculations in a worker thread.
Cancel takes effect between operations/batches, not inside an active model call
or download. Closing during work requests cancellation and automatically closes
the app when the worker finishes. No stems are exported for cancelled inference; a disk failure
during export may leave an incomplete run directory reported in the UI.

## Streaming semantics

### Live memory readings

The footer refreshes every second on a background thread for the selected
inference device (`auto` displays its resolved device, including CUDA index).
Values use decimal **GB = 1,000,000,000 bytes**:

- **Core ML (CPU or MPS DSP):** combined resident RAM (RSS) of the app and
  its child processes, including the isolated inference worker. Shared pages
  can be counted more than once; this is not GPU-only usage or an exact physical
  footprint. The percentage is whole-system unified RAM usage.

For the **PyTorch** backend:

- **CPU:** app resident RAM (RSS), including shared pages; whole-system RAM
  percentage is `(total − available) / total`, as reported by psutil.
- **CUDA:** app **PyTorch reserved** allocator memory, including cached blocks;
  whole-GPU used percentage from CUDA free/total memory includes other processes.
  The app value excludes context/driver and non-PyTorch allocations, so it is
  not an exact full-process GPU footprint and can differ from `nvidia-smi`.
- **MPS:** Metal driver allocations for this process, including caches;
  the percentage is **whole-system unified RAM usage**, not GPU-only memory.
  CPU RSS and Metal allocations overlap and must not be added together.

Hover over the reading for its definition. Invalid devices or unavailable
queries show `N/A`, not zero. Device selection changes the monitor; monitoring
does not change inference settings. Reading CUDA global memory may initialize
its runtime context. The app extra now includes `psutil`; reinstall
`app/requirements.txt` when upgrading an existing app environment.

Definitions: [PyTorch CUDA memory](https://docs.pytorch.org/docs/stable/cuda),
[Metal process allocations](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory),
and [psutil memory metrics](https://psutil.readthedocs.io/stable/index.html).

### Finalized output

`streaming.py` re-exports the shared inference engine's rolling Hamming
overlap-add. A callback receives only the prefix
that future overlapping windows cannot affect; the last callback flushes the
tail. The final audio is tested against a frozen pre-optimization engine.
This is progressive file inference, not real-time microphone processing.

Spectrograms use a 1024-sample Hann window, channel RMS magnitude (no stereo
phase cancellation), linear frequency, a shared −90 to 0 dBFS color scale,
and at most 1200 time columns. Frames appear only when their waveform is final;
the last frames use zero padding. Display overviews are time-subsampled for
long files and are not a full-resolution analysis export. The input waveform
still resides in RAM, so very long recordings need adequate input memory.
Scroll vertically through stems and zoom/pan horizontally; stem time axes link
to the mixture. Selecting a new file resets the preview.

## Tests

From the repository root:

```sh
python -m pip install pytest
python -m pytest app/tests
```

Tests cover single/joint outputs, short/tail audio, overlapping/non-overlapping
windows, batch sizes, cancellation, incremental spectrogram equivalence, silent
and anti-phase inputs, blank canvas updates, and tiny real-model worker/export.
GUI tests use Qt's offscreen backend and skip when GUI extras are absent.
Playback tests use a fake sink for deterministic seek/pause/frontier/device-error
checks, plus the real worker's emitted audio is compared with exported WAVs.
Hardware listening and packaged Qt Multimedia playback need target-OS testing.

## Build native binaries

The app uses `app/assets/subroformer-icon.png` and matching `.icns`/`.ico`
assets: the original logo on a white rounded-square tile with transparent
outer corners. Regenerate them with `python app/create_icon.py` (macOS also
generates ICNS via `iconutil`). The unframed `subroformer-logo.png` stays unchanged.

Use a clean Python 3.11/3.12 environment on the target OS. Install the intended
PyTorch build first (CPU/CUDA for Windows, native arm64 for Apple Silicon), then
from the repository root:

```sh
python -m pip install -e .
python -m pip install -r app/requirements-build.txt
python app/build.py
```

The build uses `app/SubStemStudio.spec`. The macOS bundle identifier remains
`com.archimickey.subroformer` for continuity with earlier builds.

- **macOS:** `app/dist/SubStemStudio.app` and `app/dist/SubStemStudio.dmg` — the disk
  image contains the app and an Applications shortcut for drag-and-drop installation.
- **Windows:** `app/dist/SubStemStudio/SubStemStudio.exe` — distribute the entire
  `SubStemStudio` folder, including `_internal`, not the executable alone.
- **Linux:** `app/dist/SubStemStudio/SubStemStudio` with its accompanying folder.

This freezes Python, Qt, PyTorch, and application code into a desktop bundle;
recipients do not need a separate Python installation. It is deliberately
directory-based, avoiding unpacking a large PyTorch runtime on every launch.
The config YAMLs are bundled and resolve independently of the launch directory.
Weights are **not** bundled: select external checkpoints or use the existing
Hugging Face cache/download flow. Never ship your HF token or cache credentials.

[PyInstaller is not a cross-compiler](https://pyinstaller.org/en/stable/): build
Windows binaries on Windows and macOS apps on macOS. Build Apple Silicon and
Intel distributions separately using matching Python/PyTorch wheels; availability
depends on the PyTorch release. GPU drivers are not bundled. Build in a clean
environment to avoid accidentally packaging unrelated optional dependencies.

Before release, launch the bundle outside the checkout on a clean machine and
test preview, local checkpoint inference, click-to-play/pause during separation,
cancellation, and WAV export. Verify
every intended device/architecture. These build recipes are not yet validated
as public-release binaries. A local Apple Silicon build was verified on
2026-09-06: ad-hoc signature verification, launch outside the checkout, and
tiny CPU-model inference through the packaged GUI produced four finite stereo
WAVs with matching duration. On 2026-09-07, the rebuilt GUI ran the full Base
FP16 checkpoint on both MPS and CPU in a macOS 26.6.2 arm64 VM without installing
Python/PyTorch in the guest. Each run exported four finite, nonzero stereo WAVs
matching the 2-second, 48 kHz synthetic input. This is an execution smoke test,
not a music-quality or performance qualification. Packaged playback, Intel
macOS, and Windows remain unverified. Public distribution additionally needs platform
signing (and macOS notarization), dependency/license review, and optionally an
installer; the recipe does not bypass OS trust controls or create an installer.
Build output stays under ignored `app/build/` and `app/dist/` directories.

Implementation references: [Qt worker signals](https://doc.qt.io/qtforpython-6/examples/example_widgets_thread_signals.html)
and [pyqtgraph ImageItem](https://pyqtgraph.readthedocs.io/en/pyqtgraph-0.13.6/api_reference/graphicsItems/imageitem.html).
# macOS Core ML backend

The app defaults to Core ML on macOS 15+ and PyTorch elsewhere. Choose
**Backend → Core ML** and **Device → mps** for
GPU-accelerated DSP, or CPU for CPU DSP. Auto selects MPS when available.
DSP defaults to DSP2, including the optimized Metal modulation kernel and
inference filter FFT cache on MPS. The bundled app includes the Metal shader;
no separate compiler installation, model re-export, or new weights are needed.
See [DSP2 validation and benchmarks](../docs/dsp2.md).
MPS DSP requires PyTorch 2.14+; Core ML requires macOS 15+ and blank custom
config/checkpoint fields. Select PyTorch manually for custom models.
Core ML inference runs in a dedicated spawned process so its native prediction
call cannot hold the GUI's Python GIL and starve playback refills. The worker
is closed after completion, cancellation, or failure. Checkpoints, DSP precision,
and playback buffering are unchanged. The Core ML memory indicator reports
combined app-and-worker resident RAM, rather than GUI-only Metal allocations;
system RAM usage covers all processes.
See [Core ML setup, limitations, and export](../docs/coreml.md).
