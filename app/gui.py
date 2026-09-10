from __future__ import annotations

from contextlib import ExitStack
import math
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PySide6.QtCore import QEvent, QRectF, QThread, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGraphicsRectItem, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QToolTip, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

from inference import DEFAULT_CONFIG, load_audio, load_separator, resolve_device, resolve_pretrained_paths
from coreml_inference import use_coreml
from app.coreml_worker import load_coreml_process
from app.streaming import separate_audio
from app.player import MIXTURE_KEY, StemPlayer
from app.memory import MemoryMonitor
from app.progress import estimate_remaining, format_duration


PLAYBACK_KEYS = (MIXTURE_KEY, "vocals", "bass", "drums", "other")
PLAYBACK_COLORS = {
    MIXTURE_KEY: "#c6a0f6",
    "vocals": "#f5bde6",
    "bass": "#8aadf4",
    "drums": "#f5a97f",
    "other": "#a6da95",
    "mute": "#ed8796",
    "invert": "#eed49f",
}


class Spectrogram:
    """Bounded overview with fixed dBFS scale and finalized, non-centered frames."""

    def __init__(self, samples, fft_size=1024, max_columns=1200):
        self.samples = samples
        self.fft_size = fft_size
        self.hop = max(fft_size // 4, math.ceil(samples / max_columns))
        self.starts = np.arange(0, samples, self.hop)
        self.values = np.full((fft_size // 2 + 1, len(self.starts)), -90, dtype=np.float32)
        self.window = np.hanning(fft_size).astype(np.float32)
        self.next_column = 0
        self.pending = None
        self.received = 0

    def append(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or not np.isfinite(audio).all():
            raise ValueError("Expected finite audio [channels, samples]")
        if self.received + audio.shape[-1] > self.samples:
            raise ValueError("Spectrogram received too many samples")
        buffer_start = self.received if self.pending is None else self.pending_start
        buffer = audio if self.pending is None else np.concatenate((self.pending, audio), axis=-1)
        self.received += audio.shape[-1]
        first = self.next_column
        while self.next_column < len(self.starts):
            start = int(self.starts[self.next_column])
            if start + self.fft_size > self.received and self.received < self.samples:
                break
            frame = buffer[:, start - buffer_start:start - buffer_start + self.fft_size]
            frame = np.pad(frame, ((0, 0), (0, self.fft_size - frame.shape[-1])))
            spectrum = np.fft.rfft(frame * self.window, axis=-1)
            magnitude = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=0)) * 2 / self.window.sum()
            self.values[:, self.next_column] = np.clip(20 * np.log10(np.maximum(magnitude, 1e-9)), -90, 0)
            self.next_column += 1
        next_start = int(self.starts[self.next_column]) if self.next_column < len(self.starts) else self.received
        self.pending_start = min(next_start, self.received)
        self.pending = buffer[:, self.pending_start - buffer_start:].copy()
        return first, self.values[:, first:self.next_column].copy()


class InferenceWorker(QThread):
    status = Signal(str)
    prepared = Signal(object, int, int, object)
    updated = Signal(str, int, object)
    audio_ready = Signal(str, int, object)
    progress = Signal(int)
    artifact_progress = Signal(str, object, object)
    timing = Signal(float, object)
    rtf_updated = Signal(float)
    phase_changed = Signal(str)
    precision_changed = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.options = options

    def check_cancel(self):
        if self.isInterruptionRequested():
            raise InterruptedError("Cancelled — no completed export")

    def run(self):
        try:
            self.process()
        except InterruptedError as error:
            self.phase_changed.emit("Cancelled")
            self.status.emit(str(error))
        except Exception as error:
            self.phase_changed.emit("Failed")
            self.failed.emit(f"{type(error).__name__}: {error}")

    def process(self):
        with ExitStack() as resources:
            self._process(resources)

    def _process(self, resources):
        options = self.options
        if options.get("preview"):
            self.status.emit("Reading mixture…")
            audio, rate = sf.read(options["input"], dtype="float32", always_2d=True)
            audio = audio.T
            if audio.shape[-1] == 0:
                raise ValueError("The audio file is empty")
            preview = Spectrogram(audio.shape[-1])
            preview.append(audio)
            self.check_cancel()
            self.prepared.emit(preview.values, rate, audio.shape[-1], [])
            self.audio_ready.emit(MIXTURE_KEY, 0, audio)
            self.status.emit("Ready — choose settings, then separate")
            return
        coreml = use_coreml(options.get("backend", "torch"), options["device"], options["config"], options["checkpoint"])
        device = str(resolve_device(options["device"]))
        if coreml and device not in {"cpu", "mps"}:
            raise ValueError("Core ML DSP requires the cpu or mps device")
        self.status.emit("Resolving model artifacts (missing files may download)…")
        default_config = Path(__file__).resolve().parent.parent / DEFAULT_CONFIG
        if not coreml:
            config_path, checkpoint_path = resolve_pretrained_paths(
                options["config"] or str(default_config), options["checkpoint"] or None,
                local_files_only=options["offline"],
                download_progress=self.artifact_progress.emit,
            )
        self.check_cancel()
        self.phase_changed.emit("Loading model")
        self.status.emit("Loading model…")
        if coreml:
            model, config = load_coreml_process(
                local_files_only=options["offline"], download_progress=self.artifact_progress.emit,
                dsp_device=device,
                model_loading=lambda: self.phase_changed.emit("Loading model"),
            )
            resources.callback(model.close)
        else:
            model, config = load_separator(
                config_path, checkpoint_path, device, local_files_only=True, preserve_dtype=True,
            )
        dtype = next(model.parameters()).dtype
        amp_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else None
        self.check_cancel()
        rate = int(config.audio.sample_rate)
        precision = "fp16" if coreml else {"torch.bfloat16": "bf16", "torch.float16": "fp16"}.get(str(dtype), "fp32")
        self.precision_changed.emit(f"{'Core ML' if coreml else 'PyTorch'} · {precision.upper()}")
        device_label = f"Core ML + {device.upper()} DSP" if coreml else device
        self.status.emit(f"Preparing audio · {device_label} · {precision.upper()} · {rate:,} Hz")
        audio = load_audio(Path(options["input"]), rate)
        target = config.training.get("target_instrument")
        stems = [target] if target else list(config.training.instruments)
        if not stems or len(set(stems)) != len(stems) or any(
            not isinstance(stem, str) or not stem or stem in {".", ".."}
            or any(char in stem for char in '/\\<>:"|?*') for stem in stems
        ):
            raise ValueError("Config must contain unique, safe stem names")
        if audio.shape[-1] == 0:
            raise ValueError("The audio file is empty")
        preview = Spectrogram(audio.shape[-1])
        preview.append(audio)
        displays = {stem: Spectrogram(audio.shape[-1]) for stem in stems}
        self.prepared.emit(preview.values, rate, audio.shape[-1], stems)
        self.audio_ready.emit(MIXTURE_KEY, 0, audio)
        segment_samples = int(config.audio.chunk_size)
        if not 1 <= options["overlap"] <= segment_samples:
            raise ValueError("Overlap ratio must be between 1 and the configured chunk size")
        hop_samples = segment_samples // options["overlap"]
        total_windows = 1 + math.ceil(max(0, audio.shape[-1] - segment_samples) / hop_samples)
        started = time.monotonic()
        self.phase_changed.emit("Separating")
        self.timing.emit(0.0, None)

        def on_chunk(start, chunks):
            self.check_cancel()
            for stem, chunk in chunks.items():
                writers[stem].write(chunk.T)
                self.audio_ready.emit(stem, start, chunk)
                first, values = displays[stem].append(chunk)
                if values.shape[1]:
                    self.updated.emit(stem, first, values)
            end = start + next(iter(chunks.values())).shape[-1]
            self.progress.emit(int(100 * end / audio.shape[-1]))
            elapsed = time.monotonic() - started
            completed_windows = total_windows if end == audio.shape[-1] else end // hop_samples
            self.timing.emit(elapsed, estimate_remaining(elapsed, completed_windows, total_windows))
            self.rtf_updated.emit(elapsed / (end / rate))
            self.status.emit(f"Separating · {end / rate:.1f} / {audio.shape[-1] / rate:.1f} s finalized")

        destination = Path(options["output"]).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="subroformer-", dir=destination))
        try:
            with ExitStack() as stack:
                writers = {
                    stem: stack.enter_context(sf.SoundFile(
                        folder / f"{index:02d}-{stem}.wav", mode="x",
                        samplerate=rate, channels=int(config.model.get("audio_channels", config.audio.num_channels)),
                        subtype="FLOAT",
                    ))
                    for index, stem in enumerate(stems, 1)
                }
                separate_audio(
                    model, audio, config, options["overlap"], options["batch"], amp_dtype,
                    on_chunk=on_chunk, should_cancel=self.isInterruptionRequested,
                    collect_output=False,
                )
                self.check_cancel()
                self.phase_changed.emit("Saving stems")
                self.status.emit("Finalizing float32 WAV stems…")
        except Exception:
            self.status.emit(f"Export incomplete: {folder}")
            raise
        self.completed.emit(str(folder))


class SpectrogramCard(QFrame):
    seek_requested = Signal(str, float)

    def __init__(self, name, values, rate, samples, pending=False, color=PLAYBACK_COLORS[MIXTURE_KEY]):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet(
            f'QFrame#card[selected="true"] {{ border-color: {color}; }}'
            f'QLabel#cardTitle {{ color: {color}; }}'
        )
        self.values = values.copy()
        self.name = name
        self.duration = samples / rate
        self.ready_columns = 0 if pending else values.shape[1]
        self.frequency_max = rate / 2
        layout = QVBoxLayout(self)
        title = QLabel(name.upper())
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        self.plot = pg.PlotWidget(background="#1e2030")
        self.plot.setMinimumHeight(160)
        self.plot.setLabel("bottom", "Time", units="s", color="#b8c0e0")
        self.plot.setLabel("left", "Frequency", units="Hz", color="#b8c0e0")
        for name in ("bottom", "left"):
            axis = self.plot.getAxis(name)
            axis.setPen("#494d64")
            axis.setTextPen("#b8c0e0")
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setMenuEnabled(False)
        self.image = pg.ImageItem(axisOrder="row-major")
        colors = pg.colormap.get("magma")
        self.image.setLookupTable(colors.getLookupTable(nPts=256))
        self.plot.addItem(self.image)
        self.image.setImage(self.values, levels=(-90, 0), autoLevels=False)
        self.image.setRect(QRectF(0, 0, samples / rate, rate / 2))
        self.plot.setLimits(xMin=0, xMax=samples / rate, yMin=0, yMax=rate / 2)
        self.plot.setRange(xRange=(0, samples / rate), yRange=(0, rate / 2), padding=0)
        self.pending_background = QGraphicsRectItem()
        self.pending_background.setBrush(QBrush(QColor("#363a4f")))
        self.pending_hatching = QGraphicsRectItem()
        self.pending_hatching.setBrush(QBrush(QColor("#494d64"), Qt.BrushStyle.BDiagPattern))
        for item in (self.pending_background, self.pending_hatching):
            item.setPen(pg.mkPen(None))
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setZValue(10)
            self.plot.addItem(item, ignoreBounds=True)
        self.ready_boundary = pg.InfiniteLine(angle=90, pen=pg.mkPen("#eed49f", width=1))
        self.ready_boundary.setZValue(11)
        self.plot.addItem(self.ready_boundary, ignoreBounds=True)
        self.update_pending()
        self.playhead = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#8bd5ca", width=2))
        self.playhead.setZValue(20)
        self.playhead.hide()
        self.plot.addItem(self.playhead)
        self.plot.scene().sigMouseClicked.connect(self.clicked)
        layout.addWidget(self.plot)

    def update_pending(self):
        boundary = self.duration * self.ready_columns / self.values.shape[1]
        pending = self.ready_columns < self.values.shape[1]
        for item in (self.pending_background, self.pending_hatching):
            item.setRect(QRectF(boundary, 0, self.duration - boundary, self.frequency_max))
            item.setVisible(pending)
        self.ready_boundary.setValue(boundary)
        self.ready_boundary.setVisible(pending and self.ready_columns > 0)

    def clicked(self, event):
        view = self.plot.getViewBox()
        if event.button() == Qt.MouseButton.LeftButton and view.sceneBoundingRect().contains(event.scenePos()):
            seconds = view.mapSceneToView(event.scenePos()).x()
            self.seek_requested.emit(self.name, min(self.duration, max(0, seconds)))
            if self.ready_columns < self.values.shape[1] and seconds >= self.duration * self.ready_columns / self.values.shape[1]:
                QToolTip.showText(self.plot.mapToGlobal(self.plot.mapFromScene(event.scenePos())), "Audio at this position isn’t ready yet.", self.plot)

    def replace(self, first, values):
        self.values[:, first:first + values.shape[1]] = values
        self.image.setImage(self.values, levels=(-90, 0), autoLevels=False)
        self.ready_columns = max(self.ready_columns, first + values.shape[1])
        self.update_pending()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.close_pending = False
        self.cards = {}
        self.stem_buttons = {}
        self.export_path = None
        self.player = StemPlayer(self)
        self.setWindowTitle("SubRoformer · Stem Studio")
        self.setAcceptDrops(True)
        self.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "subroformer-icon.png")))
        self.playback_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.playback_shortcut.setAutoRepeat(False)
        self.playback_shortcut.activated.connect(self.toggle_playback)
        self.stem_shortcuts = []
        for number, stem in enumerate(PLAYBACK_KEYS, 1):
            shortcut = QShortcut(QKeySequence(str(number)), self)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(lambda stem=stem: self.switch_playback(stem))
            self.stem_shortcuts.append(shortcut)
        self.resize(1180, 860)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        self.pane_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.pane_splitter.setChildrenCollapsible(False)
        self.pane_splitter.setHandleWidth(10)
        layout.addWidget(self.pane_splitter)
        config_pane = QWidget()
        config_pane.setMinimumWidth(240)
        sidebar = QVBoxLayout(config_pane)
        sidebar.setContentsMargins(0, 0, 8, 0)
        self.pane_splitter.addWidget(config_pane)
        heading = QLabel("SubRoformer")
        heading.setObjectName("heading")
        sidebar.addWidget(heading)
        sidebar.addWidget(QLabel("STEM STUDIO  /  LOCAL INFERENCE"))
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        form.setContentsMargins(0, 24, 0, 12)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.input = self.path_field(form, "01  Mixture", "Choose or drop an audio file…", "audio")
        self.output = self.path_field(form, "02  Export folder", "Choose folder…", "directory")
        self.output.setText(str(Path.home() / "Music" / "SubRoformer"))
        self.backend = QComboBox()
        self.backend.addItem("PyTorch", "torch")
        self.backend.addItem("Core ML", "coreml")
        if sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0]) >= 15:
            self.backend.setCurrentIndex(self.backend.findData("coreml"))
        self.backend.setToolTip("Core ML requires macOS 15+ and default Base model paths. Select MPS for GPU DSP (PyTorch 2.14+).")
        form.addRow("Backend", self.backend)
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(["auto", "cuda", "mps", "cpu"])
        self.device.setCurrentText(str(resolve_device("auto")))
        self.device.setToolTip("Auto prefers CUDA, then MPS, then CPU. For Core ML, this selects the DSP device; MPS accelerates DSP on the GPU.")
        form.addRow("Device", self.device)
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setToolTip("Windows per inference batch. Higher values require more memory.")
        form.addRow("Batch size", self.batch)
        self.overlap = QSpinBox()
        self.overlap.setRange(1, 32)
        self.overlap.setValue(4)
        self.overlap.setToolTip("Hop = configured chunk length / overlap. Higher is slower.")
        form.addRow("Overlap ratio", self.overlap)
        self.config = self.path_field(form, "Config (optional)", "Default Base config", "config")
        self.checkpoint = self.path_field(form, "Checkpoint (optional)", "Default Base weights", "checkpoint")
        self.offline = QCheckBox("Offline / cached files only")
        form.addRow(self.offline)
        note = QLabel("Leave model paths blank to use Base defaults. Missing defaults may download from Hugging Face; private weights require authentication.")
        note.setWordWrap(True)
        form.addRow(note)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setWidget(self.controls)
        sidebar.addWidget(self.settings_scroll, 1)
        self.start = QPushButton("Separate stems")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self.separate)
        sidebar.addWidget(self.start)
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.stop)
        sidebar.addWidget(self.cancel)
        self.open_output = QPushButton("Open exported stems ↗")
        self.open_output.setEnabled(False)
        self.open_output.clicked.connect(self.open_export)
        sidebar.addWidget(self.open_output)
        main_pane = QWidget()
        main_pane.setMinimumWidth(400)
        content = QVBoxLayout(main_pane)
        content.setContentsMargins(0, 0, 0, 0)
        self.pane_splitter.addWidget(main_pane)
        self.pane_splitter.setStretchFactor(0, 0)
        self.pane_splitter.setStretchFactor(1, 1)
        self.pane_splitter.setSizes([328, 790])
        self.pane_splitter.handle(1).setToolTip("Drag to resize the configuration pane")
        self.settings_wheel_targets = (
            config_pane, self.controls, self.backend, self.device, self.batch, self.overlap,
            *self.controls.findChildren(QLineEdit), self.pane_splitter.handle(1),
            self.settings_scroll.viewport(), self.settings_scroll.verticalScrollBar(),
        )
        for widget in self.settings_wheel_targets:
            widget.installEventFilter(self)
        title = QLabel("Hear the parts. See the detail.")
        title.setObjectName("heading")
        content.addWidget(title)
        subtitle = QLabel("SPECTRAL OVERVIEW    ·    −90 → 0 dBFS    ·    channel RMS magnitude")
        content.addWidget(subtitle)
        display_options = QHBoxLayout()
        display_options.addWidget(QLabel("Spectrogram layout"))
        self.spectrogram_layout = QComboBox()
        self.spectrogram_layout.addItems(["Stacked", "Grid", "Fit all"])
        display_options.addWidget(self.spectrogram_layout)
        display_options.addStretch()
        content.addLayout(display_options)
        self.stem_tabs = QHBoxLayout()
        for number, (key, label) in enumerate(zip(PLAYBACK_KEYS, ("Mixture", "Vocals", "Bass", "Drums", "Others")), 1):
            button = QPushButton(f"{label} ({number})")
            button.setObjectName("stemTab")
            button.setProperty("accent", PLAYBACK_COLORS[key])
            button.setCheckable(True)
            button.setEnabled(False)
            button.clicked.connect(lambda checked, stem=key: self.switch_playback(stem))
            self.stem_buttons[key] = button
            self.stem_tabs.addWidget(button)
        self.mute_all = QPushButton("Mute all (`)")
        self.mute_all.setObjectName("stemTab")
        self.mute_all.setProperty("accent", PLAYBACK_COLORS["mute"])
        self.mute_all.setEnabled(False)
        self.mute_all.setToolTip("Click or press ` to mute all without pausing playback.")
        self.mute_all.clicked.connect(lambda: self.player.select_stems([]))
        self.mute_shortcut = QShortcut(QKeySequence(Qt.Key.Key_QuoteLeft), self)
        self.mute_shortcut.setAutoRepeat(False)
        self.mute_shortcut.activated.connect(self.mute_all.click)
        self.invert_shortcut = QShortcut(self)
        self.invert_shortcut.setKeys([QKeySequence("~"), QKeySequence("Shift+~")])
        self.invert_shortcut.setAutoRepeat(False)
        self.invert_shortcut.activated.connect(self.invert_playback)
        self.stem_tabs.addWidget(self.mute_all)
        self.invert_button = QPushButton("Invert (~)")
        self.invert_button.setObjectName("stemTab")
        self.invert_button.setProperty("accent", PLAYBACK_COLORS["invert"])
        self.invert_button.setEnabled(False)
        self.invert_button.setToolTip("Click or press ~ to invert playing stems: enabled stems turn off and muted stems turn on.")
        self.invert_button.clicked.connect(self.invert_playback)
        self.stem_tabs.addWidget(self.invert_button)
        content.addLayout(self.stem_tabs)
        self.spectrogram_scroll = QScrollArea()
        self.spectrogram_scroll.setWidgetResizable(True)
        self.spectrogram_scroll.viewport().installEventFilter(self)
        container = QWidget()
        self.plots = QGridLayout(container)
        self.placeholder = QLabel("Choose a mixture to preview its spectrogram.\n\nHatched regions are not processed yet and fill left to right\nas inference finalizes each section.")
        self.plots.addWidget(self.placeholder, 0, 0)
        self.spectrogram_scroll.setWidget(container)
        self.status_splitter = QSplitter(Qt.Orientation.Vertical)
        self.status_splitter.setChildrenCollapsible(False)
        self.status_splitter.setHandleWidth(10)
        self.spectrogram_scroll.setMinimumHeight(180)
        self.status_splitter.addWidget(self.spectrogram_scroll)
        status_pane = QFrame()
        status_pane.setObjectName("infoPane")
        status_layout = QVBoxLayout(status_pane)
        status_layout.setContentsMargins(16, 14, 16, 14)
        status_layout.setSpacing(10)
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setMinimumHeight(180)
        info_scroll.setWidget(status_pane)
        self.status_splitter.addWidget(info_scroll)
        self.status_splitter.setStretchFactor(0, 1)
        self.status_splitter.setStretchFactor(1, 0)
        self.status_splitter.setSizes([340, 330])
        self.status_splitter.handle(1).setToolTip("Drag to resize the playback and status pane")
        content.addWidget(self.status_splitter, 1)
        self.spectrogram_layout.currentTextChanged.connect(self.arrange_spectrograms)
        playback = QHBoxLayout()
        self.pause_playback = QPushButton("Play")
        self.pause_playback.setObjectName("playback")
        self.pause_playback.setProperty("playing", False)
        self.pause_playback.setEnabled(False)
        self.pause_playback.clicked.connect(self.toggle_playback)
        self.pause_playback.setToolTip("Play / pause the selected audio (Space)")
        playback.addWidget(self.pause_playback)
        self.pause_playback.setMinimumWidth(80)
        self.playback_status = QLabel("Click any spectrogram to listen from that point")
        self.playback_status.setWordWrap(True)
        self.playback_status.setObjectName("playbackSummary")
        playback.addWidget(self.playback_status, 1)
        status_layout.addLayout(playback)
        shortcut_hint = QLabel("Space  Play / pause    ·    Click a spectrogram to seek")
        shortcut_hint.setObjectName("infoHint")
        shortcut_hint.setWordWrap(True)
        status_layout.addWidget(shortcut_hint)
        self.player.changed.connect(self.update_playback)
        self.player.failed.connect(self.playback_error)
        self.progress = QProgressBar()
        self.progress.setObjectName("infoProgress")
        self.progress.setValue(0)
        self.progress.setFormat("Separation · %p%")
        status_layout.addWidget(self.progress)
        self.separation_phase = "Ready"
        self.separation_elapsed = 0.0
        self.separation_eta = None
        self.separation_rtf = None
        self.timing_status = QLabel("Ready · ETA —")
        self.timing_status.setWordWrap(True)
        self.rtf_status = QLabel("RTF —")
        self.rtf_status.setWordWrap(True)
        self.rtf_status.setToolTip("Real-time factor = elapsed separation time / finalized audio duration. ≤ 1 meets real-time speed. Excludes loading and export; not live audio latency.")
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.memory_status = QLabel("Memory · reading device…")
        self.memory_status.setWordWrap(True)
        self.precision = QLabel("Not loaded")
        self.precision.setWordWrap(True)
        self.precision.setToolTip("Loaded model precision. Core ML uses FP16 with FP32 normalization and audio/DSP. PyTorch follows checkpoint precision.")
        metrics = QGridLayout()
        metrics.setSpacing(8)
        for index, (label, value) in enumerate((
            ("SESSION", self.status), ("TIMING", self.timing_status),
            ("MEMORY", self.memory_status), ("THROUGHPUT", self.rtf_status),
            ("PRECISION", self.precision),
        )):
            card = QFrame()
            card.setObjectName("infoMetric")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 9, 12, 9)
            card_layout.setSpacing(5)
            caption = QLabel(label)
            caption.setObjectName("infoCaption")
            card_layout.addWidget(caption)
            value.setObjectName("infoValue")
            value.setTextFormat(Qt.TextFormat.PlainText)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            card_layout.addWidget(value)
            metrics.addWidget(card, index // 2, index % 2)
        metrics.setColumnStretch(0, 1)
        metrics.setColumnStretch(1, 1)
        status_layout.addLayout(metrics)
        self.memory_monitor = MemoryMonitor(self)
        self.memory_monitor.requested = self.device.currentText()
        self.memory_monitor.backend = self.backend.currentData()
        self.memory_monitor.updated.connect(self.update_memory)
        self.device.currentTextChanged.connect(self.select_memory_device)
        self.backend.currentIndexChanged.connect(self.select_memory_backend)
        self.memory_monitor.start()

    @Slot(str)
    def select_memory_device(self, device):
        self.memory_monitor.requested = device
        self.memory_status.setText(f"{device.upper()} · memory: reading…")

    @Slot(object)
    def update_memory(self, usage):
        if usage.requested == self.device.currentText() and usage.backend == self.backend.currentData():
            self.memory_status.setText(usage.text())
            self.memory_status.setToolTip(usage.detail)

    @Slot(int)
    def select_memory_backend(self, index):
        self.memory_monitor.backend = self.backend.currentData()
        self.select_memory_device(self.device.currentText())

    def path_field(self, form, label, placeholder, kind):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit()
        field.setAcceptDrops(False)
        field.setPlaceholderText(placeholder)
        button = QPushButton("…")
        button.setFixedWidth(34)
        button.clicked.connect(lambda: self.browse(field, kind))
        layout.addWidget(field)
        layout.addWidget(button)
        form.addRow(label, row)
        return field

    def dropped_audio_path(self, event):
        if not self.controls.isEnabled():
            return None
        urls = event.mimeData().urls()
        if len(urls) != 1 or not urls[0].isLocalFile():
            return None
        path = Path(urls[0].toLocalFile())
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}:
            return str(path)
        return None

    def dragEnterEvent(self, event):
        if self.dropped_audio_path(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self.dropped_audio_path(event)
        if path:
            self.input.setText(path)
            event.acceptProposedAction()
            self.launch({"input": path, "preview": True})
        else:
            event.ignore()

    def browse(self, field, kind):
        if kind == "directory":
            path = QFileDialog.getExistingDirectory(self, "Export folder")
        else:
            filters = {"audio": "Audio (*.wav *.flac *.aiff *.aif *.ogg *.mp3);;All files (*)",
                       "config": "Config (*.yaml *.yml)",
                       "checkpoint": "Weights (*.safetensors *.pt *.pth)"}
            path, _ = QFileDialog.getOpenFileName(self, "Choose " + kind, "", filters[kind])
        if path:
            field.setText(path)
            if kind == "audio":
                self.launch({"input": path, "preview": True})

    @Slot()
    def separate(self):
        if not Path(self.input.text()).is_file() or not self.output.text().strip():
            QMessageBox.warning(self, "Choose files", "Select an input audio file and export folder first.")
            return
        self.export_path = None
        self.open_output.setEnabled(False)
        self.launch({
            "input": self.input.text(), "output": self.output.text(),
            "config": self.config.text().strip(), "checkpoint": self.checkpoint.text().strip(),
            "device": self.device.currentText(),
            "backend": self.backend.currentData(),
            "batch": self.batch.value(), "overlap": self.overlap.value(),
            "offline": self.offline.isChecked(),
        })

    def launch(self, options):
        self.player.reset()
        self.controls.setEnabled(False)
        self.start.setEnabled(False)
        self.cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Preparing…")
        self.separation_elapsed = 0.0
        self.separation_eta = None
        self.separation_rtf = None
        self.precision.setText("Not loaded" if options.get("preview") else "Loading…")
        self.update_phase("Reading preview" if options.get("preview") else "Loading / preparing")
        self.worker = InferenceWorker(options, self)
        self.worker.status.connect(self.status.setText)
        self.worker.precision_changed.connect(self.precision.setText)
        self.worker.prepared.connect(self.prepare)
        self.worker.updated.connect(self.update_stem)
        self.worker.audio_ready.connect(self.receive_audio, Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.artifact_progress.connect(self.update_artifact_progress)
        self.worker.timing.connect(self.update_timing)
        self.worker.rtf_updated.connect(self.update_rtf)
        self.worker.phase_changed.connect(self.update_phase)
        self.worker.failed.connect(self.error)
        self.worker.completed.connect(self.complete)
        self.worker.finished.connect(self.idle)
        self.worker.start()

    @Slot(object, int, int, object)
    def prepare(self, mixture, rate, samples, stems):
        self.player.reset(rate, [MIXTURE_KEY, *stems])
        for key, button in self.stem_buttons.items():
            button.setEnabled(key in self.player.buffers)
        while self.plots.count():
            item = self.plots.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self.cards = {}
        mixture_card = SpectrogramCard("Input mixture", mixture, rate, samples)
        mixture_card.seek_requested.connect(lambda name, seconds: self.player.seek_time(seconds))
        mixture_card.plot.setToolTip("Click to seek the enabled stems · drag to pan · scroll to zoom")
        self.cards[MIXTURE_KEY] = mixture_card
        for stem in stems:
            card = SpectrogramCard(stem, np.full_like(mixture, -90), rate, samples, pending=True, color=PLAYBACK_COLORS.get(stem, PLAYBACK_COLORS[MIXTURE_KEY]))
            card.plot.setXLink(mixture_card.plot)
            card.seek_requested.connect(lambda name, seconds: self.player.seek_time(seconds))
            card.plot.setToolTip("Hatched region: audio is not processed yet · click to seek the enabled stems · drag to pan · scroll to zoom")
            self.cards[stem] = card
        self.switch_playback(MIXTURE_KEY)
        self.arrange_spectrograms()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Separation · %p%")
        if not stems:
            self.update_phase("Preview ready")

    @Slot()
    def arrange_spectrograms(self):
        if not self.cards:
            return
        while self.plots.count():
            self.plots.takeAt(0)
        mode = self.spectrogram_layout.currentText()
        columns = 1 if mode == "Stacked" else (2 if mode == "Grid" else math.ceil(math.sqrt(len(self.cards))))
        for index, card in enumerate(self.cards.values()):
            card.plot.setMinimumHeight(0 if mode == "Fit all" else 160)
            self.plots.addWidget(card, index // columns, index % columns)
        policy = Qt.ScrollBarPolicy.ScrollBarAlwaysOff if mode == "Fit all" else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        self.spectrogram_scroll.setVerticalScrollBarPolicy(policy)
        self.resize_spectrograms()

    def resize_spectrograms(self):
        if not self.cards:
            return
        if self.spectrogram_layout.currentText() == "Fit all":
            columns = math.ceil(math.sqrt(len(self.cards)))
            rows = math.ceil(len(self.cards) / columns)
            margins = self.plots.contentsMargins()
            available = self.spectrogram_scroll.viewport().height() - margins.top() - margins.bottom()
            height = max(1, (available - self.plots.verticalSpacing() * (rows - 1)) // rows)
        else:
            height = 210
        for card in self.cards.values():
            card.setFixedHeight(height)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and watched in getattr(self, "settings_wheel_targets", ()):
            scrollbar = self.settings_scroll.verticalScrollBar()
            if event.pixelDelta().y():
                scrollbar.setValue(scrollbar.value() - event.pixelDelta().y())
                event.accept()
                return True
            if watched is not scrollbar:
                QApplication.sendEvent(scrollbar, event)
                return True
        if hasattr(self, "spectrogram_scroll") and watched is self.spectrogram_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self.resize_spectrograms()
        return super().eventFilter(watched, event)

    @Slot(str)
    def update_phase(self, phase):
        self.separation_phase = phase
        if phase == "Loading model":
            self.progress.setRange(0, 0)
            self.progress.setFormat("Loading model…")
        self.render_timing()

    @Slot(str, object, object)
    def update_artifact_progress(self, filename, completed, total):
        now = time.monotonic()
        if (self.separation_phase != "Downloading artifacts"
                or getattr(self, "download_filename", None) != filename
                or completed < (getattr(self, "download_initial_bytes", None) or 0)
                or total is None):
            self.download_filename = filename
            self.download_started = now
            self.download_initial_bytes = None
        if self.download_initial_bytes is None and completed > 0:
            self.download_initial_bytes = completed
            self.download_rate_started = now
        self.update_phase("Downloading artifacts")
        name = Path(filename).name
        if total is None or total <= 0:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Checking artifact…")
            self.status.setText(f"Checking cache / connecting · {name}")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(100, int(100 * completed / total)))
            self.progress.setFormat("Artifact download · %p%")
            self.status.setText(f"{name} · {completed / 1_000_000:.2f} / {total / 1_000_000:.2f} MB")
        elapsed = now - self.download_started
        received = completed - (self.download_initial_bytes or 0)
        eta = None
        if total and completed >= total:
            eta = 0
        elif total and received > 0 and elapsed > 0:
            eta = (now - self.download_rate_started) * (total - completed) / received
        self.timing_status.setText(f"Download {format_duration(elapsed)} · ETA {format_duration(eta)}")

    @Slot(float, object)
    def update_timing(self, elapsed, eta):
        self.separation_elapsed = elapsed
        self.separation_eta = eta
        self.render_timing()

    def render_timing(self):
        elapsed = format_duration(self.separation_elapsed)
        if self.separation_phase == "Separating":
            self.timing_status.setText(f"Elapsed {elapsed} · ETA {format_duration(self.separation_eta)}")
        else:
            self.timing_status.setText(f"{self.separation_phase} · Separation time {elapsed} · ETA —")
        if self.separation_rtf is None:
            self.rtf_status.setText("RTF —")
        else:
            scope = "" if self.separation_phase in {"Saving stems", "Complete"} else " (so far)"
            speed = "real-time or faster" if self.separation_rtf <= 1 else "slower than real-time"
            self.rtf_status.setText(f"RTF{scope} {self.separation_rtf:.3f} · {speed}")

    @Slot(float)
    def update_rtf(self, rtf):
        self.separation_rtf = rtf
        self.render_timing()

    @Slot(str, int, object)
    def update_stem(self, stem, first, values):
        self.cards[stem].replace(first, values)

    @Slot(str, int, object)
    def receive_audio(self, stem, start, audio):
        try:
            self.player.append(stem, start, audio)
        except OSError as error:
            self.stop()
            self.playback_error(f"Unable to store playback audio: {error}")

    @Slot()
    def toggle_playback(self):
        if self.player.playing:
            self.player.pause()
        else:
            self.player.resume()

    @Slot()
    def invert_playback(self):
        components = set(self.player.buffers) - {MIXTURE_KEY} or set(self.player.buffers)
        selected = components if MIXTURE_KEY in self.player.selected else set(self.player.selected)
        self.player.select_stems(components - selected)

    def switch_playback(self, stem):
        if stem not in self.player.buffers:
            return
        selected = set(self.player.selected)
        if stem == MIXTURE_KEY:
            selected = set(self.player.buffers) - {MIXTURE_KEY} or {MIXTURE_KEY}
        elif MIXTURE_KEY in selected:
            selected = set(self.player.buffers) - {MIXTURE_KEY, stem}
        elif stem in selected:
            selected.remove(stem)
        else:
            selected.add(stem)
        self.player.select_stems(selected)
        self.update_playback()

    @Slot()
    def update_playback(self):
        player = self.player
        self.mute_all.setEnabled(bool(player.selected))
        self.invert_button.setEnabled(bool(player.buffers))
        for stem, button in self.stem_buttons.items():
            button.setChecked(player.stem == MIXTURE_KEY if stem == MIXTURE_KEY else stem in player.selected)
        for stem, card in self.cards.items():
            selected = player.stem == MIXTURE_KEY if stem == MIXTURE_KEY else stem in player.selected
            if card.property("selected") != selected:
                card.setProperty("selected", selected)
                card.style().unpolish(card)
                card.style().polish(card)
            card.playhead.setVisible(player.available > 0)
            card.playhead.setValue(player.position / player.rate)
        available = player.available
        self.pause_playback.setEnabled(player.playing or player.position < available)
        self.pause_playback.setText("Pause" if player.playing else "Play")
        if self.pause_playback.property("playing") != player.playing:
            self.pause_playback.setProperty("playing", player.playing)
            self.pause_playback.style().unpolish(self.pause_playback)
            self.pause_playback.style().polish(self.pause_playback)
        if not player.selected:
            state = "Playing silently" if player.playing else ("At available edge" if player.position >= available else "Paused")
            self.playback_status.setText(f"Muted · {state} · {player.position / player.rate:.1f} / {available / player.rate:.1f} s available")
        else:
            state = "Playing" if player.playing else ("At available edge" if player.position >= available else "Paused")
            names = "Mixture (all on)" if player.stem == MIXTURE_KEY else " + ".join(
                "Others" if key == "other" else key.title()
                for key in player.selected
            )
            self.playback_status.setText(
                f"{names} · {state} · {player.position / player.rate:.1f} / {available / player.rate:.1f} s available"
            )

    @Slot(str)
    def playback_error(self, message):
        QMessageBox.warning(self, "Audio playback", message)

    @Slot(str)
    def error(self, message):
        self.status.setText("Failed — " + message)
        QMessageBox.critical(self, "Unable to separate", message + "\n\nFor memory errors, lower batch size or use CPU. Verify model paths, device, and precision.")

    @Slot(str)
    def complete(self, folder):
        self.export_path = folder
        self.open_output.setEnabled(True)
        self.status.setText("Complete — stems saved to " + folder)
        self.progress.setValue(100)
        self.update_phase("Complete")

    @Slot()
    def idle(self):
        self.controls.setEnabled(True)
        self.start.setEnabled(True)
        self.cancel.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("Separation · %p%")
        self.worker.deleteLater()
        self.worker = None
        if self.close_pending:
            self.close()

    @Slot()
    def stop(self):
        if self.worker:
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.update_phase("Cancelling")
            self.status.setText("Cancelling after the current operation (downloads / model calls must finish)…")

    @Slot()
    def open_export(self):
        if self.export_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.export_path))

    def closeEvent(self, event):
        self.player.release()
        if self.worker is not None:
            self.close_pending = True
            self.stop()
            event.ignore()
        else:
            self.player.reset()
            self.memory_monitor.stop()
            self.memory_monitor.wait()
            event.accept()


STYLE = """
QWidget { background: #24273a; color: #cad3f5; font-size: 13px; selection-background-color: #c6a0f6; selection-color: #24273a; }
QLabel#heading { font-size: 25px; font-weight: 600; }
QLabel#cardTitle { background: transparent; color: #c6a0f6; font-size: 12px; font-weight: 600; }
QFrame#card { background: #1e2030; border: 1px solid #494d64; border-radius: 10px; }
QFrame#card[selected="true"] { border-color: #c6a0f6; }
QFrame#infoPane { background: #1e2030; border: 1px solid #363a4f; border-radius: 12px; }
QFrame#infoPane QLabel { background: transparent; }
QFrame#infoMetric { background: #24273a; border: 1px solid #363a4f; border-radius: 8px; }
QLabel#infoCaption { color: #b7bdf8; font-size: 10px; font-weight: 600; }
QLabel#infoValue, QLabel#playbackSummary { color: #cad3f5; font-size: 12px; }
QLabel#infoHint { color: #a5adcb; font-size: 11px; }
QLineEdit, QComboBox, QSpinBox { background: #363a4f; border: 1px solid #494d64; border-radius: 6px; padding: 7px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #c6a0f6; }
QPushButton { background: #363a4f; border: 1px solid #494d64; border-radius: 7px; padding: 10px; }
QPushButton:hover { background: #494d64; }
QPushButton:focus { border-color: #c6a0f6; }
QPushButton:pressed { background: #5b6078; }
QPushButton#stemTab { padding: 6px; }
QPushButton#primary { background: #c6a0f6; color: #24273a; font-weight: 600; }
QPushButton#primary:hover { background: #b7bdf8; }
QPushButton#primary:pressed { background: #8aadf4; }
QPushButton#playback:enabled { background: #a6da95; color: #24273a; font-weight: 600; }
QPushButton#playback[playing="true"]:enabled { background: #ed8796; }
QPushButton#playback:enabled:hover { border-color: #cad3f5; }
QPushButton#playback:enabled:pressed { border-color: #24273a; }
QPushButton:disabled, QPushButton#primary:disabled, QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled { background: #1e2030; color: #8087a2; border-color: #363a4f; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #8087a2; border-radius: 3px; background: #363a4f; }
QCheckBox::indicator:checked { background: #c6a0f6; border-color: #c6a0f6; }
QCheckBox::indicator:hover { border-color: #c6a0f6; }
QCheckBox:disabled { color: #8087a2; }
QToolTip { background: #363a4f; color: #cad3f5; border: 1px solid #5b6078; }
QComboBox QAbstractItemView { background: #363a4f; color: #cad3f5; selection-background-color: #494d64; selection-color: #cad3f5; }
QScrollArea { border: none; }
QSplitter::handle { border: none; margin: 0; padding: 0; }
QSplitter::handle:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 transparent, stop:0.4 transparent, stop:0.45 rgba(202, 211, 245, 40), stop:0.55 rgba(202, 211, 245, 40), stop:0.6 transparent, stop:1 transparent); }
QSplitter::handle:vertical { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 transparent, stop:0.4 transparent, stop:0.45 rgba(202, 211, 245, 40), stop:0.55 rgba(202, 211, 245, 40), stop:0.6 transparent, stop:1 transparent); }
QSplitter::handle:horizontal:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 transparent, stop:0.35 transparent, stop:0.45 rgba(198, 160, 246, 130), stop:0.55 rgba(198, 160, 246, 130), stop:0.65 transparent, stop:1 transparent); }
QSplitter::handle:vertical:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 transparent, stop:0.35 transparent, stop:0.45 rgba(198, 160, 246, 130), stop:0.55 rgba(198, 160, 246, 130), stop:0.65 transparent, stop:1 transparent); }
QScrollBar:vertical { background: #1e2030; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: #5b6078; min-height: 24px; border-radius: 6px; }
QScrollBar::handle:vertical:hover { background: #8087a2; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QProgressBar { border: none; background: #363a4f; color: #cad3f5; border-radius: 5px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #494d64; border-radius: 5px; }
QProgressBar#infoProgress { background: #363a4f; height: 18px; font-size: 11px; }
QProgressBar#infoProgress::chunk { background: #5b5680; border-radius: 5px; }
"""


STYLE += "\n".join(
    f'QPushButton#stemTab[accent="{color}"]:enabled {{ color: {color}; border-color: {color}; }}'
    f'QPushButton#stemTab[accent="{color}"]:enabled:checked {{ background: {color}; color: #24273a; }}'
    f'QPushButton#stemTab[accent="{color}"]:enabled:hover {{ background: #494d64; color: {color}; }}'
    for color in PLAYBACK_COLORS.values()
)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    app.setWindowIcon(window.windowIcon())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
