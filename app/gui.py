from __future__ import annotations

from contextlib import ExitStack
import math
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
from coreml_inference import load_coreml_separator, use_coreml
from app.streaming import separate_audio
from app.player import MIXTURE_KEY, NULL_KEY, StemPlayer
from app.memory import MemoryMonitor
from app.progress import estimate_remaining, format_duration


PLAYBACK_KEYS = (MIXTURE_KEY, "vocals", "bass", "drums", "other", NULL_KEY)


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
        coreml = use_coreml("auto", options["device"], options["config"], options["checkpoint"])
        device = "Core ML" if coreml else resolve_device(options["device"])
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
            model, config = load_coreml_separator(
                local_files_only=options["offline"], download_progress=self.artifact_progress.emit,
            )
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
        self.status.emit(f"Preparing audio · {device} · {precision.upper()} · {rate:,} Hz")
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
        displays = {stem: Spectrogram(audio.shape[-1]) for stem in [*stems, NULL_KEY]}
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
            residual = audio[:, start:end] - sum(chunks.values())
            first, values = displays[NULL_KEY].append(residual)
            if values.shape[1]:
                self.updated.emit(NULL_KEY, first, values)
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

    def __init__(self, name, values, rate, samples, pending=False):
        super().__init__()
        self.setObjectName("card")
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
QPushButton#stemTab:checked { background: #494d64; border-color: #c6a0f6; color: #c6a0f6; }
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
