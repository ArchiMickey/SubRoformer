import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QEventLoop, QMimeData, QPoint, QPointF, QTimer, Qt, QUrl
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit
from PySide6.QtTest import QTest
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPalette, QWheelEvent
import pyqtgraph as pg

from app.gui import InferenceWorker, MainWindow, Spectrogram, STYLE, MIXTURE_KEY, NULL_KEY
from app.memory import MemoryUsage
from model_utils import build_model
from tests.helpers import tiny_config


@pytest.mark.parametrize("samples", [1, 512, 1024, 3333, 10000])
def test_incremental_spectrogram(samples):
    audio = np.random.default_rng(7).normal(0, .1, (2, samples)).astype(np.float32)
    full = Spectrogram(samples)
    full.append(audio)
    streamed = Spectrogram(samples)
    assert np.all(streamed.values == -90)
    for start in range(0, samples, 137):
        streamed.append(audio[:, start:start + 137])
        assert np.all(streamed.values[:, streamed.next_column:] == -90)
    np.testing.assert_array_equal(streamed.values, full.values)


def test_silent_and_antiphase_audio():
    silent = Spectrogram(2048)
    silent.append(np.zeros((2, 2048), dtype=np.float32))
    assert np.all(silent.values == -90)
    tone = np.sin(2 * np.pi * np.arange(2048) / 32).astype(np.float32)
    display = Spectrogram(2048)
    display.append(np.stack([tone, -tone]))
    assert display.values.max() > -1


@pytest.fixture(scope="module")
def application():
    application = QApplication.instance() or QApplication([])
    application.setStyleSheet(STYLE)
    yield application


def test_cards_start_blank_and_replace_only_received_columns(application):
    window = MainWindow()
    mixture = np.full((513, 30), -20, dtype=np.float32)
    window.prepare(mixture, 1600, 8000, ["vocals", "drums"])
    assert np.all(window.cards["vocals"].values == -90)
    window.update_stem("vocals", 0, mixture[:, :3])
    assert np.all(window.cards["vocals"].values[:, :3] == -20)
    assert np.all(window.cards["vocals"].values[:, 3:] == -90)
    assert np.all(window.cards["drums"].values == -90)
    assert np.all(window.cards[NULL_KEY].values == -90)
    window.close()


@pytest.mark.parametrize("case", ["audio", "busy", "multiple", "remote", "directory", "missing", "unsupported"])
def test_audio_file_drop(application, tmp_path, monkeypatch, case):
    window = MainWindow()
    source = tmp_path / "我的 mixture.WAV"
    sf.write(source, np.zeros((100, 2), dtype=np.float32), 16000)
    urls = [QUrl.fromLocalFile(str(source))]
    if case == "busy":
        window.controls.setEnabled(False)
    elif case == "multiple":
        urls *= 2
    elif case == "remote":
        urls = [QUrl("https://example.com/audio.wav")]
    elif case == "directory":
        urls = [QUrl.fromLocalFile(str(tmp_path))]
    elif case == "missing":
        urls = [QUrl.fromLocalFile(str(tmp_path / "missing.wav"))]
    elif case == "unsupported":
        source = tmp_path / "config.yaml"
        source.write_text("test")
        urls = [QUrl.fromLocalFile(str(source))]
    mime = QMimeData()
    mime.setUrls(urls)
    launched = []
    monkeypatch.setattr(window, "launch", launched.append)
    window.input.setText("previous.wav")
    drag = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.dragEnterEvent(drag)
    drop = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.dropEvent(drop)
    assert drag.isAccepted() == (case == "audio")
    assert drop.isAccepted() == (case == "audio")
    assert not window.input.acceptDrops()
    if case == "audio":
        assert window.input.text() == str(source)
        assert launched == [{"input": str(source), "preview": True}]
    else:
        assert window.input.text() == "previous.wav"
        assert not launched
    window.close()


def test_pending_overlay_tracks_finalized_columns_not_amplitude(application):
    window = MainWindow()
    window.prepare(np.full((513, 30), -90, dtype=np.float32), 1600, 8000, ["vocals"])
    window.show()
    application.processEvents()
    card = window.cards["vocals"]
    assert not window.cards[MIXTURE_KEY].pending_background.isVisible()
    assert card.pending_background.isVisible()
    assert card.pending_hatching.isVisible()
    assert card.pending_background.rect().width() == 5
    silent = np.full((513, 3), -90, dtype=np.float32)
    window.update_stem("vocals", 0, silent)
    assert card.ready_columns == 3
    assert card.pending_background.rect().left() == .5
    assert card.ready_boundary.value() == .5
    assert card.ready_boundary.isVisible()
    window.update_phase("Cancelled")
    assert card.pending_background.isVisible()
    window.update_phase("Failed")
    assert card.pending_background.isVisible()
    card.plot.setXRange(0, .4, padding=0)
    assert card.pending_background.rect().left() == .5
    window.update_stem("vocals", 3, np.full((513, 27), -90, dtype=np.float32))
    assert not card.pending_background.isVisible()
    assert not card.pending_hatching.isVisible()
    assert not card.ready_boundary.isVisible()
    assert window.cards[NULL_KEY].pending_background.isVisible()
    window.close()


def test_playback_button_label_and_macchiato_colors(application, monkeypatch):
    from app.tests.test_player import FakeSink

    window = MainWindow()
    try:
        assert window.pause_playback.text() == "Play"
        assert not window.pause_playback.isEnabled()
        window.prepare(np.full((513, 30), -20, dtype=np.float32), 1000, 4000, [])
        window.receive_audio(MIXTURE_KEY, 0, np.full((2, 4000), .1, dtype=np.float32))
        monkeypatch.setattr(window.player, "create_sink", lambda channels: FakeSink())
        window.show()
        application.processEvents()
        for label, color in [("Play", "#a6da95"), ("Pause", "#ed8796"), ("Play", "#a6da95")]:
            assert window.pause_playback.text() == label
            assert window.pause_playback.isEnabled()
            assert window.pause_playback.palette().color(QPalette.ColorRole.Button).name() == color
            assert window.pause_playback.palette().color(QPalette.ColorRole.ButtonText).name() == "#24273a"
            window.pause_playback.click()
    finally:
        window.close()


def test_macchiato_theme_and_magma_spectrogram(application):
    window = MainWindow()
    values = np.linspace(-90, 0, 513 * 30, dtype=np.float32).reshape(513, 30)
    window.prepare(values, 1600, 8000, ["vocals"])
    window.show()
    application.processEvents()
    assert window.palette().color(QPalette.ColorRole.Window).name() == "#24273a"
    assert window.palette().color(QPalette.ColorRole.WindowText).name() == "#cad3f5"
    for card in window.cards.values():
        np.testing.assert_array_equal(card.image.lut, pg.colormap.get("magma").getLookupTable(nPts=256))
        np.testing.assert_array_equal(card.image.getLevels(), [-90, 0])
        assert card.playhead.pen.color().name() == "#8bd5ca"
        assert card.plot.getAxis("left").textPen().color().name() == "#b8c0e0"
        assert card.plot.getAxis("left").labelText == "Frequency"
        assert card.plot.getAxis("bottom").labelText == "Time"
    window.close()


def test_pane_dividers_resize_without_resetting_playback(application, monkeypatch):
    from app.tests.test_player import FakeSink

    window = MainWindow()
    values = np.full((513, 30), -20, dtype=np.float32)
    window.prepare(values, 1600, 16000, ["vocals"])
    window.receive_audio("vocals", 0, np.zeros((2, 16000), dtype=np.float32))
    monkeypatch.setattr(window.player, "create_sink", lambda channels: FakeSink())
    window.player.seek("vocals", 1.0)
    window.spectrogram_layout.setCurrentText("Fit all")
    window.show()
    application.processEvents()
    cards = dict(window.cards)
    controls_width = window.controls.width()
    for splitter, offset in [
        (window.pane_splitter, QPoint(70, 0)),
        (window.status_splitter, QPoint(0, -60)),
    ]:
        before = splitter.sizes()
        handle = splitter.handle(1)
        center = handle.rect().center()
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(handle, center + offset)
        QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
        application.processEvents()
        assert splitter.sizes() != before
        assert all(size > 0 for size in splitter.sizes())
        assert not splitter.childrenCollapsible()
    assert window.controls.width() > controls_width
    assert window.cards == cards
    assert window.player.playing and window.player.position == 1600
    assert window.spectrogram_scroll.verticalScrollBar().maximum() == 0
    np.testing.assert_array_equal(window.cards[MIXTURE_KEY].values, values)
    window.close()


@pytest.mark.parametrize("pixels", [False, True])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("target", ["batch", "overlap", "device", "input", "divider", "gutter", "viewport", "scrollbar"])
def test_config_wheel_scrolls_without_changing_fields(application, pixels, direction, enabled, target):
    window = MainWindow()
    try:
        window.resize(1180, 600)
        window.show()
        application.processEvents()
        window.controls.setEnabled(enabled)
        scroll = window.settings_scroll
        scrollbar = scroll.verticalScrollBar()
        assert scrollbar.maximum() > 0
        scrollbar.setValue(scrollbar.maximum() // 2)
        before = scrollbar.value()
        values = window.batch.value(), window.overlap.value(), window.device.currentText()
        widget = {
            "batch": window.batch, "overlap": window.overlap, "device": window.device, "input": window.input,
            "divider": window.pane_splitter.handle(1), "viewport": scroll.viewport(),
            "gutter": window.pane_splitter.widget(0), "scrollbar": scrollbar,
        }[target]
        position = widget.rect().center()
        event = QWheelEvent(
            QPointF(position), QPointF(widget.mapToGlobal(position)),
            QPoint(0, direction * 30) if pixels else QPoint(),
            QPoint() if pixels else QPoint(0, direction * 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate if pixels else Qt.ScrollPhase.NoScrollPhase, False,
        )
        QApplication.sendEvent(widget, event)
        assert (scrollbar.value() - before) * direction < 0
        assert (window.batch.value(), window.overlap.value(), window.device.currentText()) == values
    finally:
        window.close()


def test_spectrogram_pane_has_no_gap_beside_divider(application):
    window = MainWindow()
    try:
        window.resize(1180, 600)
        window.prepare(np.full((513, 30), -20, dtype=np.float32), 1600, 16000, ["vocals", "bass", "drums", "other"])
        window.show()
        application.processEvents()
        main_pane = window.pane_splitter.widget(1)
        divider = window.pane_splitter.handle(1)
        assert main_pane.geometry().left() == divider.geometry().right() + 1
        assert window.spectrogram_scroll.mapTo(main_pane, QPoint(0, 0)).x() == 0
        assert main_pane.layout().contentsMargins().left() == 0
    finally:
        window.close()


def test_icon_and_spacebar_playback(application, monkeypatch):
    from app.tests.test_player import FakeSink

    window = MainWindow()
    assert not window.windowIcon().isNull()
    icon = QImage(str(Path(__file__).parents[1] / "assets" / "subroformer-icon.png"))
    assert icon.width() == icon.height() == 1024
    assert icon.hasAlphaChannel()
    assert icon.pixelColor(0, 0).alpha() == 0
    assert icon.pixelColor(1023, 1023).alpha() == 0
    assert icon.pixelColor(512, 96).alpha() == 255
    assert icon.pixelColor(512, 96).name() == "#ffffff"
    assert icon.pixelColor(96, 512).alpha() == 255
    window.prepare(np.full((513, 30), -20, dtype=np.float32), 1600, 16000, ["vocals"])
    window.receive_audio("vocals", 0, np.zeros((2, 16000), dtype=np.float32))
    monkeypatch.setattr(window.player, "create_sink", lambda channels: FakeSink())
    window.show()
    window.activateWindow()
    window.pause_playback.setFocus()
    application.processEvents()
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert window.player.stem == MIXTURE_KEY and not window.player.playing
    window.player.seek("vocals", 1.0)
    QTest.keyClick(window.pause_playback, Qt.Key.Key_Space)
    assert not window.player.playing
    position = window.player.position
    QTest.keyClick(window.pause_playback, Qt.Key.Key_Space)
    assert window.player.playing and window.player.position == position
    field = window.findChild(QLineEdit)
    field.setFocus()
    field.clear()
    QTest.keyClick(field, Qt.Key.Key_Space)
    assert field.text() == " " and window.player.playing
    assert not window.playback_shortcut.autoRepeat()
    window.close()


def test_stem_key_labels_and_backtick_mute(application, monkeypatch):
    from app.tests.test_player import FakeSink

    window = MainWindow()
    try:
        stems = ["vocals", "bass", "drums", "other"]
        window.prepare(np.full((513, 30), -20, dtype=np.float32), 1000, 4000, stems)
        window.receive_audio(MIXTURE_KEY, 0, np.full((2, 4000), .8, dtype=np.float32))
        for stem in stems:
            window.receive_audio(stem, 0, np.full((2, 4000), .1, dtype=np.float32))
        monkeypatch.setattr(window.player, "create_sink", lambda channels: FakeSink())
        window.show()
        window.activateWindow()
        window.pause_playback.setFocus()
        application.processEvents()
        for number, (stem, button) in enumerate(window.stem_buttons.items(), 1):
            assert button.text().endswith(f"({number})")
            QTest.keyClick(window, Qt.Key.Key_QuoteLeft)
            assert not window.player.selected
            QTest.keyClick(window, getattr(Qt.Key, f"Key_{number}"))
            assert button.isChecked()
        assert window.mute_all.text() == "Mute all (`)"
        assert not window.mute_shortcut.autoRepeat()
        assert "Toggle stems" not in window.findChild(QLabel, "infoHint").text()
        window.player.seek_time(.5)
        QTest.keyClick(window, Qt.Key.Key_QuoteLeft)
        assert not window.player.selected
        assert window.player.playing and window.player.position == 500
        field = window.input
        field.setFocus()
        field.clear()
        QTest.keyClick(field, Qt.Key.Key_QuoteLeft)
        assert field.text() == "`"
    finally:
        window.close()


def test_memory_display_follows_selection_and_ignores_stale_readings(application):
    window = MainWindow()
    window.device.setCurrentText("cpu")
    window.update_memory(MemoryUsage("cpu", "cpu", 2_000_000_000, 40, detail="CPU detail"))
    assert "2.00 GB" in window.memory_status.text()
    assert "40.0%" in window.memory_status.text()
    assert window.memory_status.toolTip() == "CPU detail"
    window.device.setCurrentText("cuda:1")
    window.update_memory(MemoryUsage("cpu", "cpu", 3_000_000_000, 50))
    assert "reading" in window.memory_status.text()
    assert window.memory_monitor.requested == "cuda:1"
    window.close()
    assert not window.memory_monitor.isRunning()


@pytest.mark.parametrize("config_exists,checkpoint_exists", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_worker_real_inference_and_export(application, tmp_path, monkeypatch, config_exists, checkpoint_exists, dtype):
    from inference import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_separator
    from tests.legacy_overlap_add import separate_overlap_add as legacy_overlap_add

    config = tiny_config()
    config.model.num_stems = 4
    config_path = tmp_path / "config.yaml"
    checkpoint = tmp_path / "weights.pt"
    source = tmp_path / "mixture.wav"
    OmegaConf.save(config, config_path)
    torch.save(build_model(config).to(dtype).state_dict(), checkpoint)
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs["filename"])
        assert not kwargs["local_files_only"]
        path = config_path if kwargs["filename"] == DEFAULT_CONFIG else checkpoint
        size = path.stat().st_size
        with kwargs["tqdm_class"](total=size, disable=True) as bar:
            bar.update(size // 2)
            bar.update(size - size // 2)
        return str(path)

    monkeypatch.setattr("inference.hf_hub_download", download)
    sf.write(source, np.random.default_rng(4).normal(0, .01, (1500, 2)), 1600)
    worker = InferenceWorker({
        "input": str(source), "output": str(tmp_path / "outputs"),
        "config": str(config_path if config_exists else tmp_path / "missing.yaml"),
        "checkpoint": str(checkpoint if checkpoint_exists else tmp_path / "missing.pt"),
        "device": "cpu", "offline": False,
        "batch": 2, "overlap": 4,
    })
    errors, exports, updates, progress, chunks, timings, phases = [], [], [], [], [], [], []
    worker.failed.connect(errors.append)
    worker.completed.connect(exports.append)
    worker.updated.connect(lambda stem, first, values: updates.append((stem, first, values)))
    worker.progress.connect(progress.append)
    worker.timing.connect(lambda elapsed, eta: timings.append((elapsed, eta)))
    rtfs = []
    worker.rtf_updated.connect(rtfs.append)
    worker.phase_changed.connect(phases.append)
    artifacts = []
    worker.artifact_progress.connect(lambda name, completed, total: artifacts.append((name, completed, total)))
    worker.audio_ready.connect(lambda stem, start, audio: chunks.append((stem, start, audio)))
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(worker.requestInterruption)
    worker.finished.connect(loop.quit)
    timer.start(60000)
    worker.start()
    loop.exec()
    worker.wait()
    timer.stop()
    assert not errors
    assert len(exports) == 1
    assert updates and progress[-1] == 100
    assert timings[0] == (0, None)
    assert timings[-1][0] > 0 and timings[-1][1] == 0
    assert len(rtfs) == len(timings) - 1
    assert rtfs[-1] == pytest.approx(timings[-1][0] / (1500 / 1600))
    assert all(eta >= 0 for elapsed, eta in timings[1:])
    assert phases == ["Loading model", "Separating", "Saving stems"]
    assert downloads == [name for name, exists in [(DEFAULT_CONFIG, config_exists), (DEFAULT_CHECKPOINT, checkpoint_exists)] if not exists]
    assert {name for name, completed, total in artifacts} == set(downloads)
    for name in downloads:
        assert any(0 < completed < total for artifact, completed, total in artifacts if artifact == name and total)
    mixture_chunks = [chunk for name, start, chunk in chunks if name == MIXTURE_KEY]
    assert len(mixture_chunks) == 1
    np.testing.assert_array_equal(mixture_chunks[0].T, sf.read(source, dtype="float32", always_2d=True)[0])
    paths = list(Path(exports[0]).glob("*.wav"))
    assert len(paths) == 4
    reference_model, _ = load_separator(config_path, checkpoint, "cpu", preserve_dtype=True)
    mixture_audio = sf.read(source, dtype="float32", always_2d=True)[0].T
    with torch.autocast("cpu", dtype=torch.float16, enabled=dtype == torch.float16):
        expected = legacy_overlap_add(reference_model, mixture_audio, 128, 32, 2)
    for path in paths:
        audio, rate = sf.read(path, always_2d=True)
        assert audio.shape == (1500, 2) and rate == 1600
        assert np.isfinite(audio).all()
        stem = path.stem.split("-", 1)[1]
        np.testing.assert_array_equal(audio.T, expected[list(config.training.instruments).index(stem)])
        streamed = [chunk for name, start, chunk in chunks if name == stem]
        np.testing.assert_array_equal(np.concatenate(streamed, axis=-1).T, audio)
    mixture_audio = sf.read(source, dtype="float32", always_2d=True)[0].T
    residual = mixture_audio - sum(sf.read(path, dtype="float32", always_2d=True)[0].T for path in paths)
    expected_null = Spectrogram(residual.shape[-1])
    expected_null.append(residual)
    actual_null = np.full_like(expected_null.values, -90)
    for stem, first, values in updates:
        if stem == NULL_KEY:
            actual_null[:, first:first + values.shape[1]] = values
    np.testing.assert_allclose(actual_null, expected_null.values, atol=1e-3)
