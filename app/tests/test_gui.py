import os
from pathlib import Path
from types import SimpleNamespace

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

from app.gui import InferenceWorker, MainWindow, Spectrogram, STYLE, MIXTURE_KEY
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
    assert set(window.cards) == {MIXTURE_KEY, "vocals", "drums"}
    assert set(window.stem_buttons) == {MIXTURE_KEY, "vocals", "bass", "drums", "other"}
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


def test_macchiato_frames_and_buttons_with_magma_spectrograms(application):
    from app.gui import PLAYBACK_COLORS

    window = MainWindow()
    values = np.linspace(-90, 0, 513 * 30, dtype=np.float32).reshape(513, 30)
    window.prepare(values, 1600, 8000, ["vocals", "bass", "drums", "other"])
    window.show()
    application.processEvents()
    assert window.palette().color(QPalette.ColorRole.Window).name() == "#24273a"
    assert window.palette().color(QPalette.ColorRole.WindowText).name() == "#cad3f5"
    assert window.stem_tabs.count() == 7
    buttons = {**window.stem_buttons, "mute": window.mute_all, "invert": window.invert_button}
    assert len({button.property("accent") for button in buttons.values()}) == 7
    for key, button in buttons.items():
        assert button.property("accent") == PLAYBACK_COLORS[key]
    for key, card in window.cards.items():
        color = PLAYBACK_COLORS[key]
        assert f'QFrame#card[selected="true"] {{ border-color: {color}; }}' in card.styleSheet()
        expected = pg.colormap.get("magma")
        np.testing.assert_array_equal(card.image.lut, expected.getLookupTable(nPts=256))
        np.testing.assert_array_equal(card.image.getLevels(), [-90, 0])
        assert card.playhead.pen.color().name() == "#8bd5ca"
        assert card.plot.getAxis("left").textPen().color().name() == "#b8c0e0"
        assert card.plot.getAxis("left").labelText == "Frequency"
        assert card.plot.getAxis("bottom").labelText == "Time"
    window.close()


def test_invert_button_remains_available_when_all_stems_are_muted(application):
    window = MainWindow()
    try:
        assert not window.invert_button.isEnabled()
        stems = ["vocals", "bass", "drums", "other"]
        window.prepare(np.full((513, 30), -90, dtype=np.float32), 1600, 8000, stems)
        window.player.select_stems(["vocals"])
        window.invert_button.click()
        assert set(window.player.selected) == {"bass", "drums", "other"}
        window.mute_all.click()
        assert not window.player.selected
        assert window.invert_button.isEnabled()
        window.invert_button.click()
        assert set(window.player.selected) == set(stems)
        window.invert_button.click()
        assert not window.player.selected
    finally:
        window.close()


def test_pane_dividers_resize_without_resetting_playback(application, monkeypatch):
    from app.tests.test_player import FakeSink

    window = MainWindow()
    values = np.full((513, 30), -20, dtype=np.float32)
    window.prepare(values, 1600, 16000, ["vocals"])
    window.receive_audio(MIXTURE_KEY, 0, np.zeros((2, 16000), dtype=np.float32))
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
    window.receive_audio(MIXTURE_KEY, 0, np.zeros((2, 16000), dtype=np.float32))
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
        assert "~" not in window.mute_all.toolTip()
        assert window.invert_button.text() == "Invert (~)"
        assert "press ~ to invert playing stems" in window.invert_button.toolTip()
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


@pytest.mark.parametrize("modifiers", [Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier])
def test_tilde_inverts_playback(application, monkeypatch, modifiers):
    from app.tests.test_player import FakeSink

    window = MainWindow()
    try:
        window.invert_playback()
        assert not window.player.selected
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
        window.player.select_stems(["vocals"])
        window.player.seek_time(.5)
        QTest.keyClick(window, Qt.Key.Key_AsciiTilde, modifiers)
        assert set(window.player.selected) == {"bass", "drums", "other"}
        assert not window.stem_buttons["vocals"].isChecked()
        assert window.player.playing and window.player.position == 500
        QTest.keyClick(window, Qt.Key.Key_AsciiTilde, modifiers)
        assert window.player.selected == ("vocals",)
        window.switch_playback(MIXTURE_KEY)
        QTest.keyClick(window, Qt.Key.Key_AsciiTilde, modifiers)
        assert not window.player.selected
        assert window.player.playing and window.player.position == 500
        QTest.keyClick(window, Qt.Key.Key_AsciiTilde, modifiers)
        assert window.player.stem == MIXTURE_KEY
        assert window.stem_buttons[MIXTURE_KEY].isChecked()
        window.player.pause()
        QTest.keyClick(window, Qt.Key.Key_AsciiTilde, modifiers)
        assert not window.player.selected and not window.player.playing
        assert window.player.position == 500
        assert not window.invert_shortcut.autoRepeat()
        window.input.setFocus()
        window.input.clear()
        QTest.keyClick(window.input, Qt.Key.Key_AsciiTilde, modifiers)
        assert window.input.text() == "~"
        assert not window.player.selected
        window.prepare(np.full((513, 30), -20, dtype=np.float32), 1000, 4000, [])
        window.invert_playback()
        assert not window.player.selected
        window.invert_playback()
        assert window.player.stem == MIXTURE_KEY
    finally:
        window.close()


def test_memory_display_follows_selection_and_ignores_stale_readings(application):
    window = MainWindow()
    window.backend.setCurrentIndex(window.backend.findData("torch"))
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


def test_memory_display_tracks_coreml_backend(application):
    window = MainWindow()
    try:
        window.device.setCurrentText("mps")
        window.backend.setCurrentIndex(window.backend.findData("coreml"))
        assert window.memory_monitor.backend == "coreml"
        window.update_memory(MemoryUsage("mps", "mps", 2_000_000_000, backend="coreml"))
        assert "2.00 GB" in window.memory_status.text()
        window.update_memory(MemoryUsage("mps", "mps", 0))
        assert "2.00 GB" in window.memory_status.text()
        window.backend.setCurrentIndex(window.backend.findData("torch"))
        assert window.memory_monitor.backend == "torch"
        window.update_memory(MemoryUsage("mps", "mps", 2_000_000_000, backend="coreml"))
        assert "reading" in window.memory_status.text()
    finally:
        window.close()


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
    assert {stem for stem, first, values in updates} == {"vocals", "bass", "drums", "other"}


def test_click_maps_zoomed_spectrogram_time(application, monkeypatch):
    window = MainWindow()
    window.prepare(np.full((513, 30), -20, dtype=np.float32), 1600, 16000, ["vocals"])
    window.show()
    application.processEvents()
    card = window.cards["vocals"]
    view = card.plot.getViewBox()
    view.setRange(xRange=(2, 6), padding=0)
    requested = []
    card.seek_requested.connect(lambda stem, seconds: requested.append((stem, seconds)))

    class Click:
        def button(self):
            return Qt.MouseButton.LeftButton

        def scenePos(self):
            return view.mapViewToScene(QPointF(3.5, 400))

    card.clicked(Click())
    assert requested[0][0] == "vocals"
    assert requested[0][1] == pytest.approx(3.5)
    assert window.player.position == 0 and not window.player.playing
    assert not window.pause_playback.isEnabled()
    window.receive_audio(MIXTURE_KEY, 0, np.zeros((2, 1600), dtype=np.float32))
    assert window.pause_playback.isEnabled()
    window.close()


def test_progress_eta_and_terminal_states(application):
    window = MainWindow()
    window.update_phase("Separating")
    assert "estimating" in window.timing_status.text()
    window.update_timing(12, 28)
    assert window.timing_status.text() == "Elapsed 00:12 · ETA 00:28"
    for phase in ["Cancelled", "Failed", "Saving stems", "Complete"]:
        window.update_phase(phase)
        assert phase in window.timing_status.text()
        assert "ETA —" in window.timing_status.text()
    window.close()


@pytest.mark.parametrize("rtf,description", [(0.5, "real-time or faster"), (1.0, "real-time or faster"), (1.2, "slower than real-time")])
def test_rtf_status_reports_throughput(application, rtf, description):
    window = MainWindow()
    assert window.rtf_status.text() == "RTF —"
    window.update_phase("Separating")
    window.update_rtf(rtf)
    assert window.rtf_status.text() == f"RTF (so far) {rtf:.3f} · {description}"
    window.update_phase("Cancelled")
    assert "so far" in window.rtf_status.text()
    window.update_phase("Complete")
    assert window.rtf_status.text() == f"RTF {rtf:.3f} · {description}"
    window.separation_rtf = None
    window.update_phase("Loading / preparing")
    assert window.rtf_status.text() == "RTF —"
    window.close()


def test_stem_tabs_mix_and_spectrograms_only_seek(application, monkeypatch):
    from app.tests.test_player import FakeSink

    window = MainWindow()
    try:
        stems = ["vocals", "bass", "drums", "other"]
        window.prepare(np.full((513, 30), -20, dtype=np.float32), 1000, 4000, stems)
        window.receive_audio(MIXTURE_KEY, 0, np.full((2, 4000), .8, dtype=np.float32))
        for stem in stems:
            window.receive_audio(stem, 0, np.full((2, 4000), .1, dtype=np.float32))
        monkeypatch.setattr(window.player, "create_sink", lambda channels: FakeSink())
        assert all(button.isChecked() for button in window.stem_buttons.values())
        window.stem_buttons[MIXTURE_KEY].click()
        assert all(button.isChecked() for button in window.stem_buttons.values())
        window.player.seek_time(.5)
        np.testing.assert_allclose(np.frombuffer(window.player.sink.writes, dtype=np.float32), .8)
        window.stem_buttons["vocals"].click()
        assert not window.stem_buttons[MIXTURE_KEY].isChecked()
        assert not window.stem_buttons["vocals"].isChecked()
        assert set(window.player.selected) == {"bass", "drums", "other"}
        assert window.player.playing and window.player.position == 500
        np.testing.assert_allclose(np.frombuffer(window.player.sink.writes, dtype=np.float32), .3, atol=1e-6)
        selected = window.player.selected
        window.cards["vocals"].seek_requested.emit("vocals", 1.5)
        assert window.player.selected == selected and window.player.position == 1500
        window.cards["drums"].seek_requested.emit("drums", 1.5)
        assert window.player.selected == selected and window.player.position == 1500
        window.stem_buttons["vocals"].click()
        assert all(button.isChecked() for button in window.stem_buttons.values())
        for stem in stems:
            window.stem_buttons[stem].click()
        assert not window.player.selected and window.player.playing
        window.player.pause()
        window.stem_buttons[MIXTURE_KEY].click()
        assert all(button.isChecked() for button in window.stem_buttons.values())
        assert not window.player.playing and window.player.position == 1500
        window.toggle_playback()
        assert window.player.playing
        sink = window.player.sink
        window.mute_all.click()
        assert not window.player.selected and window.player.playing
        assert window.player.position == 1500 and sink.disposed
        np.testing.assert_array_equal(np.frombuffer(window.player.sink.writes, dtype=np.float32), 0)
        window.player.sink.processed = 100000
        window.player.tick()
        assert window.player.position == 1600
        assert "Playing silently" in window.playback_status.text()
        assert window.pause_playback.text() == "Pause"
        assert not any(button.isChecked() for button in window.stem_buttons.values())
        assert not window.mute_all.isEnabled()
        window.stem_buttons[MIXTURE_KEY].click()
        assert window.mute_all.isEnabled()
        assert window.player.position == 1600 and window.player.playing
        window.player.pause()
        window.mute_all.click()
        assert not window.player.playing and window.player.position == 1600
    finally:
        window.close()


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
def test_app_defaults_prefer_accelerator_and_fp16(application, monkeypatch, device):
    monkeypatch.setattr("app.gui.resolve_device", lambda requested: torch.device(device))
    window = MainWindow()
    try:
        assert window.device.currentText() == device
        assert not window.mute_all.isEnabled()
        assert window.memory_monitor.requested == device
        assert window.precision.text() == "Not loaded"
        assert not window.controls.isAncestorOf(window.precision)
        assert window.precision.parentWidget().objectName() == "infoMetric"
    finally:
        window.close()


@pytest.mark.parametrize("dtype,expected", [(torch.float16, "FP16"), (torch.float32, "FP32"), (torch.bfloat16, "BF16")])
def test_worker_precision_follows_checkpoint(monkeypatch, dtype, expected):
    calls = []
    precisions = []
    monkeypatch.setattr("app.gui.use_coreml", lambda *args: False)
    monkeypatch.setattr("app.gui.resolve_device", lambda requested: torch.device("cpu"))
    monkeypatch.setattr("app.gui.resolve_pretrained_paths", lambda *args, **kwargs: ("config.yaml", "checkpoint.pt"))

    def load(*args, **kwargs):
        assert kwargs["preserve_dtype"] is True
        return build_model(tiny_config()).to(dtype), tiny_config()

    def stop_at_audio(*args):
        raise RuntimeError("stop before audio")

    monkeypatch.setattr("app.gui.load_separator", load)
    monkeypatch.setattr("app.gui.load_audio", stop_at_audio)
    worker = InferenceWorker({"config": "", "checkpoint": "", "device": "auto", "offline": True, "input": "song.wav"})
    worker.status.connect(calls.append)
    worker.precision_changed.connect(precisions.append)
    with pytest.raises(RuntimeError, match="stop before audio"):
        worker.process()
    assert f"cpu · {expected}" in calls[-1]
    assert precisions == [f"PyTorch · {expected}"]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_worker_loads_coreml_with_selected_dsp(monkeypatch, device):
    calls = []
    precisions = []
    closed = []
    def select_backend(backend, *args):
        assert backend == "coreml"
        return True

    monkeypatch.setattr("app.gui.use_coreml", select_backend)
    monkeypatch.setattr("app.gui.resolve_device", lambda requested: torch.device(device))
    monkeypatch.setattr("app.gui.resolve_pretrained_paths", lambda *args, **kwargs: pytest.fail("PyTorch download"))

    def load(**kwargs):
        assert kwargs["local_files_only"] is True
        assert callable(kwargs["download_progress"])
        assert kwargs["dsp_device"] == device
        model = build_model(tiny_config())
        model.close = lambda: closed.append(True)
        return model, tiny_config()

    def stop_at_audio(*args):
        raise RuntimeError("stop before audio")

    monkeypatch.setattr("app.gui.load_coreml_process", load)
    monkeypatch.setattr("app.gui.load_audio", stop_at_audio)
    worker = InferenceWorker({"config": "", "checkpoint": "", "device": "auto", "backend": "coreml", "offline": True, "input": "song.wav"})
    worker.status.connect(calls.append)
    worker.precision_changed.connect(precisions.append)
    with pytest.raises(RuntimeError, match="stop before audio"):
        worker.process()
    assert f"Core ML + {device.upper()} DSP · FP16" in calls[-1]
    assert precisions == ["Core ML · FP16"]
    assert closed == [True]


@pytest.mark.parametrize("system,version,expected", [
    ("darwin", "15.0", "coreml"), ("darwin", "26.6", "coreml"),
    ("darwin", "14.7", "torch"), ("linux", "", "torch"), ("win32", "", "torch"),
])
def test_backend_selection_reaches_worker(application, tmp_path, monkeypatch, system, version, expected):
    monkeypatch.setattr("app.gui.sys", SimpleNamespace(platform=system))
    monkeypatch.setattr("app.gui.platform.mac_ver", lambda: (version, (), ""))
    audio = tmp_path / "mixture.wav"
    sf.write(audio, np.zeros((128, 2), dtype=np.float32), 1600)
    window = MainWindow()
    calls = []
    monkeypatch.setattr(window, "launch", calls.append)
    try:
        assert window.backend.currentData() == expected
        window.input.setText(str(audio))
        window.backend.setCurrentIndex(window.backend.findData("coreml"))
        window.device.setCurrentText("mps")
        window.separate()
        assert calls[0]["backend"] == "coreml"
        assert calls[0]["device"] == "mps"
    finally:
        window.close()


def test_precision_info_persists_and_resets(application, monkeypatch):
    monkeypatch.setattr(InferenceWorker, "start", lambda self: None)
    window = MainWindow()
    try:
        window.launch({"input": "song.wav"})
        assert window.precision.text() == "Loading…"
        window.worker.precision_changed.emit("Core ML · FP16")
        window.worker.status.emit("Separating…")
        window.update_phase("Complete")
        assert window.precision.text() == "Core ML · FP16"
        window.worker.finished.emit()
        window.launch({"input": "other.wav", "preview": True})
        assert window.precision.text() == "Not loaded"
    finally:
        if window.worker is not None:
            window.worker.finished.emit()
        window.close()


def test_precision_card_visible_in_info_pane(application):
    window = MainWindow()
    try:
        window.precision.setText("Core ML · FP16")
        window.show()
        application.processEvents()
        assert window.precision.visibleRegion().boundingRect().height() == window.precision.height()
    finally:
        window.close()


def test_artifact_progress_bar_and_separation_reset(application):
    window = MainWindow()
    try:
        window.update_artifact_progress("checkpoints/model.safetensors", 0, None)
        assert window.progress.maximum() == 0
        assert "model.safetensors" in window.status.text()
        window.update_artifact_progress("checkpoints/model.safetensors", 4_000_000_000, 8_000_000_000)
        assert window.progress.maximum() == 100 and window.progress.value() == 50
        assert "4000.00 / 8000.00 MB" in window.status.text()
        window.update_artifact_progress("checkpoints/model.safetensors", 8_000_000_000, 8_000_000_000)
        assert window.progress.value() == 100
        window.update_phase("Loading model")
        assert window.progress.maximum() == 0
        window.prepare(np.full((513, 30), -90, dtype=np.float32), 1600, 1600, ["vocals"])
        assert window.progress.maximum() == 100 and window.progress.value() == 0
        assert window.progress.format() == "Separation · %p%"
    finally:
        window.close()


def test_download_eta_ignores_cached_initial_bytes(application, monkeypatch):
    window = MainWindow()
    now = [0.0]
    monkeypatch.setattr("app.gui.time.monotonic", lambda: now[0])
    try:
        window.update_artifact_progress("Core ML model", 0, None)
        assert "estimating" in window.timing_status.text()
        now[0] = 1
        window.update_artifact_progress("Core ML model", 50, 100)
        assert "estimating" in window.timing_status.text()
        now[0] = 3
        window.update_artifact_progress("Core ML model", 75, 100)
        assert window.progress.value() == 75
        assert window.timing_status.text() == "Download 00:03 · ETA 00:02"
        window.update_artifact_progress("Core ML model", 100, 100)
        assert "ETA 00:00" in window.timing_status.text()
        window.update_phase("Loading model")
        assert "ETA —" in window.timing_status.text()
        window.update_artifact_progress("another model", 0, None)
        assert "estimating" in window.timing_status.text()
    finally:
        window.close()


def test_mixture_preview_emits_audio_and_click_plays(application, tmp_path, monkeypatch):
    from app.tests.test_player import FakeSink

    source = tmp_path / "preview.wav"
    sf.write(source, np.full((4000, 2), .1, dtype=np.float32), 1600)
    worker = InferenceWorker({"preview": True, "input": str(source)})
    window = MainWindow()
    worker.prepared.connect(window.prepare)
    worker.audio_ready.connect(window.receive_audio)
    worker.process()
    assert window.player.buffers[MIXTURE_KEY].frames == 4000
    monkeypatch.setattr(window.player, "create_sink", lambda channels: FakeSink())
    window.cards[MIXTURE_KEY].seek_requested.emit("Input mixture", 1.0)
    assert window.player.stem == MIXTURE_KEY and window.player.playing
    assert window.player.position == 1600
    assert "Mixture" in window.playback_status.text()
    window.toggle_playback()
    assert not window.player.playing
    window.close()


@pytest.mark.parametrize("close_window", [False, True])
def test_launch_backpressures_audio_and_cancels_without_deadlock(application, monkeypatch, close_window):
    window = MainWindow()
    delivered = []
    acknowledged = []
    append = window.player.append

    def receive(stem, start, audio):
        append(stem, start, audio)
        delivered.append(start)
        if len(delivered) == 2:
            if close_window:
                window.close()
            else:
                window.cancel.click()
            assert window.isVisible()

    def process(worker):
        worker.prepared.emit(np.full((513, 4), -90, dtype=np.float32), 1000, 320, [])
        for start in range(0, 320, 32):
            worker.check_cancel()
            worker.audio_ready.emit(MIXTURE_KEY, start, np.zeros((2, 32), dtype=np.float32))
            acknowledged.append(len(delivered))

    monkeypatch.setattr(window.player, "append", receive)
    monkeypatch.setattr(InferenceWorker, "process", process)
    window.show()
    window.launch({"preview": True})
    worker = window.worker
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    try:
        assert not worker.isRunning()
        assert delivered == [0, 32]
        assert acknowledged == [1, 2]
        application.processEvents()
        assert window.worker is None
        assert window.isVisible() is not close_window
        assert window.memory_monitor.isRunning() is not close_window
    finally:
        window.close()


def test_cancelled_export_is_closed_and_not_reported_complete(application, tmp_path, monkeypatch):
    import app.gui as gui

    config = tiny_config()
    config.model.num_stems = 4
    model = build_model(config).eval()
    monkeypatch.setattr(gui, "resolve_pretrained_paths", lambda *args, **kwargs: ("config", "checkpoint"))
    monkeypatch.setattr(gui, "load_separator", lambda *args, **kwargs: (model, config))
    monkeypatch.setattr(gui, "load_audio", lambda *args: np.zeros((2, 400), dtype=np.float32))
    worker = InferenceWorker({
        "config": "", "checkpoint": "", "device": "cpu", "offline": True,
        "input": "unused.wav", "output": str(tmp_path), "batch": 1, "overlap": 4,
    })
    completed, failures, phases = [], [], []
    worker.completed.connect(completed.append)
    worker.failed.connect(failures.append)
    worker.phase_changed.connect(phases.append)
    worker.audio_ready.connect(
        lambda stem, start, audio: worker.requestInterruption() if stem != MIXTURE_KEY else None,
        Qt.ConnectionType.DirectConnection,
    )
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    worker.start()
    loop.exec()
    worker.wait()
    assert not completed and not failures and phases[-1] == "Cancelled"
    paths = list(tmp_path.glob("*/*.wav"))
    assert len(paths) == 4
    for path in paths:
        assert sf.info(path).frames == 32
        path.unlink()


def test_playback_storage_failure_requests_cancellation(application, monkeypatch):
    window = MainWindow()
    errors, cancellations = [], []

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(window.player, "append", fail)
    monkeypatch.setattr(window, "stop", lambda: cancellations.append(True))
    monkeypatch.setattr(window, "playback_error", errors.append)
    try:
        window.receive_audio(MIXTURE_KEY, 0, np.zeros((2, 32), dtype=np.float32))
        assert cancellations == [True]
        assert errors == ["Unable to store playback audio: disk full"]
    finally:
        window.close()


def test_fit_all_keeps_every_card_visible_and_preserves_data(application):
    window = MainWindow()
    mixture = np.full((513, 30), -20, dtype=np.float32)
    window.prepare(mixture, 1600, 16000, ["vocals", "bass", "drums", "other"])
    cards = dict(window.cards)
    window.show()
    for mode in ["Grid", "Fit all", "Stacked", "Fit all"]:
        window.spectrogram_layout.setCurrentText(mode)
        application.processEvents()
        assert window.cards == cards
        assert window.plots.count() == 5
        np.testing.assert_array_equal(window.cards[MIXTURE_KEY].values, mixture)
        if mode == "Fit all":
            viewport = window.spectrogram_scroll.viewport()
            assert window.spectrogram_scroll.verticalScrollBar().maximum() == 0
            for card in window.cards.values():
                top_left = card.mapTo(viewport, card.rect().topLeft())
                bottom_right = card.mapTo(viewport, card.rect().bottomRight())
                assert viewport.rect().contains(top_left)
                assert viewport.rect().contains(bottom_right)
    window.resize(1100, 720)
    application.processEvents()
    assert window.spectrogram_scroll.verticalScrollBar().maximum() == 0
    window.close()
