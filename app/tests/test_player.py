import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtMultimedia import QtAudio

from app.player import MIXTURE_KEY, NULL_KEY, StemAudio, StemPlayer


class FakeSink:
    def __init__(self):
        self.writes = bytearray()
        self.processed = 0
        self.current_state = QtAudio.State.IdleState
        self.current_error = QtAudio.Error.NoError
        self.disposed = False

    def setBufferSize(self, size):
        pass

    def start(self):
        return self

    def error(self):
        return self.current_error

    def state(self):
        return self.current_state

    def bytesFree(self):
        return 8000

    def write(self, data):
        self.writes.extend(data)
        self.current_state = QtAudio.State.ActiveState
        return len(data)

    def processedUSecs(self):
        return self.processed

    def reset(self):
        self.disposed = True

    def deleteLater(self):
        pass

    def suspend(self):
        self.current_state = QtAudio.State.SuspendedState

    def resume(self):
        self.current_state = QtAudio.State.ActiveState


@pytest.fixture
def player(monkeypatch):
    application = QCoreApplication.instance() or QCoreApplication([])
    player = StemPlayer()
    monkeypatch.setattr(player, "create_sink", lambda channels: FakeSink())
    player.reset(1000, ["vocals", "drums"])
    player.append("vocals", 0, np.full((2, 1000), .25, dtype=np.float32))
    player.append("drums", 0, np.full((2, 1000), -.5, dtype=np.float32))
    yield player
    player.release()
    assert application is not None


def test_pcm_interleaves_channels_without_clipping_source():
    audio = np.array([[2, .3], [-2, -.3]], dtype=np.float32)
    buffer = StemAudio()
    buffer.append(0, audio)
    np.testing.assert_array_equal(buffer.read(0, 2).reshape(-1), [2, -2, np.float32(.3), np.float32(-.3)])
    assert audio[0, 0] == 2
    assert buffer.frames == 2
    with pytest.raises(ValueError, match="contiguous"):
        buffer.append(0, audio)


def test_seek_and_pause_resume_preserve_position(player):
    player.seek("vocals", .25)
    sink = player.sink
    assert player.playing and player.position == 250
    assert len(sink.writes) == 750 * 2 * 4
    sink.processed = 100000
    player.pause()
    assert not player.playing and player.position == 350
    assert sink.state() == QtAudio.State.SuspendedState
    player.resume()
    assert player.playing and player.sink is sink
    assert player.position == 350


def test_playback_extends_only_while_running_then_stops_at_edge(player):
    player.seek("vocals", .5)
    sink = player.sink
    player.append("vocals", 1000, np.full((2, 300), .1, dtype=np.float32))
    player.tick()
    assert len(sink.writes) == 800 * 2 * 4
    sink.current_state = QtAudio.State.IdleState
    sink.processed = 800000
    player.tick()
    assert not player.playing and player.position == 1300
    assert player.sink is None and sink.disposed
    player.append("vocals", 1300, np.zeros((2, 50), dtype=np.float32))
    assert not player.playing
    player.resume()
    assert player.playing and player.position == 1300
    assert len(player.sink.writes) == 50 * 2 * 4


def test_future_seek_stays_at_edge_and_switch_discards_old_audio(player):
    player.seek("vocals", .2)
    previous = player.sink
    player.seek("drums", .3)
    assert previous.disposed
    assert player.stem == "drums"
    assert np.frombuffer(player.sink.writes, dtype=np.float32)[0] == -.5
    player.seek("vocals", 50)
    assert player.position == 1000 and not player.playing and player.sink is None
    player.seek("vocals", -2)
    assert player.position == 0


def test_reset_discards_playback_and_all_previous_stems(player):
    player.seek("vocals", 0)
    previous = player.sink
    files = [buffer.file for buffer in player.buffers.values()]
    player.reset(44100, ["bass"])
    assert previous.disposed
    assert all(file.closed for file in files)
    assert not player.playing and player.stem is None
    assert list(player.buffers) == ["bass"]


def test_device_failure_stops_playback(player):
    errors = []
    player.failed.connect(errors.append)
    player.seek("vocals", 0)
    player.sink.current_error = QtAudio.Error.IOError
    player.tick()
    assert errors and not player.playing and player.sink is None


def test_missing_device_reports_error(player, monkeypatch):
    def unavailable(channels):
        raise ValueError("No audio output device")

    monkeypatch.setattr(player, "create_sink", unavailable)
    errors = []
    player.failed.connect(errors.append)
    player.seek("vocals", 0)
    assert errors == ["No audio output device"]
    assert not player.playing


def test_mix_sums_enabled_stems_and_clips_only_final_output(player):
    player.reset(1000, ["vocals", "drums"])
    player.append("vocals", 0, np.full((2, 1000), 2, dtype=np.float32))
    player.append("drums", 0, np.full((2, 1000), -1.5, dtype=np.float32))
    player.select_stems(["vocals", "drums"])
    player.seek_time(.25)
    np.testing.assert_allclose(np.frombuffer(player.sink.writes, dtype=np.float32), .5)
    previous = player.sink
    previous.processed = 100000
    player.select_stems(["vocals"])
    assert previous.disposed and player.position == 350 and player.playing
    np.testing.assert_allclose(np.frombuffer(player.sink.writes, dtype=np.float32), 1)
    player.select_stems([])
    assert player.playing and player.available == 1000 and player.position == 350
    np.testing.assert_array_equal(np.frombuffer(player.sink.writes, dtype=np.float32), 0)
    player.sink.processed = 100000
    player.tick()
    assert player.position == 450
    player.pause()
    assert not player.playing and player.position == 450
    player.resume()
    assert player.playing
    player.sink.processed = 650000
    player.sink.current_state = QtAudio.State.IdleState
    player.tick()
    assert not player.playing and player.position == 1000


def test_null_waits_for_all_stems_and_uses_unclipped_samples(player):
    player.reset(1000, [MIXTURE_KEY, "vocals", "drums", NULL_KEY])
    player.append(MIXTURE_KEY, 0, np.full((2, 1000), .8, dtype=np.float32))
    player.select_stems(["vocals", "drums", NULL_KEY])
    assert player.available == 1000
    player.seek_time(0)
    np.testing.assert_allclose(np.frombuffer(player.sink.writes, dtype=np.float32), .8)
    player.append("vocals", 0, np.full((2, 600), 2, dtype=np.float32))
    assert player.buffers[NULL_KEY].frames == 0
    player.append("drums", 0, np.full((2, 400), -1.5, dtype=np.float32))
    assert player.buffers[NULL_KEY].frames == 400
    np.testing.assert_allclose(player.samples(NULL_KEY, 0, 400), .3, atol=1e-6)
    player.select_stems([NULL_KEY])
    assert player.available == 400
    np.testing.assert_allclose(np.frombuffer(player.sink.writes, dtype=np.float32), .3, atol=1e-6)
    player.append("drums", 400, np.full((2, 200), -1.5, dtype=np.float32))
    assert player.available == 600
    player.pause()
    player.select_stems(["vocals", NULL_KEY])
    assert not player.playing


def test_disk_backed_random_reads_preserve_float32():
    buffer = StemAudio()
    audio = np.random.default_rng(20).normal(size=(2, 150000)).astype(np.float32)
    try:
        buffer.append(0, audio[:, :70000])
        buffer.append(70000, audio[:, 70000:])
        assert not hasattr(buffer, "data")
        assert buffer.file.seek(0, 2) == audio.nbytes
        for start, end in [(140000, 150000), (12, 40), (65530, 70010), (0, 10)]:
            np.testing.assert_array_equal(buffer.read(start, end).T, audio[:, start:end])
    finally:
        buffer.close()


def test_residual_has_no_pcm_storage_and_preserves_subtraction_order(player):
    player.reset(1000, [MIXTURE_KEY, "vocals", "drums", NULL_KEY])
    sources = np.random.default_rng(4).normal(size=(3, 2, 1000)).astype(np.float32)
    for stem, audio in zip([MIXTURE_KEY, "vocals", "drums"], sources):
        player.append(stem, 0, audio)
    residual = player.buffers[NULL_KEY]
    assert not hasattr(residual, "file") and not hasattr(residual, "data")
    np.testing.assert_array_equal(
        player.samples(NULL_KEY, 125, 410).T,
        (sources[0] - sources[1] - sources[2])[:, 125:410],
    )


def test_disk_read_failure_stops_playback(player, monkeypatch):
    errors = []
    player.failed.connect(errors.append)

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(player.buffers["vocals"], "read", fail)
    player.seek("vocals", 0)
    assert not player.playing and player.sink is None
    assert errors == ["Unable to read playback audio: disk unavailable"]
