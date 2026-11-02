import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication
from PySide6.QtMultimedia import QAudioFormat, QtAudio

from app.player import MIXTURE_KEY, StemAudio, StemPlayer


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
    player.lookahead = 10
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


def test_future_seek_stays_at_edge_and_switch_keeps_sink_but_replaces_unplayed_audio(player):
    player.seek("vocals", .2)
    previous = player.sink
    player.seek("drums", .3)
    assert player.sink is previous and not previous.disposed
    assert player.stem == "drums"
    np.testing.assert_array_equal(np.frombuffer(previous.writes, dtype=np.float32)[-1400:], -.5)
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
    sink = player.sink
    sink.processed = 100000
    queued = len(sink.writes)
    player.select_stems(["vocals"])
    assert player.sink is sink and not sink.disposed and player.position == 350 and player.playing
    np.testing.assert_allclose(np.frombuffer(sink.writes, dtype=np.float32)[queued // 4:], 1)
    queued = len(sink.writes)
    player.select_stems([])
    assert player.playing and player.available == 1000 and player.position == 350
    np.testing.assert_array_equal(np.frombuffer(sink.writes, dtype=np.float32)[queued // 4:], 0)
    sink.processed = (queued // 8 + 100) * 1000
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


def test_all_stems_play_mixture_and_subsets_sum_only_selected_stems(player):
    player.reset(1000, [MIXTURE_KEY, "vocals", "drums", "bass"])
    player.append(MIXTURE_KEY, 0, np.full((2, 1000), .8, dtype=np.float32))
    player.select_stems(["vocals", "drums", "bass"])
    assert player.playback_sources == (MIXTURE_KEY,)
    assert player.available == 1000
    player.seek_time(0)
    sink = player.sink
    np.testing.assert_allclose(np.frombuffer(sink.writes, dtype=np.float32), .8)
    player.append("vocals", 0, np.full((2, 600), 2, dtype=np.float32))
    player.append("drums", 0, np.full((2, 400), -1.5, dtype=np.float32))
    player.select_stems(["vocals", "drums"])
    assert player.playback_sources == ("vocals", "drums")
    assert player.available == 400
    np.testing.assert_allclose(np.frombuffer(sink.writes, dtype=np.float32)[2000:], .5)
    player.select_stems([])
    assert player.playback_sources == ()
    np.testing.assert_array_equal(np.frombuffer(sink.writes, dtype=np.float32)[2800:], 0)


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


def test_disk_read_failure_stops_playback(player, monkeypatch):
    errors = []
    player.failed.connect(errors.append)

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(player.buffers["vocals"], "read", fail)
    player.seek("vocals", 0)
    assert not player.playing and player.sink is None
    assert errors == ["Unable to read playback audio: disk unavailable"]


@pytest.mark.parametrize("sample_format,dtype,expected", [
    (QAudioFormat.SampleFormat.Float, np.float32, .25),
    (QAudioFormat.SampleFormat.UInt8, np.uint8, 160),
    (QAudioFormat.SampleFormat.Int16, np.int16, 8192),
    (QAudioFormat.SampleFormat.Int32, np.int32, 536870912),
])
def test_unsupported_format_uses_device_preference(player, monkeypatch, sample_format, dtype, expected):
    from app import player as player_module

    preferred = QAudioFormat()
    preferred.setSampleRate(44100)
    preferred.setChannelCount(2)
    preferred.setSampleFormat(sample_format)

    class Device:
        def isNull(self):
            return False

        def isFormatSupported(self, audio_format):
            return audio_format == preferred

        def preferredFormat(self):
            return preferred

    monkeypatch.setattr(player_module.QMediaDevices, "defaultAudioOutput", Device)
    monkeypatch.setattr(player_module, "QAudioSink", lambda *args: FakeSink())
    monkeypatch.setattr(player, "create_sink", StemPlayer.create_sink.__get__(player))
    player.reset(48000, ["vocals"])
    player.append("vocals", 0, np.full((1, 48000), .25, dtype=np.float32))
    player.seek("vocals", .25)
    sink = player.sink
    for step in range(100):
        player.tick()
    audio = np.frombuffer(sink.writes, dtype=dtype).reshape(-1, 2)
    assert len(audio) == 33075
    np.testing.assert_array_equal(audio[:, 0], audio[:, 1])
    np.testing.assert_allclose(audio[100:-100], expected, rtol=.002, atol=1e-4)
    sink.processed = 100000
    player.pause()
    assert player.position == 16800
    player.resume()
    assert player.sink is sink
    sink.processed = 750000
    sink.current_state = QtAudio.State.IdleState
    player.tick()
    assert player.position == 48000 and not player.playing


@pytest.mark.parametrize("output_rate", [24000, 44100, 96000])
def test_resampling_chunks_match_single_conversion(player, output_rate):
    from scipy.signal import resample_poly

    player.reset(48000, ["vocals"])
    audio = np.random.default_rng(12).uniform(-.2, .2, (2, 48000)).astype(np.float32)
    player.append("vocals", 0, audio)
    player.select_stems(["vocals"])
    player.origin = 0
    player.output_format = QAudioFormat()
    player.output_format.setSampleRate(output_rate)
    player.output_format.setChannelCount(1)
    player.output_format.setSampleFormat(QAudioFormat.SampleFormat.Float)
    converted = b"".join(player.output_samples(first, min(first + 733, output_rate)) for first in range(0, output_rate, 733))
    expected = resample_poly(audio.T, output_rate, 48000, axis=0).mean(axis=1)
    np.testing.assert_allclose(np.frombuffer(converted, np.float32), expected, atol=1e-7)


def test_partial_unaligned_writes_preserve_pcm(player, monkeypatch):
    original_write = FakeSink.write
    monkeypatch.setattr(FakeSink, "write", lambda sink, data: original_write(sink, data[:317]))
    player.seek("vocals", 0)
    for step in range(100):
        player.tick()
    assert len(player.sink.writes) == 8000
    np.testing.assert_array_equal(np.frombuffer(player.sink.writes, np.float32), .25)


def test_seek_and_stem_switch_while_playing_never_reset_the_sink(player):
    player.lookahead = StemPlayer.lookahead
    player.seek("vocals", .1)
    sink = player.sink
    player.seek_time(.4)
    player.select_stems(["drums"])
    player.select_stems([])
    player.seek("vocals", .7)
    assert player.sink is sink and not sink.disposed and player.playing
    assert player.position == 700


def test_stale_audio_is_bounded_by_lookahead(player):
    player.lookahead = StemPlayer.lookahead
    player.seek("vocals", 0)
    sink = player.sink
    assert len(sink.writes) == 60 * 8
    player.tick()
    assert len(sink.writes) == 60 * 8
    sink.processed = 30000
    player.tick()
    assert len(sink.writes) == 90 * 8
    sink.processed = 90000
    player.seek("drums", .5)
    stale = len(sink.writes)
    assert stale == 150 * 8 or stale == 90 * 8 + 60 * 8
    assert len(sink.writes) - 90000 // 1000 * 8 <= 60 * 8
    sink.processed = 150000
    player.tick()
    np.testing.assert_array_equal(np.frombuffer(sink.writes, dtype=np.float32)[150 * 2:], -.5)


def test_position_holds_at_target_until_stale_audio_has_played(player):
    player.lookahead = StemPlayer.lookahead
    player.seek("vocals", 0)
    sink = player.sink
    sink.processed = 40000
    base = len(sink.writes) // 8
    player.seek_time(.6)
    assert player.position == 600
    sink.processed = (base - 10) * 1000
    player.tick()
    assert player.position == 600
    sink.processed = (base + 25) * 1000
    player.tick()
    assert player.position == 625
    player.pause()
    assert player.position == 625


def test_retarget_completes_a_partially_written_frame_first(player, monkeypatch):
    original_write = FakeSink.write
    monkeypatch.setattr(FakeSink, "write", lambda sink, data: original_write(sink, data[:317]))
    player.seek("vocals", 0)
    sink = player.sink
    player.tick()
    assert len(sink.writes) % 8 != 0
    player.select_stems(["drums"])
    for step in range(10):
        player.tick()
    audio = np.frombuffer(bytes(sink.writes[:len(sink.writes) // 4 * 4]), dtype=np.float32)
    first_new = np.argmax(audio < 0)
    assert first_new % 2 == 0
    assert np.all(audio[:first_new] == .25) and np.all(audio[first_new:] == -.5)


def test_stem_change_while_paused_defers_the_sink_reset_to_resume(player):
    player.seek("vocals", .2)
    sink = player.sink
    sink.processed = 50000
    player.pause()
    player.select_stems(["drums"])
    assert not sink.disposed and not player.playing and player.position == 250
    player.resume()
    assert sink.disposed and player.playing and player.sink is not sink
    assert np.frombuffer(player.sink.writes, dtype=np.float32)[0] == -.5
    assert player.position == 250
