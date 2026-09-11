from __future__ import annotations

import tempfile
from math import gcd

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices, QtAudio
from scipy.signal import resample_poly


MIXTURE_KEY = "\x00mixture"


class StemAudio:
    """Disk-backed finalized PCM; all access is on the GUI thread."""

    def __init__(self):
        self.channels = None
        self.file = tempfile.TemporaryFile()
        self._frames = 0

    @property
    def frames(self):
        return self._frames

    def append(self, start, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or not np.isfinite(audio).all() or audio.shape[0] < 1:
            raise ValueError("Expected finite audio [channels, samples]")
        if start != self.frames or self.channels not in (None, audio.shape[0]):
            raise ValueError("Playback chunks must be contiguous with matching channels")
        self.channels = audio.shape[0]
        self.file.seek(0, 2)
        for offset in range(0, audio.shape[-1], 65536):
            self.file.write(np.ascontiguousarray(audio[:, offset:offset + 65536].T).tobytes())
        self._frames += audio.shape[-1]

    def read(self, start, end):
        self.file.seek(start * self.channels * 4)
        data = self.file.read((min(end, self.frames) - start) * self.channels * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(-1, self.channels)

    def close(self):
        self.file.close()


class StemPlayer(QObject):
    changed = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sink = None
        self.timer = QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.tick)
        self.buffers = {}
        self.rate = 1
        self.selected = ()
        self.position = 0
        self.playing = False

    def reset(self, rate=1, stems=()):
        self.release()
        self.rate = rate
        for buffer in self.buffers.values():
            buffer.close()
        self.buffers = {stem: StemAudio() for stem in stems}
        self.selected = ()
        self.position = 0
        self.changed.emit()

    def append(self, stem, start, audio):
        self.buffers[stem].append(start, audio)
        self.changed.emit()

    @property
    def stem(self):
        sources = self.playback_sources
        return sources[0] if len(sources) == 1 else None

    @property
    def playback_sources(self):
        components = set(self.buffers) - {MIXTURE_KEY}
        if MIXTURE_KEY in self.buffers and components and set(self.selected) == components:
            return (MIXTURE_KEY,)
        return self.selected

    @property
    def available(self):
        if not self.selected:
            reference = self.buffers.get(MIXTURE_KEY) or next(iter(self.buffers.values()), None)
            return reference.frames if reference is not None else 0
        return min((self.buffers[key].frames for key in self.playback_sources), default=0)

    @property
    def channels(self):
        if not self.selected:
            reference = self.buffers.get(MIXTURE_KEY) or next(iter(self.buffers.values()), None)
            return reference.channels if reference is not None else None
        return self.buffers[self.playback_sources[0]].channels

    def samples(self, stem, start, end):
        return self.buffers[stem].read(start, end)

    def select_stems(self, stems):
        selected = tuple(key for key in self.buffers if key in stems)
        if selected == self.selected:
            return
        was_playing = self.playing
        if was_playing:
            self.update_position()
        self.release()
        self.selected = selected
        if was_playing:
            self.resume()
        self.changed.emit()

    def release(self):
        self.timer.stop()
        self.playing = False
        if self.sink is not None:
            self.sink.reset()
            self.sink.deleteLater()
            self.sink = None

    def create_sink(self, channels):
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            raise ValueError("No audio output device is available")
        audio_format = QAudioFormat()
        audio_format.setSampleRate(self.rate)
        audio_format.setChannelCount(channels)
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Float)
        if not device.isFormatSupported(audio_format):
            audio_format = device.preferredFormat()
            if not audio_format.isValid() or not device.isFormatSupported(audio_format):
                raise ValueError("Audio output has no supported PCM format")
        self.output_format = audio_format
        return QAudioSink(device, audio_format, self)

    def output_samples(self, first, end):
        output_rate = self.output_format.sampleRate()
        divisor = gcd(self.rate, output_rate)
        up, down = output_rate // divisor, self.rate // divisor
        margin = 10 * max(up, down) // up + 2
        start = max(0, (first * down // up - margin) // down * down)
        stop = min(self.available - self.origin, (end * down + up - 1) // up + margin)
        if up == down:
            start, stop = first, end
        mixed = np.zeros((stop - start, self.channels), dtype=np.float32)
        for stem in self.playback_sources:
            mixed += self.samples(stem, self.origin + start, self.origin + stop)
        if up != down:
            mixed = resample_poly(mixed, up, down, axis=0)
            offset = first - start * up // down
            mixed = mixed[offset:offset + end - first]
        channels = self.output_format.channelCount()
        if channels == 1 and self.channels > 1:
            mixed = mixed.mean(axis=1, keepdims=True)
        elif channels != self.channels:
            mixed = mixed[:, np.arange(channels) % self.channels]
        mixed = np.clip(mixed, -1, 1)
        sample_format = self.output_format.sampleFormat()
        if sample_format == QAudioFormat.SampleFormat.UInt8:
            mixed = np.clip(np.rint((mixed + 1) * 128), 0, 255).astype(np.uint8)
        elif sample_format == QAudioFormat.SampleFormat.Int16:
            mixed = np.clip(np.rint(mixed.astype(np.float64) * 32768), -32768, 32767).astype(np.int16)
        elif sample_format == QAudioFormat.SampleFormat.Int32:
            mixed = np.clip(np.rint(mixed.astype(np.float64) * 2147483648), -2147483648, 2147483647).astype(np.int32)
        return np.ascontiguousarray(mixed).tobytes()

    def seek(self, stem, seconds):
        self.select_stems([stem])
        self.seek_time(seconds)

    def seek_time(self, seconds):
        self.release()
        self.position = min(self.available, max(0, int(seconds * self.rate)))
        self.resume()

    def resume(self):
        if self.playing or self.channels is None:
            return
        if self.position >= self.available:
            self.changed.emit()
            return
        try:
            if self.sink is not None:
                self.sink.resume()
            else:
                self.output_format = QAudioFormat()
                self.output_format.setSampleRate(self.rate)
                self.output_format.setChannelCount(self.channels)
                self.output_format.setSampleFormat(QAudioFormat.SampleFormat.Float)
                self.sink = self.create_sink(self.channels)
                self.sink.setBufferSize(max(1, self.output_format.sampleRate() // 10) * self.output_format.bytesPerFrame())
                self.origin = self.position
                self.queued_bytes = 0
                self.pending = b""
                self.output = self.sink.start()
                if self.output is None or self.sink.error() != QtAudio.Error.NoError:
                    raise RuntimeError("Unable to start the audio output device")
            self.playing = True
            self.timer.start()
            self.tick()
        except Exception as error:
            self.release()
            self.failed.emit(str(error))
        self.changed.emit()

    def pause(self):
        if self.playing:
            self.update_position()
            self.sink.suspend()
            self.timer.stop()
            self.playing = False
            self.changed.emit()

    def update_position(self):
        self.position = min(
            self.available,
            self.origin + self.queued_bytes // self.output_format.bytesPerFrame() * self.rate // self.output_format.sampleRate(),
            self.origin + int(self.sink.processedUSecs() * self.rate / 1_000_000),
        )

    def tick(self):
        if not self.playing:
            return
        if self.sink.error() not in (QtAudio.Error.NoError, QtAudio.Error.UnderrunError):
            self.release()
            self.failed.emit("Audio output failed or was disconnected")
            self.changed.emit()
            return
        self.update_position()
        frame_bytes = self.output_format.bytesPerFrame()
        output_rate = self.output_format.sampleRate()
        available_frames = ((self.available - self.origin) * output_rate + self.rate - 1) // self.rate
        available_bytes = available_frames * frame_bytes
        if self.queued_bytes == available_bytes and self.sink.state() == QtAudio.State.IdleState:
            self.position = self.available
            self.release()
            self.changed.emit()
            return
        count = min(self.sink.bytesFree(), available_bytes - self.queued_bytes)
        if not self.pending:
            count -= count % frame_bytes
        if count > 0:
            try:
                if not self.pending:
                    first = self.queued_bytes // frame_bytes
                    end = first + count // frame_bytes
                    self.pending = self.output_samples(first, end)
            except OSError as error:
                self.release()
                self.failed.emit(f"Unable to read playback audio: {error}")
                self.changed.emit()
                return
            written = self.output.write(self.pending[:count])
            if written < 0:
                self.release()
                self.failed.emit("Unable to write to the audio output")
            else:
                self.queued_bytes += written
                self.pending = self.pending[written:]
        self.changed.emit()
