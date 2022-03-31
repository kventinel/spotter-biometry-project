import wave
import pyaudio as pa


class AudioRecoder:
    def __init__(self, chunk=1024, format=pa.paInt32, channels=1, rate=44100):
        self._chunk = chunk
        self._format = format
        self._channels = channels
        self._rate = rate

    def record(self, output_path, rec_seconds=None):
        p = pa.PyAudio()
        stream = p.open(
            format=self._format, channels=self._channels, rate=self._rate,
            input=True, frames_per_buffer=self._chunk
        )

        frames = []

        if rec_seconds is None:
            pass
        else:
            for _ in range(0, int(self._rate / self._chunk * rec_seconds)):
                data = stream.read(self._chunk)
                frames.append(data)

        stream.stop_stream()

        waveform = wave.open(output_path, 'wb')
        waveform.setnchannels(self._channels)
        waveform.setsampwidth(p.get_sample_size(self._format))
        waveform.setframerate(self._rate)
        waveform.writeframesraw(b''.join(frames))
        waveform.close()

        stream.close()
        p.terminate()


if __name__ == "__main__":
    import os

    n_seconds = 10

    audio_recoder = AudioRecoder()

    print("----Start recoding----")
    print(f"The recording will last {n_seconds} seconds")
    audio_recoder.record("tmp/test.wav", n_seconds)
    print("----Stop recoding----")
    print(f"The result is available: {os.path.join(os.getcwd(), 'tmp/test.wav')}")
