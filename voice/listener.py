"""Microphone listener: Vosk spots stop words in partial results, an energy VAD cuts utterances and
faster-whisper transcribes them. Both read one sounddevice stream, 16 kHz int16 mono.

Vosk's small model is restricted to the stop words by a grammar (it is poor at free speech);
Whisper does the free speech. Models download into data/models/ on first use.
"""

from __future__ import annotations

import collections
import json
import queue
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

RATE = 16000
BLOCK = 1600  # samples per block, 100 ms
MODELS = Path(__file__).resolve().parents[1] / "data" / "models"
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_NAME = "vosk-model-small-en-us-0.15"
STOP_WORDS = ["stop", "wait", "no no", "halt"]
_PREROLL_BLOCKS = 3  # kept from before the speech started, so its onset isn't clipped
_MIN_SPEECH_S = 0.3
_MAX_SPEECH_S = 15.0
# Biases Whisper toward the words people say to the feeder.
_PROMPT = "Talking to a feeding robot: a donut hole, a grape, some yogurt, please. Yes. No. Stop."


def vosk_model_dir() -> Path:
    """The small English Vosk model, downloaded and unzipped into data/models/ on first use."""
    path = MODELS / VOSK_NAME
    if not path.is_dir():
        MODELS.mkdir(parents=True, exist_ok=True)
        archive = MODELS / f"{VOSK_NAME}.zip"
        print(f"downloading {VOSK_URL} (41 MB)", file=sys.stderr)
        urllib.request.urlretrieve(VOSK_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(MODELS)
        archive.unlink()
    return path


def whisper_model(name: str):
    """A CPU faster-whisper model from data/models/whisper/ (downloaded on first use); tiny.en
    when name fails to load."""
    from faster_whisper import WhisperModel

    error = None
    for model in dict.fromkeys((name, "tiny.en")):
        try:
            return WhisperModel(model, device="cpu", compute_type="int8",
                                download_root=str(MODELS / "whisper"))
        except Exception as e:  # download or load failure: try the fallback
            print(f"whisper {model} failed to load: {e}", file=sys.stderr)
            error = e
    raise error


def _has_word(text: str, word: str) -> bool:
    return f" {word} " in f" {text} "


class Listener(threading.Thread):
    """on_stop(word) fires on a stop word in a partial result, once per utterance;
    on_utterance(text) gets each transcribed utterance; on_partial(text), when given, each new
    Vosk partial. cfg is brain.yaml's voice section (or the whole brain.yaml). `last_text` is the
    last utterance, `level` the last block's RMS (0-1), `error` why the thread died."""

    def __init__(self, cfg: dict, on_utterance, on_stop, on_partial=None):
        super().__init__(name="listener", daemon=True)
        cfg = cfg.get("voice", cfg)
        self.device = cfg.get("device")
        self.stop_words = [w.lower() for w in cfg.get("stop_words") or STOP_WORDS]
        self.whisper_name = cfg.get("whisper_model", "base.en")
        self.vad_threshold = float(cfg.get("vad_threshold", 0.02))
        self.silence_s = cfg.get("silence_ms", 700) / 1000
        self.on_utterance, self.on_stop, self.on_partial = on_utterance, on_stop, on_partial
        self.last_text = ""
        self.level = 0.0
        self.error = ""
        self._halt = threading.Event()
        self._rec = None
        self._whisper = None
        self._fired = False
        self._partial = ""
        self._speech: list | None = None  # sample arrays while someone speaks
        self._quiet_s = 0.0
        self._preroll: collections.deque = collections.deque(maxlen=_PREROLL_BLOCKS)
        self._utterances: queue.Queue = queue.Queue()

    def load(self) -> None:
        """Load (and on first use download) both models and start the transcriber."""
        import vosk

        vosk.SetLogLevel(-1)
        grammar = json.dumps([*self.stop_words, "[unk]"])
        self._rec = vosk.KaldiRecognizer(vosk.Model(str(vosk_model_dir())), RATE, grammar)
        self._whisper = whisper_model(self.whisper_name)
        threading.Thread(target=self._transcribe, name="transcriber", daemon=True).start()

    def run(self) -> None:
        import sounddevice as sd

        try:
            if self._rec is None:
                self.load()
            blocks: queue.Queue = queue.Queue()
            with sd.RawInputStream(samplerate=RATE, blocksize=BLOCK, dtype="int16", channels=1,
                                   device=self.device,
                                   callback=lambda data, *_: blocks.put(bytes(data))):
                while not self._halt.is_set():
                    try:
                        self.feed(blocks.get(timeout=0.5))
                    except queue.Empty:
                        continue
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            raise

    def stop(self) -> None:
        self._halt.set()

    def feed(self, data: bytes) -> None:
        """Process one block of 16 kHz int16 mono audio (run() feeds the microphone)."""
        import numpy as np

        samples = np.frombuffer(data, np.int16).astype(np.float32) / 32768
        self.level = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        self._spot(data)
        self._segment(samples)

    def drain(self) -> None:
        """End the utterance in progress and wait until every utterance is transcribed."""
        if self._speech is not None:
            self._end_speech()
        self._utterances.join()

    def _spot(self, data: bytes) -> None:
        if self._rec.AcceptWaveform(data):
            text, final = json.loads(self._rec.Result()).get("text", ""), True
        else:
            text, final = json.loads(self._rec.PartialResult()).get("partial", ""), False
        if text and text != self._partial and self.on_partial:
            self.on_partial(text)
        self._partial = "" if final else text
        word = next((w for w in self.stop_words if _has_word(text, w)), None)
        if word and not self._fired:
            self._fired = True
            self.on_stop(word)
        if final and self._speech is None:  # else the VAD's utterance end re-arms it
            self._fired = False

    def _segment(self, samples) -> None:
        loud = self.level > self.vad_threshold
        if self._speech is None:
            self._preroll.append(samples)
            if loud:
                self._speech, self._quiet_s = list(self._preroll), 0.0
                self._preroll.clear()
            return
        self._speech.append(samples)
        self._quiet_s = 0.0 if loud else self._quiet_s + samples.size / RATE
        length = sum(s.size for s in self._speech) / RATE
        if self._quiet_s >= self.silence_s or length >= _MAX_SPEECH_S:
            self._end_speech()

    def _end_speech(self) -> None:
        import numpy as np

        audio = np.concatenate(self._speech)
        if audio.size / RATE - self._quiet_s >= _MIN_SPEECH_S:
            self._utterances.put(audio)
        # Re-arm the stop word for the next utterance; Reset drops Vosk's lingering partial, which
        # would fire it again.
        self._rec.Reset()
        self._speech, self._quiet_s, self._fired, self._partial = None, 0.0, False, ""

    def _transcribe(self) -> None:
        while True:
            audio = self._utterances.get()
            try:
                segments, _ = self._whisper.transcribe(
                    audio, beam_size=1, language="en", condition_on_previous_text=False,
                    initial_prompt=_PROMPT)
                text = " ".join(s.text.strip() for s in segments).strip()
                if text:
                    self.last_text = text
                    self.on_utterance(text)
            except Exception as e:  # one bad utterance must not end the transcriber
                print(f"transcription failed: {e}", file=sys.stderr)
            finally:
                self._utterances.task_done()
