"""The ``saywhat`` speech-to-text provider: local faster-whisper with word confidence."""

from __future__ import annotations

import importlib.util
import threading
from typing import Any, Optional

from agent.transcription_provider import TranscriptionProvider

from .align import normalize
from .confidence import analyze, hotword_terms, words_from_segments
from .runtime import Runtime

NAME = "saywhat"
MODELS = ("tiny", "base", "small", "medium", "large-v3", "distil-large-v3", "turbo")


class Recognizer:
    """faster-whisper models, loaded once per (model, device) and shared across threads."""

    def __init__(self):
        self._models: dict[tuple[str, str], Any] = {}
        self._lock = threading.Lock()

    def _model(self, name: str, device: str):
        key = (name, device)
        with self._lock:
            if key not in self._models:
                from faster_whisper import WhisperModel
                compute = "int8" if device in ("auto", "cpu") else "default"
                self._models[key] = WhisperModel(name, device=device, compute_type=compute)
            return self._models[key]

    def words(self, file_path: str, *, model: str, device: str, language: Optional[str],
              prompt: Optional[str], hotwords: Optional[str]) -> list[dict]:
        whisper = self._model(model, device)
        segments, _info = whisper.transcribe(
            file_path, language=language or None, initial_prompt=prompt or None,
            hotwords=hotwords or None, word_timestamps=True, beam_size=5,
            condition_on_previous_text=False)
        return words_from_segments(list(segments))


def transcribe_utterance(runtime: Runtime, recognizer: Recognizer, file_path: str, *,
                         model: Optional[str] = None, language: Optional[str] = None,
                         prompt: Optional[str] = None, source: str = "") -> dict:
    """Recognize ``file_path``, re-listen around uncertain words, and return the utterance."""
    model = model or runtime.setting("whisper_model")
    language = language or runtime.setting("language") or None
    device = runtime.setting("device")
    # While practising, listen like a stranger would: no personal vocabulary, no learned fixes,
    # so the score reflects how clearly the user is understood, not how well saywhat knows them.
    practising = bool(runtime.practice_target())
    personal = [] if practising else runtime.dictionary.prompt_terms(30)
    words = recognizer.words(file_path, model=model, device=device, language=language,
                             prompt=None if practising else prompt, hotwords=" ".join(personal) or None)
    d = runtime.dictionary

    def build(alt_text=None):
        return analyze(words, vocabulary=d.vocabulary(), corrections=d.corrections(),
                       auto={} if practising else d.auto_corrections(runtime.setting("auto_correct_after")),
                       threshold=runtime.setting("threshold"),
                       max_candidates=runtime.setting("max_candidates"),
                       alt_text=alt_text, source=source, personal=() if practising else personal)

    utterance = build()
    if utterance["spans"] and not practising:
        # Second listen, biased toward the terms the first pass might have meant.
        bias = hotword_terms(utterance) + [t for t in personal if t not in hotword_terms(utterance)]
        alt_words = recognizer.words(file_path, model=model, device=device, language=language,
                                     prompt=", ".join(bias), hotwords=" ".join(bias))
        alt_text = " ".join(w["w"] for w in alt_words)
        if normalize(alt_text) != normalize(utterance["raw_text"]):
            utterance = build(alt_text)
    runtime.remember(utterance)
    return utterance


class SaywhatTranscriptionProvider(TranscriptionProvider):
    def __init__(self, runtime: Runtime, recognizer: Recognizer | None = None):
        self._runtime = runtime
        self._recognizer = recognizer or Recognizer()

    @property
    def name(self) -> str:
        return NAME

    @property
    def display_name(self) -> str:
        return "saywhat (local, asks when unsure)"

    def is_available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": m, "display": m} for m in MODELS]

    def default_model(self) -> Optional[str]:
        return self._runtime.setting("whisper_model")

    def transcribe(self, file_path: str, *, model: Optional[str] = None,
                   language: Optional[str] = None, **extra: Any) -> dict[str, Any]:
        try:
            utterance = transcribe_utterance(
                self._runtime, self._recognizer, file_path, model=model, language=language,
                prompt=extra.get("prompt"), source=str(extra.get("source") or ""))
            return {"success": True, "transcript": utterance["text"], "provider": NAME}
        except Exception as exc:  # the provider contract forbids raising
            return {"success": False, "transcript": "", "provider": NAME, "error": f"saywhat: {exc}"}
