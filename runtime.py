"""State shared by saywhat's provider, hooks and tools within one Hermes process."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

from .align import normalize
from .dictionary import PersonalDictionary

DEFAULTS: dict[str, Any] = {
    "whisper_model": "base",
    "language": "",
    "device": "auto",
    "threshold": 0.6,
    "auto_correct_after": 3,
    "max_candidates": 4,
    "match_window_seconds": 300,
}


class Runtime:
    def __init__(self, store, get_setting: Callable[[str, Any], Any]):
        self.store = store
        self._get_setting = get_setting
        self.dictionary = PersonalDictionary(store)
        self._recent: deque[dict] = deque(maxlen=8)
        self._lock = threading.Lock()

    def setting(self, key: str) -> Any:
        default = DEFAULTS[key]
        value = self._get_setting(key, default)
        try:
            return type(default)(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    # -- utterances ----------------------------------------------------------
    def remember(self, utterance: dict) -> None:
        with self._lock:
            self._recent.append(utterance)
        self.store.set("last_utterance", utterance)

    def last_utterance(self) -> dict | None:
        with self._lock:
            if self._recent:
                return self._recent[-1]
        return self.store.get("last_utterance", None)

    def match(self, message: str) -> dict | None:
        """The fresh, unresolved utterance this user message was dictated from."""
        msg = normalize(message)
        if not msg:
            return None
        horizon = time.time() - self.setting("match_window_seconds")
        with self._lock:
            candidates = list(reversed(self._recent))
        for utt in candidates:
            text = normalize(utt["text"])
            if utt["ts"] >= horizon and not utt.get("consumed") and text and text in msg:
                return utt
        return None

    def mark(self, utterance: dict, **fields) -> None:
        with self._lock:
            utterance.update(fields)
        if (self.store.get("last_utterance", None) or {}).get("id") == utterance["id"]:
            self.store.set("last_utterance", utterance)

    def pending_clarification(self) -> dict | None:
        """The most recent utterance the agent was asked to clarify and has not resolved."""
        with self._lock:
            for utt in reversed(self._recent):
                if utt.get("consumed") and not utt.get("resolved") and utt.get("spans"):
                    return utt
        return None

    # -- practice ------------------------------------------------------------
    def practice_target(self) -> str | None:
        return self.store.get("practice_target", None)

    def set_practice_target(self, text: str | None) -> None:
        self.store.set("practice_target", text)
