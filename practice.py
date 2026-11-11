"""Pronunciation practice: compare what the user meant with what the recognizer heard.

The recognizer stands in for "a listener who doesn't know what you meant to say". A word
it misheard or barely caught is a word another listener (human or AI) may miss too.
"""

from __future__ import annotations

import time
from difflib import SequenceMatcher

from .align import norm, normalize, opcodes, tokens
from .phonetics import is_acronym, letters, skeleton, similarity

_HISTORY_KEY = "practice_history"
_TROUBLE_KEY = "trouble_words"
_MAX_ATTEMPTS = 50
_MAX_PHRASES = 100


def _sound_diff(target: str, heard: str) -> str:
    """Name the consonant sounds that went missing or appeared, e.g. "the 'pg' sound was lost"."""
    t, h = skeleton(target), skeleton(heard)
    lost, extra = [], []
    for tag, a1, a2, b1, b2 in SequenceMatcher(None, t, h).get_opcodes():
        if tag in ("delete", "replace"):
            lost.append(t[a1:a2])
        if tag in ("insert", "replace"):
            extra.append(h[b1:b2])
    parts = []
    if lost:
        parts.append("the " + ", ".join(f"'{x}'" for x in lost) + " sound didn't come through")
    if extra:
        parts.append("it caught " + ", ".join(f"'{x}'" for x in extra) + " instead")
    return "; ".join(parts)


def tip(target: str, heard: str | None, p: float) -> str:
    if heard is None:
        return "It was not heard at all: give it a clear start and don't swallow it into the next word."
    if is_acronym(target):
        spaced = ". ".join(letters(target).upper())
        return f"Spell it out letter by letter with tiny pauses: {spaced}."
    if norm(heard) == norm(target):
        return "Heard correctly but with low confidence: say it a little slower and fully voice the ending."
    closeness = similarity(heard, target)
    detail = _sound_diff(target, heard)
    if closeness >= 0.7 and not detail:
        return (f"Close: it heard '{heard}'. Every consonant came through; the vowels or stress blurred it, "
                "so give each syllable its own beat.")
    if closeness >= 0.7:
        return f"Close: it heard '{heard}'. {detail[0].upper() + detail[1:]}."
    return (f"It heard a different word ('{heard}'). Slow down, stress the first syllable"
            + (f"; {detail}." if detail else "."))


def score_attempt(target: str, words: list[dict], threshold: float) -> dict:
    """Word-by-word verdicts for one attempt at saying ``target``."""
    want, got = tokens(target), [w["w"] for w in words]
    verdicts = []
    for tag, a1, a2, b1, b2 in opcodes(want, got):
        if tag == "equal":
            for k in range(a2 - a1):
                w = words[b1 + k]
                ok = w["p"] >= threshold
                verdicts.append({"word": want[a1 + k], "verdict": "clear" if ok else "unclear",
                                 "heard": w["w"].strip(",.!?"), "p": w["p"],
                                 **({} if ok else {"tip": tip(want[a1 + k], w["w"], w["p"])})})
        elif tag in ("replace", "delete"):
            heard = " ".join(got[b1:b2]).strip(",.!?") or None
            p = min((w["p"] for w in words[b1:b2]), default=0.0)
            for k in range(a1, a2):
                verdicts.append({"word": want[k], "verdict": "misheard" if heard else "missed",
                                 "heard": heard, "p": p, "tip": tip(want[k], heard, p)})
    credit = {"clear": 1.0, "misheard": 0.0, "missed": 0.0}
    total = sum(credit.get(v["verdict"], min(v["p"] / threshold, 1.0) * 0.8) for v in verdicts)
    score = round(100 * total / max(len(verdicts), 1))
    return {"target": target, "heard": " ".join(got), "score": score, "words": verdicts,
            "focus": [v["word"] for v in verdicts if v["verdict"] != "clear"]}


class PracticeLog:
    """Attempt history per phrase and a running tally of trouble words: the speaking profile."""

    def __init__(self, store):
        self._store = store

    def record(self, attempt: dict) -> dict:
        history = self._store.get(_HISTORY_KEY, None) or {}
        key = normalize(attempt["target"])
        runs = history.setdefault(key, {"target": attempt["target"], "attempts": []})
        previous = runs["attempts"][-1]["score"] if runs["attempts"] else None
        runs["attempts"] = (runs["attempts"] + [{"ts": int(time.time()), "score": attempt["score"],
                                                 "focus": attempt["focus"]}])[-_MAX_ATTEMPTS:]
        if len(history) > _MAX_PHRASES:
            oldest = min(history, key=lambda k: history[k]["attempts"][-1]["ts"])
            history.pop(oldest)
        self._store.set(_HISTORY_KEY, history)

        trouble = self._store.get(_TROUBLE_KEY, None) or {}
        for word in attempt["focus"]:
            trouble[word] = trouble.get(word, 0) + 1
        self._store.set(_TROUBLE_KEY, trouble)
        return {"attempt": len(runs["attempts"]), "previous_score": previous,
                "change": None if previous is None else attempt["score"] - previous,
                "best": max(a["score"] for a in runs["attempts"])}

    def progress(self) -> dict:
        history = self._store.get(_HISTORY_KEY, None) or {}
        phrases = []
        for runs in history.values():
            scores = [a["score"] for a in runs["attempts"]]
            phrases.append({"phrase": runs["target"], "attempts": len(scores), "first": scores[0],
                            "latest": scores[-1], "best": max(scores), "trend": scores[-10:]})
        phrases.sort(key=lambda p: -p["attempts"])
        trouble = self._store.get(_TROUBLE_KEY, None) or {}
        top = sorted(trouble.items(), key=lambda kv: -kv[1])[:15]
        return {"phrases": phrases, "trouble_words": [{"word": w, "times": n} for w, n in top]}
