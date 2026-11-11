"""Turn word-level recognizer confidence into things the user can choose between.

Input is the recognizer's words with probabilities; output is an *utterance* record:
which spans were uncertain, ranked candidates for each, learned auto-corrections that
were applied, and whole-sentence readings to offer first.
"""

from __future__ import annotations

import time
import uuid
from itertools import product

from .align import norm, normalize, opcodes, tokens
from .phonetics import similarity

MIN_MATCH = 0.7          # sound-alike score needed to offer a vocabulary term
WINDOW_BONUS = 0.8       # a two-word window must score this well to widen a span
AUTO_TRUST = 0.9         # never auto-correct a word the recognizer was this sure of
MAX_READINGS = 3         # leaves the 4th clarify slot for "pick word by word"
NEIGHBOUR_MAX_P = 0.95   # only a not-quite-sure neighbour may join a span
_FUNCTION_WORDS = frozenset(
    "a an the to of in on at for with and or but is are was were be it this that then than "
    "i you we they he she my your our their me us them do does did can could will would "
    "should please just so if as by from into".split())


def _word(text: str, p: float, start: float = 0.0, end: float = 0.0) -> dict:
    return {"w": text.strip(), "p": round(float(p), 3), "s": round(float(start), 2), "e": round(float(end), 2)}


def words_from_segments(segments) -> list[dict]:
    """faster-whisper segments (with word_timestamps) -> plain word dicts."""
    out = []
    for seg in segments:
        for w in seg.words or []:
            if w.word.strip():
                out.append(_word(w.word, w.probability, w.start, w.end))
    return out


def _join(words: list[dict]) -> str:
    return " ".join(w["w"] for w in words)


def _strip_punct(text: str) -> str:
    return text.strip().strip(",.!?;:\"'")


def _match_terms(heard: str, vocabulary) -> list[tuple[str, float]]:
    scored = []
    for term, aliases in vocabulary:
        score = max([similarity(heard, term)] + [similarity(heard, a) for a in aliases])
        if score >= MIN_MATCH and norm(term) != norm(heard):
            scored.append((term, score))
    return sorted(scored, key=lambda ts: -ts[1])


def _apply_auto(words: list[dict], auto: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """Replace learned mishearings where the recognizer was not already confident."""
    if not auto:
        return words, []
    out, applied, i = [], [], 0
    keyed = sorted(((k.split(), v) for k, v in auto.items()), key=lambda kv: -len(kv[0]))
    while i < len(words):
        for heard_tokens, chosen in keyed:
            n = len(heard_tokens)
            window = words[i:i + n]
            if len(window) == n and [norm(w["w"]) for w in window] == heard_tokens \
                    and min(w["p"] for w in window) < AUTO_TRUST:
                tail = window[-1]["w"][len(_strip_punct(window[-1]["w"])):] if window[-1]["w"] else ""
                out.append(_word(chosen + tail, 1.0, window[0]["s"], window[-1]["e"]) | {"auto": _join(window)})
                applied.append({"heard": _strip_punct(_join(window)), "chosen": chosen})
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out, applied


def _low_runs(words: list[dict], threshold: float) -> list[tuple[int, int]]:
    runs, start = [], None
    for idx, w in enumerate(words + [{"w": "", "p": 1.0}]):
        low = w["p"] < threshold and not w.get("auto") and norm(w["w"])
        if low and start is None:
            start = idx
        elif not low and start is not None:
            runs.append((start, idx))
            start = None
    return runs


def _alt_replacement(words: list[dict], i: int, j: int, alt_text: str | None) -> str | None:
    """What the biased re-listen pass heard where the first pass heard words[i:j]."""
    if not alt_text:
        return None
    heard, alt = [w["w"] for w in words], tokens(alt_text)
    anchor = {}  # heard index -> alt index, for words both passes agree on
    for tag, a1, a2, b1, b2 in opcodes(heard, alt):
        if tag == "equal":
            anchor.update({a1 + k: b1 + k for k in range(a2 - a1)})
    before = max((a for a in anchor if a < i), default=-1)
    after = min((a for a in anchor if a >= j), default=len(heard))
    if (before + 1, after) != (i, j) and not all(a in anchor for a in range(i, j)):
        return None  # the passes disagree on more than this span
    lo = anchor.get(before, -1) + 1
    hi = anchor.get(after, len(alt))
    region = " ".join(alt[lo:hi])
    return region if region and normalize(region) != normalize(" ".join(heard[before + 1:after])) else None


def _span(words, i, j, vocabulary, corrections, alt_text, max_candidates):
    heard = _strip_punct(_join(words[i:j]))
    p = min(w["p"] for w in words[i:j])
    cands: dict[str, dict] = {}

    def offer(text, score, why):
        text = _strip_punct(text)
        key = norm(text)
        if not key:
            return
        if key in cands:  # two independent signals agree: corroborate
            prev = cands[key]
            cands[key] = {"text": prev["text"], "score": round(min(max(prev["score"], score) + 0.1, 0.99), 3),
                          "why": f'{prev["why"]} + {why}'}
        else:
            cands[key] = {"text": text, "score": round(min(score, 0.99), 3), "why": why}

    for chosen, count in corrections.get(normalize(heard), {}).items():
        offer(chosen, 0.8 + 0.05 * count, f"you picked this {count}x before")
    for term, score in _match_terms(heard, vocabulary)[:max_candidates]:
        offer(term, score, "sounds alike")
    alt = _alt_replacement(words, i, j, alt_text)
    if alt:
        offer(alt, 0.85, "heard on a vocabulary-biased re-listen")
    as_heard = {"text": heard, "score": round(p, 3), "why": "as heard"}
    cands.pop(norm(heard), None)
    ranked = sorted(cands.values(), key=lambda c: -c["score"])[:max(max_candidates - 1, 1)]
    ranked = sorted(ranked + [as_heard], key=lambda c: -c["score"])
    return {"i": i, "j": j, "heard": heard, "p": round(p, 3), "candidates": ranked}


def _widen(words, i, j, vocabulary):
    """Grow a span by one neighbour when the pair sounds like one term ("cooper netties")."""
    best_score = max((s for _, s in _match_terms(_join(words[i:j]), vocabulary)[:1]), default=0.0)
    for a, b in ((i - 1, j), (i, j + 1)):
        if a < 0 or b > len(words):
            continue
        neighbour = words[a] if a < i else words[b - 1]
        if norm(neighbour["w"]) in _FUNCTION_WORDS or neighbour["p"] >= NEIGHBOUR_MAX_P:
            continue
        top = _match_terms(_strip_punct(_join(words[a:b])), vocabulary)[:1]
        if top and top[0][1] >= max(WINDOW_BONUS, best_score + 0.05):
            return a, b
    return i, j


def _readings(words: list[dict], spans: list[dict], alt_text: str | None) -> list[str]:
    """Whole-sentence options, most plausible first."""
    def render(choice: dict[int, str]) -> str:
        parts, idx = [], 0
        for sp in spans:
            parts.extend(w["w"] for w in words[idx:sp["i"]])
            tail = words[sp["j"] - 1]["w"][len(_strip_punct(words[sp["j"] - 1]["w"])):]
            parts.append(choice[sp["i"]] + tail)
            idx = sp["j"]
        parts.extend(w["w"] for w in words[idx:])
        return " ".join(parts)

    options = [sp["candidates"][:3] for sp in spans]
    combos = []
    for combo in product(*options):
        score = 1.0
        for c in combo:
            score *= c["score"]
        combos.append((score, render({sp["i"]: c["text"] for sp, c in zip(spans, combo)})))
    combos.sort(key=lambda sc: -sc[0])
    readings: list[str] = []
    for text in [t for _, t in combos[:1]] + ([alt_text] if alt_text else []) + [t for _, t in combos[1:]]:
        if text and normalize(text) not in {normalize(r) for r in readings}:
            readings.append(text.strip())
        if len(readings) == MAX_READINGS:
            break
    return readings


def analyze(words: list[dict], *, vocabulary, corrections: dict, auto: dict[str, str],
            threshold: float, max_candidates: int = 4, alt_text: str | None = None,
            source: str = "") -> dict:
    """Build the utterance record for one transcription."""
    raw_text = _join(words)
    fixed, applied = _apply_auto(words, auto)
    spans = []
    for i, j in _low_runs(fixed, threshold):
        if spans and i < spans[-1]["j"]:
            continue
        i, j = _widen(fixed, i, j, vocabulary)
        if spans and i < spans[-1]["j"]:
            i = spans[-1]["j"]
        spans.append(_span(fixed, i, j, vocabulary, corrections, alt_text, max_candidates))
    # A span whose only candidate is what was heard has nothing to choose between.
    spans = [sp for sp in spans if len(sp["candidates"]) > 1]
    return {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "source": source or "",
        "raw_text": raw_text,
        "text": _join(fixed),
        "words": words,
        "auto": applied,
        "spans": spans,
        "readings": _readings(fixed, spans, alt_text) if spans else [],
        "resolved": False,
    }


def hotword_terms(utterance: dict, limit: int = 12) -> list[str]:
    """Vocabulary candidates worth biasing a re-listen pass toward."""
    terms = []
    for sp in utterance["spans"]:
        for c in sp["candidates"]:
            if c["why"] != "as heard" and c["text"] not in terms:
                terms.append(c["text"])
    return terms[:limit]
