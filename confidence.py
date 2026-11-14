"""Turn word-level recognizer confidence into things the user can choose between.

Input is the recognizer's words with probabilities; output is an *utterance* record:
which spans were uncertain, ranked candidates for each, learned auto-corrections that
were applied, and whole-sentence readings to offer first.
"""

from __future__ import annotations

import time
import uuid

from .align import norm, normalize, opcodes, tokens
from .phonetics import similarity

MIN_MATCH = 0.7          # sound-alike score needed to offer a vocabulary term
WINDOW_BONUS = 0.8       # a two-word window must score this well to widen a span
AUTO_TRUST = 0.9         # never auto-correct a word the recognizer was this sure of
MAX_READINGS = 3         # leaves the 4th clarify slot for "pick word by word"
READINGS_BEAM = MAX_READINGS * 4  # spare room for readings that dedupe to the same text
PERSONAL_HEARD = 0.995   # above every other candidate (capped at 0.99): the user's own term, heard
PERSONAL_NEAR_MISS = 0.9  # confident words this close to a personal term are still questioned
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


def _suspect_words(words: list[dict], threshold: float, personal: tuple[str, ...]) -> list[int]:
    """Indexes worth asking about: words the recognizer was unsure of, plus confident near-misses
    of the user's own terms ("gvector" when they say pgvector), which vocabulary biasing produces."""
    out = []
    for i, w in enumerate(words):
        key = norm(w["w"])
        if not key or w.get("auto"):
            continue
        near_miss = any(key != norm(t) and similarity(w["w"], t) >= PERSONAL_NEAR_MISS for t in personal)
        if w["p"] < threshold or near_miss:
            out.append(i)
    return out


def _alt_replacement(words: list[dict], i: int, j: int, alt_text: str | None) -> str | None:
    """What the biased re-listen pass heard where the first pass heard words[i:j]."""
    if not alt_text:
        return None
    heard, alt = [w["w"] for w in words], tokens(alt_text)
    anchor = {-1: -1, len(heard): len(alt)}  # heard index -> alt index where both passes agree
    for tag, a1, a2, b1, b2 in opcodes(heard, alt):
        if tag == "equal":
            anchor.update({a1 + k: b1 + k for k in range(a2 - a1)})
    if i - 1 not in anchor or j not in anchor:
        return None  # the passes also disagree next to this span; can't attribute the change
    region = " ".join(alt[anchor[i - 1] + 1:anchor[j]])
    return region if region and normalize(region) != normalize(" ".join(heard[i:j])) else None


def _span(words, i, j, vocabulary, corrections, alt_text, max_candidates, threshold, personal=frozenset()):
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
    # The surer the recognizer was, the closer a sound-alike must be to be worth offering.
    bar = min(MIN_MATCH + (1 - MIN_MATCH) * 0.8 * min(p / threshold, 1.0), PERSONAL_NEAR_MISS)
    for term, score in _match_terms(heard, vocabulary)[:max_candidates]:
        if score >= bar:
            offer(term, score, "sounds alike")
    alt = _alt_replacement(words, i, j, alt_text)
    if alt:
        # A partial re-listen ("gvector") supports the term it resembles rather than competing.
        twin = next((c["text"] for c in cands.values() if similarity(alt, c["text"]) >= 0.85), None)
        offer(twin or alt, 0.85, "heard on a vocabulary-biased re-listen")
    # Hearing one of the user's own terms is evidence in itself, whatever the acoustic score.
    mine = norm(heard) in personal
    as_heard = {"text": heard, "score": round(max(p, PERSONAL_HEARD) if mine else p, 3),
                "why": "as heard (in your dictionary)" if mine else "as heard"}
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


def _readings(words: list[dict], spans: list[dict]) -> list[str]:
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

    # Beam over spans: the score is a product of per-span scores, so keeping the best few partial
    # choices at each span finds the same top readings as trying every combination, in linear time.
    beam: list[tuple[float, tuple[str, ...]]] = [(1.0, ())]
    for sp in spans:
        beam = sorted(((score * c["score"], picked + (c["text"],))
                       for score, picked in beam for c in sp["candidates"][:3]),
                      key=lambda sc: -sc[0])[:READINGS_BEAM]
    readings: list[str] = []
    for _, picked in beam:
        text = render({sp["i"]: t for sp, t in zip(spans, picked)})
        if text and normalize(text) not in {normalize(r) for r in readings}:
            readings.append(text.strip())
        if len(readings) == MAX_READINGS:
            break
    return readings


def analyze(words: list[dict], *, vocabulary, corrections: dict, auto: dict[str, str],
            threshold: float, max_candidates: int = 4, alt_text: str | None = None,
            source: str = "", personal=()) -> dict:
    """Build the utterance record for one transcription."""
    raw_text = _join(words)
    fixed, applied = _apply_auto(words, auto)
    spans = []
    for idx in _suspect_words(fixed, threshold, tuple(personal)):
        if spans and idx < spans[-1]["j"]:
            continue
        i, j = _widen(fixed, idx, idx + 1, vocabulary)
        if spans and i < spans[-1]["j"]:
            i = spans[-1]["j"]
        spans.append(_span(fixed, i, j, vocabulary, corrections, alt_text, max_candidates, threshold,
                           frozenset(norm(t) for t in personal)))
    # Nothing to ask when there is nothing to choose between, or the best choice is the user's own
    # term exactly as heard.
    spans = [sp for sp in spans if len(sp["candidates"]) > 1
             and sp["candidates"][0]["why"] != "as heard (in your dictionary)"]
    return {
        "id": uuid.uuid4().hex[:12],
        "ts": time.time(),
        "source": source or "",
        "raw_text": raw_text,
        "text": _join(fixed),
        "words": words,
        "auto": applied,
        "spans": spans,
        "readings": _readings(fixed, spans) if spans else [],
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
