"""Sound-alike matching between what the recognizer heard and vocabulary terms.

Whisper mishears technical terms in two typical ways: it spells them the way they
sound ("cubectal" for kubectl) or it hears an acronym's letter names as a word
("x" for EKS). ``skeleton`` reduces text to its consonant shape and ``spoken_forms``
expands acronyms into letter names, so both errors land close to the right term.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_LETTER_NAMES = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef", "g": "gee",
    "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el", "m": "em", "n": "en",
    "o": "oh", "p": "pee", "q": "cue", "r": "ar", "s": "ess", "t": "tee", "u": "you",
    "v": "vee", "w": "doubleyou", "x": "ex", "y": "why", "z": "zee",
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_DIGRAPHS = (("ph", "f"), ("ck", "k"), ("dg", "j"), ("wh", "w"), ("sch", "sk"), ("th", "t"),
             ("qu", "kw"), ("gh", "g"), ("x", "ks"), ("q", "k"), ("z", "s"), ("v", "f"))
_VOWELS = set("aeiouy")


def letters(text: str) -> str:
    """Lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def skeleton(text: str) -> str:
    """Consonant shape of ``text``, word by word: digraphs folded, soft/hard c resolved,
    vowels after each word's first letter dropped, repeated letters collapsed."""
    return "".join(_word_skeleton(w) for w in text.split())


def _word_skeleton(word: str) -> str:
    s = letters(word)
    if not s:
        return ""
    s = re.sub(r"c(?=[eiy])", "s", s).replace("c", "k")
    for src, dst in _DIGRAPHS:
        s = s.replace(src, dst)
    out = s[0] + "".join(ch for ch in s[1:] if ch not in _VOWELS)
    return re.sub(r"(.)\1+", r"\1", out)


def is_acronym(term: str) -> bool:
    """Short all-caps tokens (EKS, gRPC's RPC, S3) are spoken letter by letter."""
    core = re.sub(r"[^A-Za-z0-9]", "", term)
    return 1 < len(core) <= 6 and core.upper() == core and any(c.isalpha() for c in core)


def spoken_forms(term: str) -> list[str]:
    """Ways ``term`` is likely to sound: as written, and letter by letter for acronyms.
    Short or vowel-less acronyms (EKS, XSS) are only ever spelled out, so they lose their written form."""
    if not is_acronym(term):
        # Digits are said as words: "k8s" sounds like "k eight s", never like "ks".
        return [re.sub(r"[0-9]", lambda m: " " + _LETTER_NAMES[m.group()] + " ", term)]
    spelled = " ".join(_LETTER_NAMES.get(c, c) for c in letters(term))
    pronounceable = len(letters(term)) >= 4 and set(letters(term)) & set("aeiou")  # JSON, YAML, CUDA
    return [term, spelled] if pronounceable else [spelled]


def similarity(heard: str, term: str) -> float:
    """0..1 sound-alike score between a heard span and a vocabulary term."""
    h_letters = letters(heard)
    if not h_letters:
        return 0.0
    # A lone letter in a transcript ("x") stands for its spoken name ("ex").
    heard_forms = [heard] + ([" ".join(_LETTER_NAMES.get(c, c) for c in h_letters)] if len(h_letters) <= 2 else [])
    best = 0.0
    for form in spoken_forms(term):
        t_sk = skeleton(form)
        for h_form in heard_forms:
            h_sk = skeleton(h_form)
            if not h_sk or not t_sk:
                continue
            score = SequenceMatcher(None, h_sk, t_sk).ratio()
            if min(len(h_sk), len(t_sk)) <= 2 and score < 1.0:
                score *= 0.9  # two-consonant shapes collide easily; demand more
            best = max(best, score, 0.9 * SequenceMatcher(None, letters(h_form), letters(form)).ratio())
    t_letters = letters(term)
    if len(h_letters) >= 4 and h_letters in t_letters and h_letters != t_letters:
        # Recognizer dropped part of a compound ("vector" for pgvector).
        best = max(best, 0.7 + 0.3 * len(h_letters) / len(t_letters))
    return round(min(best, 1.0), 3)
