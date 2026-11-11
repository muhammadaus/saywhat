"""Personal dictionary: the user's vocabulary and their recurring mishearings.

Persisted through any ``get(key, default)`` / ``set(key, value)`` store; inside
Hermes that is the plugin's profile-scoped ``ctx.state``.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from .align import norm, normalize

_KEY = "dictionary"
_LEXICON = Path(__file__).with_name("lexicon.txt")


@lru_cache(maxsize=1)
def seed_lexicon() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(term, aliases) pairs from the bundled lexicon."""
    entries = []
    for line in _LEXICON.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term, *aliases = [part.strip() for part in line.split("|")]
        entries.append((term, tuple(a for a in aliases if a)))
    return tuple(entries)


class PersonalDictionary:
    def __init__(self, store):
        self._store = store

    def _load(self) -> dict:
        data = self._store.get(_KEY, None) or {}
        data.setdefault("terms", {})
        data.setdefault("corrections", {})
        return data

    def _save(self, data: dict) -> None:
        self._store.set(_KEY, data)

    # -- terms -------------------------------------------------------------
    def terms(self) -> dict[str, dict]:
        return self._load()["terms"]

    def add_term(self, term: str, source: str = "manual") -> bool:
        term = term.strip()
        if not norm(term):
            return False
        data = self._load()
        entry = data["terms"].setdefault(term, {"count": 0, "source": source, "added": int(time.time())})
        entry["count"] += 1
        self._save(data)
        return entry["count"] == 1

    def remove_term(self, term: str) -> bool:
        data = self._load()
        key = next((t for t in data["terms"] if norm(t) == norm(term)), None)
        removed = data["terms"].pop(key, None) is not None if key else False
        for heard in list(data["corrections"]):
            data["corrections"][heard].pop(term, None)
            if not data["corrections"][heard]:
                del data["corrections"][heard]
        self._save(data)
        return removed

    # -- corrections -------------------------------------------------------
    def record_correction(self, heard: str, chosen: str, weight: int = 1) -> None:
        """The recognizer produced ``heard`` where the user meant ``chosen``."""
        key = normalize(heard)
        if not key or not chosen.strip() or key == normalize(chosen):
            return
        data = self._load()
        counts = data["corrections"].setdefault(key, {})
        counts[chosen.strip()] = counts.get(chosen.strip(), 0) + weight
        entry = data["terms"].setdefault(chosen.strip(), {"count": 0, "source": "learned", "added": int(time.time())})
        entry["count"] += weight
        self._save(data)

    def corrections(self) -> dict[str, dict[str, int]]:
        return self._load()["corrections"]

    def auto_corrections(self, min_count: int) -> dict[str, str]:
        """heard -> chosen for mishearings confirmed ``min_count`` times with a clear winner."""
        out = {}
        for heard, counts in self.corrections().items():
            chosen, best = max(counts.items(), key=lambda kv: kv[1])
            if best >= min_count and best * 3 >= sum(counts.values()) * 2:
                out[heard] = chosen
        return out

    # -- recognizer biasing -------------------------------------------------
    def vocabulary(self, include_seed: bool = True) -> list[tuple[str, tuple[str, ...]]]:
        """Personal terms first, then the seed lexicon, as (term, aliases)."""
        personal = sorted(self.terms().items(), key=lambda kv: -kv[1]["count"])
        seen, out = set(), []
        for term, _ in personal:
            out.append((term, ()))
            seen.add(norm(term))
        if include_seed:
            out.extend(e for e in seed_lexicon() if norm(e[0]) not in seen)
        return out

    def prompt_terms(self, limit: int = 40) -> list[str]:
        """Most-used personal terms, for the Whisper prompt / hotwords."""
        ranked = sorted(self.terms().items(), key=lambda kv: -kv[1]["count"])
        return [term for term, _ in ranked[:limit]]
