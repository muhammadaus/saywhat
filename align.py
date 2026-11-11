"""Token-level alignment between two renderings of the same utterance."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'./+#-]*")


def tokens(text: str) -> list[str]:
    """Words of ``text`` with surrounding punctuation stripped, original casing kept."""
    return [t.rstrip(".") or t for t in _TOKEN.findall(text or "")]


def norm(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def normalize(text: str) -> str:
    """Case/punctuation-insensitive form used to match transcripts against messages."""
    return " ".join(n for n in (norm(t) for t in tokens(text)) if n)


def opcodes(a: list[str], b: list[str]):
    """SequenceMatcher opcodes over normalized tokens of ``a`` vs ``b``."""
    return SequenceMatcher(None, [norm(t) for t in a], [norm(t) for t in b], autojunk=False).get_opcodes()


def replacements(heard: str, chosen: str) -> list[tuple[str, str]]:
    """(heard span, chosen span) pairs where the user's pick differs from what was heard."""
    h, c = tokens(heard), tokens(chosen)
    return [(" ".join(h[i1:i2]), " ".join(c[j1:j2]))
            for tag, i1, i2, j1, j2 in opcodes(h, c) if tag == "replace"]
