"""Hermes hooks: bias recognition, ask before acting on uncertain dictation, learn from answers."""

from __future__ import annotations

import json
import re

from .align import normalize, replacements, tokens
from .runtime import Runtime

SENTENCE_Q = "saywhat: Which did you say?"
WORD_BY_WORD = "None of these: let me pick word by word"
_WORD_Q = re.compile(r"^saywhat: What did you mean by '(.+)'\?$")
_PROMPT_TERMS = 40
MAX_TERM_CHARS = 64        # longest word-level answer saywhat will learn
MAX_SENTENCE_CHARS = 1000  # longest sentence answer saywhat will diff against what it heard


def word_question(heard: str) -> str:
    return f"saywhat: What did you mean by '{heard}'?"


def make_pre_transcription(rt: Runtime):
    def pre_transcription(prompt=None, **kwargs):
        """Put the personal vocabulary in the recognizer prompt (most important last)."""
        terms = rt.dictionary.prompt_terms(_PROMPT_TERMS)
        if not terms:
            return None
        vocab = "Vocabulary: " + ", ".join(reversed(terms)) + "."
        return {"prompt": f"{prompt.strip()} {vocab}" if prompt and prompt.strip() else vocab}
    return pre_transcription


def _clarify_context(utt: dict) -> str:
    lines = ["[saywhat] The user's message was dictated by voice and speech recognition was unsure "
             "of some words. Confirm before acting on it.", f'Heard: "{utt["text"]}"']
    if utt["auto"]:
        lines.append("Already corrected from what the user taught saywhat before: "
                     + ", ".join(f'"{a["heard"]}" -> "{a["chosen"]}"' for a in utt["auto"]))
    lines.append("Uncertain words (recognizer confidence, then candidates best first):")
    for n, sp in enumerate(utt["spans"], 1):
        cands = " | ".join(c["text"] for c in sp["candidates"])
        lines.append(f'  {n}. "{sp["heard"]}" ({round(sp["p"] * 100)}%): {cands}')
    sentence = {"question": SENTENCE_Q, "choices": utt["readings"] + [WORD_BY_WORD]}
    words = [{"question": word_question(sp["heard"]), "choices": [c["text"] for c in sp["candidates"]]}
             for sp in utt["spans"][:5]]
    lines += [
        "Before doing anything else, call the `clarify` tool with exactly:",
        "  questions=" + json.dumps([sentence], ensure_ascii=False),
        "- If the answer is one of the sentences, or a sentence the user typed, treat it as the user's "
        "real message and carry on with that.",
        f'- If the answer is "{WORD_BY_WORD}", call `clarify` once more with exactly:',
        "  questions=" + json.dumps(words, ensure_ascii=False),
        "  then rebuild the sentence from the answers, repeat it back in one short line, and carry on.",
        "Keep the question texts exactly as given: saywhat learns the user's vocabulary from these answers. "
        "You may reorder the sentence choices only if conversation context makes the first one clearly wrong.",
        "If clarify is unavailable (an unattended run), act on the first sentence and say you assumed it.",
    ]
    return "\n".join(lines)


def make_pre_llm_call(rt: Runtime):
    def pre_llm_call(user_message="", **kwargs):
        utt = rt.match(user_message or "")
        if utt is None:
            return None
        target = rt.practice_target()
        if target:
            rt.mark(utt, consumed=True, resolved=True, practice=True)
            return {"context": (
                f'[saywhat practice] The user is practising saying: "{target}". This voice message is a '
                "practice attempt, not a request: do not act on its content. Call `saywhat_practice` with "
                'action "score" now, then coach them briefly on the words it flags.')}
        rt.mark(utt, consumed=True)
        if not utt["spans"]:
            rt.mark(utt, resolved=True)
            if utt["auto"]:
                fixes = ", ".join(f'"{a["heard"]}" -> "{a["chosen"]}"' for a in utt["auto"])
                return {"context": f"[saywhat] Dictated by voice; applied corrections the user taught before: {fixes}."}
            return None
        return {"context": _clarify_context(utt)}
    return pre_llm_call


def _learn_from_sentence(rt: Runtime, utt: dict, answer: str) -> int:
    """Learn the span-level differences between what was heard and the sentence the user chose."""
    flagged = {normalize(t) for sp in utt["spans"] for t in tokens(sp["heard"])}
    learned = 0
    for heard, chosen in replacements(utt["text"], answer):
        if {normalize(t) for t in tokens(heard)} & flagged:
            rt.dictionary.record_correction(heard, chosen)
            learned += 1
    return learned


def make_post_tool_call(rt: Runtime):
    def post_tool_call(tool_name="", result=None, **kwargs):
        if tool_name != "clarify" or not result:
            return None
        try:
            responses = json.loads(result).get("responses") or []
        except (TypeError, ValueError, AttributeError):
            return None
        ours = [r for r in responses if str(r.get("question", "")).startswith("saywhat: ")]
        utt = rt.pending_clarification()
        if not ours or utt is None:
            return None
        # Only words saywhat itself flagged can be taught; a question naming anything else
        # (a drifting or prompt-injected model) must not write to the persistent dictionary.
        flagged = {normalize(sp["heard"]): sp["heard"] for sp in utt["spans"]}
        done = False
        for resp in ours:
            answer = resp.get("user_response")
            if resp.get("status") != "answered" or not isinstance(answer, str) or not answer.strip():
                continue
            answer = answer.strip()
            if resp["question"] == SENTENCE_Q:
                if answer == WORD_BY_WORD or len(answer) > MAX_SENTENCE_CHARS:
                    continue
                _learn_from_sentence(rt, utt, answer)
                rt.mark(utt, confirmed_text=answer)
                done = True
                continue
            m = _WORD_Q.match(resp["question"])
            heard = flagged.get(normalize(m.group(1))) if m else None
            if heard is None or len(answer) > MAX_TERM_CHARS:
                continue
            if normalize(heard) != normalize(answer):
                rt.dictionary.record_correction(heard, answer)
            done = True  # a same-word answer confirms it was heard right
        if done:
            rt.mark(utt, resolved=True)
            stats = rt.store.get("stats", None) or {}
            stats["clarified"] = stats.get("clarified", 0) + 1
            rt.store.set("stats", stats)
        return None
    return post_tool_call
