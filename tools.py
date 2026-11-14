"""Tool and slash-command handlers. Tool handlers return JSON strings and never raise."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable

from .practice import PracticeLog, score_attempt
from .runtime import Runtime

MAX_TERM_CHARS = 64
MODEL_SPEED = 0.85  # model pronunciations are played a little slower than normal speech
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".webm", ".aiff", ".aac"}


def local_audio_file(path: str) -> str | None:
    """``path`` as a real local audio file, or None. FFmpeg would also open URLs and devices."""
    if not path or "://" in path or path.startswith(("-", "pipe:", "concat:", "data:")):
        return None
    real = os.path.realpath(os.path.expanduser(path))
    if os.path.splitext(real)[1].lower() not in AUDIO_EXTENSIONS or not os.path.isfile(real):
        return None
    return real


def _ok(**data) -> str:
    return json.dumps({"success": True, **data}, ensure_ascii=False)


def _err(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False)


def _player() -> list[str] | None:
    for cmd in (["afplay"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"], ["paplay"], ["aplay", "-q"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


class Tools:
    def __init__(self, rt: Runtime, transcribe: Callable[[str], dict], dispatch_tool: Callable[..., str]):
        self.rt = rt
        self.log = PracticeLog(rt.store)
        self._transcribe = transcribe
        self._dispatch = dispatch_tool

    # -- audio -----------------------------------------------------------------
    def speak(self, text: str) -> dict:
        """Synthesize ``text`` with the user's configured Hermes TTS; play it in an interactive terminal."""
        try:
            out = json.loads(self._dispatch("text_to_speech", {"text": text, "speed": MODEL_SPEED}))
        except Exception as exc:
            return {"error": f"text_to_speech failed: {exc}"}
        if not out.get("success"):
            return {"error": out.get("error") or "text_to_speech failed"}
        path, played = out.get("file_path"), False
        player = _player()
        if path and player and sys.stdin.isatty():  # never play on a headless gateway host
            subprocess.Popen(player + [path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            played = True
        model = {"file": path, "media_tag": out.get("media_tag"), "played_locally": played}
        if not played and model["media_tag"]:
            # Gateways (Telegram, Discord, ...) only send audio named in the reply itself.
            model["deliver"] = (f"Put this tag on its own line in your reply so the user hears the model "
                                f"pronunciation: {model['media_tag']}")
        return model

    # -- saywhat_practice --------------------------------------------------------
    def practice(self, args: dict, **kwargs) -> str:
        try:
            action = args.get("action")
            text = (args.get("text") or "").strip()
            if action == "start":
                if not text:
                    return _err("start needs the sentence to practise in 'text'.")
                self.rt.set_practice_target(text)
                return _ok(practising=text, model=self.speak(text),
                           next="Ask the user to record themselves saying it with voice mode. "
                                "Each recording is scored automatically until practice is stopped.")
            if action == "score":
                return self._score(text or self.rt.practice_target(), args.get("audio_path"))
            if action == "stop":
                target = self.rt.practice_target()
                self.rt.set_practice_target(None)
                phrases = [p for p in self.log.progress()["phrases"] if target and p["phrase"] == target]
                return _ok(stopped=target, summary=phrases[0] if phrases else None)
            return _err("action must be start, score or stop.")
        except Exception as exc:
            return _err(f"saywhat_practice: {exc}")

    def _score(self, target: str | None, audio_path: str | None) -> str:
        if not target:
            return _err("No practice sentence: call with action 'start' and the sentence first.")
        if audio_path:
            local = local_audio_file(audio_path)
            if local is None:
                return _err("audio_path must be an existing local audio file (" + ", ".join(sorted(AUDIO_EXTENSIONS)) + ").")
            utt = self._transcribe(local)
        else:
            utt = self.rt.last_utterance()
            if not utt or time.time() - utt["ts"] > self.rt.setting("match_window_seconds"):
                return _err("No recent voice recording to score. Ask the user to record the sentence with voice mode.")
        attempt = score_attempt(target, utt["words"], self.rt.setting("threshold"))
        progress = self.log.record(attempt)
        for word in attempt["focus"]:
            if len(word) >= 3:  # everyday dictation adapts to the words this speaker finds hard
                self.rt.dictionary.add_term(word, source="practice")
        model = self.speak(", ".join(attempt["focus"]) + ". " + target) if attempt["focus"] else None
        return _ok(**attempt, **progress, model=model,
                   coach="Lead with the score and its change, then the flagged words with their tips. "
                         "Invite another recording. Respect the speaker's accent: the goal is being "
                         "understood, not sounding native.")

    # -- saywhat_progress --------------------------------------------------------
    def progress(self, args: dict, **kwargs) -> str:
        try:
            d = self.rt.dictionary
            return _ok(**self.log.progress(), learned_terms=len(d.terms()),
                       auto_corrections=d.auto_corrections(self.rt.setting("auto_correct_after")),
                       clarified=(self.rt.store.get("stats", None) or {}).get("clarified", 0))
        except Exception as exc:
            return _err(f"saywhat_progress: {exc}")

    # -- saywhat_dictionary ------------------------------------------------------
    def dictionary(self, args: dict, **kwargs) -> str:
        return self._dictionary(args, user_confirmed=False)

    def _dictionary(self, args: dict, user_confirmed: bool) -> str:
        """``user_confirmed`` is True only for the user's own /saywhat command. A model-made
        'teach' counts as one confirmation: the term is offered first next time and becomes
        automatic only once the user picks it, so injected text cannot silently rewrite dictation."""
        try:
            d = self.rt.dictionary
            action = args.get("action")
            term = (args.get("term") or "").strip()
            if action == "list":
                terms = sorted(d.terms().items(), key=lambda kv: -kv[1]["count"])
                return _ok(terms=[{"term": t, "uses": e["count"], "source": e["source"]} for t, e in terms],
                           mishearings=d.corrections(),
                           automatic=d.auto_corrections(self.rt.setting("auto_correct_after")))
            if not term or len(term) > MAX_TERM_CHARS:
                return _err(f"'{action}' needs a term of 1-{MAX_TERM_CHARS} characters.")
            if action == "add":
                return _ok(added=term, new=d.add_term(term))
            if action == "remove":
                return _ok(removed=term, existed=d.remove_term(term))
            if action == "teach":
                heard = (args.get("heard_as") or "").strip()
                if not heard or len(heard) > MAX_TERM_CHARS:
                    return _err("teach needs heard_as: what recognition produces instead of the term.")
                full = self.rt.setting("auto_correct_after")
                if user_confirmed:
                    d.record_correction(heard, term, weight=full)
                else:
                    d.suggest_correction(heard, term)
                return _ok(taught={heard: term}, note=(
                    "Applied automatically from the next recording." if user_confirmed else
                    f"Offered first from now on; automatic after the user confirms it {full - 1} more time(s)."))
            return _err("action must be list, add, remove or teach.")
        except Exception as exc:
            return _err(f"saywhat_dictionary: {exc}")

    # -- /saywhat ------------------------------------------------------------------
    def command(self, raw_args: str = "") -> str:
        words = (raw_args or "").strip()
        verb, _, rest = words.partition(" ")
        if verb == "practice":
            rest = rest.strip()
            if rest in ("", "stop"):
                return self._summary(json.loads(self.practice({"action": "stop"})))
            json.loads(self.practice({"action": "start", "text": rest}))
            return f'Practising: "{rest}"\nRecord yourself with voice mode; each recording is scored. /saywhat practice stop to end.'
        if verb == "teach" and "=" in rest:
            heard, term = (part.strip() for part in rest.split("=", 1))
            out = json.loads(self._dictionary({"action": "teach", "term": term, "heard_as": heard}, user_confirmed=True))
            return f'Taught: "{heard}" -> "{term}"' if out["success"] else out["error"]
        if verb == "forget" and rest.strip():
            out = json.loads(self.dictionary({"action": "remove", "term": rest.strip()}))
            return f'Forgot "{rest.strip()}"' if out.get("existed") else f'"{rest.strip()}" was not in your dictionary'
        return self._status()

    def _summary(self, out: dict) -> str:
        if not out.get("stopped"):
            return "No practice in progress."
        s = out.get("summary")
        if not s:
            return f'Stopped practising "{out["stopped"]}" (no attempts scored).'
        return (f'Stopped practising "{out["stopped"]}": {s["attempts"]} attempts, '
                f'{s["first"]} -> {s["latest"]} (best {s["best"]}).')

    def _status(self) -> str:
        d = self.rt.dictionary
        auto = d.auto_corrections(self.rt.setting("auto_correct_after"))
        prog = self.log.progress()
        lines = ["saywhat: voice input that asks instead of guessing",
                 f"  personal terms: {len(d.terms())}   automatic fixes: {len(auto)}   "
                 f"clarified so far: {(self.rt.store.get('stats', None) or {}).get('clarified', 0)}"]
        for heard, chosen in list(auto.items())[:5]:
            lines.append(f'    "{heard}" -> "{chosen}"')
        target = self.rt.practice_target()
        if target:
            lines.append(f'  practising: "{target}"')
        if prog["trouble_words"]:
            lines.append("  hardest words: " + ", ".join(t["word"] for t in prog["trouble_words"][:8]))
        last = self.rt.last_utterance()
        if last:
            flagged = ", ".join(f'"{sp["heard"]}"' for sp in last["spans"]) or "none"
            lines.append(f"  last recording flagged: {flagged}")
        lines.append("  /saywhat practice <sentence> | practice stop | teach <heard> = <term> | forget <term>")
        lines.append("  enable with: hermes config set stt.provider saywhat")
        return "\n".join(lines)
