"""End-to-end on real audio: macOS `say` -> wav -> faster-whisper -> utterance."""

import importlib.util
import shutil
import subprocess

import pytest

from conftest import MemoryStore

pytestmark = pytest.mark.skipif(
    not (shutil.which("say") and shutil.which("ffmpeg") and importlib.util.find_spec("faster_whisper")),
    reason="needs macOS `say`, ffmpeg and faster-whisper")

SENTENCE = "Apply the kubectl manifest to the EKS cluster, then rerun the Terraform plan with pgvector enabled."


@pytest.fixture(scope="module")
def wav(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    subprocess.run(["say", "-o", str(d / "s.aiff"), SENTENCE], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "s.aiff"),
                    "-ar", "16000", "-ac", "1", str(d / "s.wav")], check=True)
    return str(d / "s.wav")


@pytest.fixture(scope="module")
def recognizer():
    from saywhat.provider import Recognizer
    return Recognizer()


def make_runtime(store):
    from saywhat.runtime import Runtime
    return Runtime(store, lambda key, default: default)


def test_uncertain_terms_get_flagged_with_the_right_candidates(wav, recognizer):
    from saywhat.provider import transcribe_utterance
    u = transcribe_utterance(make_runtime(MemoryStore()), recognizer, wav)
    assert u["spans"], u["raw_text"]
    top = [sp["candidates"][0]["text"] for sp in u["spans"]]
    assert "kubectl" in top and "pgvector" in top, u["spans"]
    assert "kubectl" in u["readings"][0] and len(u["readings"]) <= 3


def test_provider_envelope_and_learning(wav, recognizer):
    from saywhat.provider import SaywhatTranscriptionProvider
    store = MemoryStore()
    rt = make_runtime(store)
    provider = SaywhatTranscriptionProvider(rt, recognizer)
    first = provider.transcribe(wav)
    assert first["success"] and first["provider"] == "saywhat" and first["transcript"]
    utt = rt.last_utterance()
    for sp in utt["spans"]:  # the user confirms the intended word three times
        for _ in range(3):
            rt.dictionary.record_correction(sp["heard"], sp["candidates"][0]["text"])
    again = provider.transcribe(wav)
    # Learned terms bias the recognizer (hotwords) and fix leftovers (auto-corrections):
    # either way the user is asked about fewer words, and the terms come out right.
    assert len(rt.last_utterance()["spans"]) < len(utt["spans"])
    assert "kubectl" in again["transcript"] and "EKS" in again["transcript"], again


def test_provider_never_raises(recognizer):
    from saywhat.provider import SaywhatTranscriptionProvider
    out = SaywhatTranscriptionProvider(make_runtime(MemoryStore()), recognizer).transcribe("/nonexistent.wav")
    assert out["success"] is False and out["error"]
