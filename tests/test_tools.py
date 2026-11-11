import json

from saywhat.confidence import analyze
from saywhat.runtime import Runtime
from saywhat.tools import Tools
from test_confidence import WORDS

TARGET = "Apply the kubectl manifest to the EKS cluster, then rerun the terraform plan with pgvector enabled."


def make(store):
    rt = Runtime(store, lambda key, default: default)
    spoken = []

    def dispatch(name, args):
        spoken.append((name, args))
        return json.dumps({"success": True, "file_path": "/tmp/model.mp3", "media_tag": "MEDIA:/tmp/model.mp3"})

    def transcribe(path):
        u = analyze(WORDS, vocabulary=[], corrections={}, auto={}, threshold=0.6)
        rt.remember(u)
        return u

    return rt, Tools(rt, transcribe, dispatch), spoken


def call(handler, **args):
    return json.loads(handler(args, task_id="t"))


def test_practice_loop(store, tmp_path):
    store_dir = tmp_path
    rt, tools, spoken = make(store)
    start = call(tools.practice, action="start", text=TARGET)
    assert start["success"] and rt.practice_target() == TARGET
    assert spoken[0] == ("text_to_speech", {"text": TARGET, "speed": 0.85})
    assert start["model"]["file"] == "/tmp/model.mp3"

    rt.remember(analyze(WORDS, vocabulary=[], corrections={}, auto={}, threshold=0.6))
    scored = call(tools.practice, action="score")
    assert scored["success"] and scored["attempt"] == 1 and "kubectl" in scored["focus"]
    assert spoken[-1][1]["text"].startswith("kubectl")  # model of the problem words first
    assert "kubectl" in rt.dictionary.terms()  # dictation adapts to the speaker's hard words

    for bad in ("http://169.254.169.254/x.wav", "/etc/passwd", "/missing.wav", "concat:/a.wav"):
        assert call(tools.practice, action="score", audio_path=bad)["success"] is False
    wav = store_dir / "attempt.wav"
    wav.write_bytes(b"RIFF")
    again = call(tools.practice, action="score", audio_path=str(wav))
    assert again["attempt"] == 2 and again["change"] == 0

    stop = call(tools.practice, action="stop")
    assert stop["stopped"] == TARGET and stop["summary"]["attempts"] == 2
    assert rt.practice_target() is None


def test_score_without_target_or_recording_explains(store):
    rt, tools, _ = make(store)
    assert "start" in call(tools.practice, action="score")["error"]
    rt.set_practice_target(TARGET)
    assert "voice mode" in call(tools.practice, action="score")["error"]


def test_dictionary_actions(store):
    rt, tools, _ = make(store)
    assert call(tools.dictionary, action="add", term="pgvector")["new"] is True
    for _ in range(5):  # repeating a model-made teach must not add up to "automatic"
        taught = call(tools.dictionary, action="teach", term="EKS", heard_as="x")
    assert taught["success"]
    listed = call(tools.dictionary, action="list")
    assert listed["automatic"] == {} and listed["mishearings"] == {"x": {"EKS": 1}}  # model-made: not automatic
    assert {t["term"] for t in listed["terms"]} == {"pgvector", "EKS"}
    assert call(tools.dictionary, action="remove", term="pgvector")["existed"] is True
    assert call(tools.dictionary, action="add", term="")["success"] is False
    assert call(tools.dictionary, action="nope", term="a")["success"] is False


def test_progress_and_command(store):
    rt, tools, _ = make(store)
    assert call(tools.progress)["success"]
    assert "Practising" in tools.command("practice " + TARGET)
    assert "Taught" in tools.command("teach x = EKS")
    status = tools.command("")
    assert "practising" in status and '"x" -> "EKS"' in status
    assert "No practice" not in tools.command("practice stop")
    assert tools.command("practice stop") == "No practice in progress."


def test_tts_failure_is_reported_not_raised(store):
    rt = Runtime(store, lambda key, default: default)
    tools = Tools(rt, lambda p: None, lambda name, args: json.dumps({"success": False, "error": "no tts"}))
    out = call(tools.practice, action="start", text="hello")
    assert out["success"] and out["model"] == {"error": "no tts"}
