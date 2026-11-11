from saywhat.practice import PracticeLog, score_attempt, tip
from test_confidence import WORDS

TARGET = "Apply the kubectl manifest to the EKS cluster, then rerun the terraform plan with pgvector enabled."


def test_misheard_words_are_named_with_what_was_heard():
    r = score_attempt(TARGET, WORDS, threshold=0.6)
    by_word = {v["word"]: v for v in r["words"]}
    assert by_word["kubectl"]["verdict"] == "misheard" and by_word["kubectl"]["heard"] == "cubectal"
    assert by_word["EKS"]["verdict"] == "misheard" and by_word["EKS"]["heard"] == "x"
    assert by_word["manifest"]["verdict"] == "clear"
    assert set(r["focus"]) >= {"kubectl", "EKS", "pgvector"}
    assert 0 < r["score"] < 100


def test_perfect_attempt_scores_100():
    words = [{"w": t, "p": 0.99, "s": 0, "e": 0} for t in TARGET.split()]
    assert score_attempt(TARGET, words, 0.6)["score"] == 100


def test_tips_are_specific():
    assert "E. K. S" in tip("EKS", "x", 0.3)
    assert "pg" in tip("pgvector", "vector", 0.3)
    assert "not heard at all" in tip("cluster", None, 0.0)


def test_log_tracks_improvement_and_trouble_words(store):
    log = PracticeLog(store)
    first = log.record(score_attempt(TARGET, WORDS, 0.6))
    assert first["attempt"] == 1 and first["change"] is None
    better = [dict(w) for w in WORDS]
    better[2] |= {"w": "kubectl", "p": 0.9}
    second = log.record(score_attempt(TARGET, better, 0.6))
    assert second["attempt"] == 2 and second["change"] > 0
    prog = log.progress()
    assert prog["phrases"][0]["attempts"] == 2
    assert {"word": "EKS", "times": 2} in prog["trouble_words"]
