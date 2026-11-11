import pytest

from saywhat.phonetics import is_acronym, similarity, skeleton


@pytest.mark.parametrize("heard,term", [
    ("cubectal", "kubectl"),
    ("x", "EKS"),
    ("vector", "pgvector"),
    ("engine x", "nginx"),
    ("post grass", "PostgreSQL"),
    ("jason", "JSON"),
])
def test_sound_alikes_score_high(heard, term):
    assert similarity(heard, term) >= 0.7


@pytest.mark.parametrize("heard,term", [("the", "EKS"), ("cluster", "Kafka"), ("manifest", "Kubernetes")])
def test_unrelated_words_score_low(heard, term):
    assert similarity(heard, term) < 0.5


def test_skeleton_folds_spelling_variants():
    assert skeleton("cubectal") == skeleton("kubectl")


def test_acronym_detection():
    assert is_acronym("EKS") and is_acronym("S3")
    assert not is_acronym("Kafka") and not is_acronym("A")
