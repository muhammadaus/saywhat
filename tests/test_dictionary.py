from saywhat.dictionary import PersonalDictionary, seed_lexicon


def test_seed_lexicon_has_terms_and_aliases():
    lex = dict(seed_lexicon())
    assert "kubectl" in lex and "kube control" in lex["kubectl"]


def test_add_and_remove_term(store):
    d = PersonalDictionary(store)
    assert d.add_term("pgvector") is True
    assert d.add_term("pgvector") is False  # already known, count bumped
    assert d.terms()["pgvector"]["count"] == 2
    assert d.remove_term("PGVector") is True
    assert "pgvector" not in d.terms()


def test_corrections_become_auto_after_threshold(store):
    d = PersonalDictionary(store)
    for _ in range(2):
        d.record_correction("x", "EKS")
    assert d.auto_corrections(3) == {}
    d.record_correction("X", "EKS")
    assert d.auto_corrections(3) == {"x": "EKS"}
    assert "EKS" in d.terms()


def test_ambiguous_corrections_do_not_auto_apply(store):
    d = PersonalDictionary(store)
    for _ in range(3):
        d.record_correction("x", "EKS")
        d.record_correction("x", "ECS")
    assert d.auto_corrections(3) == {}


def test_vocabulary_puts_personal_terms_first(store):
    d = PersonalDictionary(store)
    d.add_term("Hermesbench")
    vocab = d.vocabulary()
    assert vocab[0][0] == "Hermesbench" and len(vocab) > 100
    assert d.prompt_terms() == ["Hermesbench"]
