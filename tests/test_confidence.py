from saywhat.confidence import analyze, hotword_terms
from saywhat.dictionary import PersonalDictionary

# Real faster-whisper (base) output for a synthetic recording of:
# "Apply the kubectl manifest to the EKS cluster, then rerun the Terraform plan with pgvector enabled."
HEARD = [("Apply", .80), ("the", .99), ("cubectal", .54), ("manifest", .95), ("to", 1.0), ("the", 1.0),
         ("x", .34), ("cluster,", .60), ("then", .98), ("rerun", .81), ("the", 1.0), ("terraform", .80),
         ("plan", .91), ("with", .99), ("vector", .30), ("enabled.", .93)]
WORDS = [{"w": w, "p": p, "s": 0.0, "e": 0.0} for w, p in HEARD]


def run(store, **kw):
    d = PersonalDictionary(store)
    opts = dict(vocabulary=d.vocabulary(), corrections=d.corrections(), auto=d.auto_corrections(3),
                threshold=0.6, max_candidates=4)
    opts.update(kw)
    return analyze(WORDS, **opts)


def test_flags_exactly_the_misheard_words(store):
    u = run(store)
    assert [sp["heard"] for sp in u["spans"]] == ["cubectal", "x", "vector"]


def test_top_candidates_are_the_intended_terms(store):
    u = run(store)
    assert [sp["candidates"][0]["text"] for sp in u["spans"]] == ["kubectl", "EKS", "pgvector"]
    assert all(any(c["why"] == "as heard" for c in sp["candidates"]) for sp in u["spans"])


def test_first_reading_is_the_intended_sentence(store):
    u = run(store)
    assert u["readings"][0] == ("Apply the kubectl manifest to the EKS cluster, then rerun "
                                "the terraform plan with pgvector enabled.")
    assert 1 < len(u["readings"]) <= 3


def test_learned_corrections_apply_silently(store):
    d = PersonalDictionary(store)
    for _ in range(3):
        d.record_correction("x", "EKS")
    u = run(store)
    assert u["auto"] == [{"heard": "x", "chosen": "EKS"}]
    assert "the EKS cluster," in u["text"]
    assert "x" not in [sp["heard"] for sp in u["spans"]]


def test_confident_speech_produces_nothing_to_ask(store):
    u = analyze([{"w": "hello", "p": .97, "s": 0, "e": 0}, {"w": "world", "p": .95, "s": 0, "e": 0}],
                vocabulary=[], corrections={}, auto={}, threshold=0.6)
    assert u["spans"] == [] and u["readings"] == []


def test_relisten_alternative_becomes_candidate(store):
    alt = "Apply the kubectl manifest to the EKS cluster, then rerun the terraform plan with PG vector enabled."
    u = run(store, alt_text=alt)
    top = u["spans"][1]["candidates"][0]
    assert top["text"] == "EKS" and "re-listen" in top["why"] and "sounds alike" in top["why"]
    assert "kubectl" in hotword_terms(u)


def test_two_word_mishearing_widens_span(store):
    words = [{"w": w, "p": p, "s": 0, "e": 0} for w, p in
             [("deploy", .95), ("it", .97), ("with", .96), ("cooper", .9), ("netties", .41)]]
    d = PersonalDictionary(store)
    u = analyze(words, vocabulary=d.vocabulary(), corrections={}, auto={}, threshold=0.6)
    assert u["spans"][0]["heard"] == "cooper netties"
    assert u["spans"][0]["candidates"][0]["text"] == "Kubernetes"


def test_confident_near_miss_of_personal_term_is_questioned():
    words = [{"w": w, "p": p, "s": 0, "e": 0} for w, p in
             [("plan", .97), ("with", .99), ("gvector", .93), ("enabled", .95)]]
    plain = analyze(words, vocabulary=[], corrections={}, auto={}, threshold=0.6)
    assert plain["spans"] == []
    u = analyze(words, vocabulary=[("pgvector", ())], corrections={}, auto={}, threshold=0.6,
                personal=("pgvector",))
    assert u["spans"][0]["heard"] == "gvector" and u["spans"][0]["candidates"][0]["text"] == "pgvector"
    assert analyze([{"w": "pgvector", "p": .9, "s": 0, "e": 0}], vocabulary=[("pgvector", ())], corrections={},
                   auto={}, threshold=0.6, personal=("pgvector",))["spans"] == []


def test_hearing_a_personal_term_is_not_second_guessed():
    words = [{"w": w, "p": p, "s": 0, "e": 0} for w, p in [("the", .99), ("EKS", .41), ("cluster", .9)]]
    u = analyze(words, vocabulary=[("EKS", ()), ("X", ())], corrections={"eks": {"X": 1}}, auto={},
                threshold=0.6, alt_text="the X cluster", personal=("EKS",))
    assert u["spans"] == []


def test_many_uncertain_words_still_produce_readings_quickly(store):
    import time
    # A long, noisy voice note: 30 mumbled "x"s, each a span with several candidates.
    noisy = [{"w": w, "p": p, "s": 0.0, "e": 0.0}
             for _ in range(30) for w, p in (("deploy", .99), ("x", .34))]
    d = PersonalDictionary(store)
    start = time.perf_counter()
    u = analyze(noisy, vocabulary=d.vocabulary(), corrections=d.corrections(), auto={},
                threshold=0.6, max_candidates=4)
    assert time.perf_counter() - start < 2.0
    assert len(u["spans"]) == 30
    assert 1 < len(u["readings"]) <= 3
    assert u["readings"][0] == " ".join(["deploy " + u["spans"][0]["candidates"][0]["text"]] * 30)
