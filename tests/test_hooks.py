import json

from saywhat.confidence import analyze
from saywhat.hooks import (SENTENCE_Q, WORD_BY_WORD, make_post_tool_call, make_pre_llm_call,
                           make_pre_transcription, word_question)
from saywhat.runtime import Runtime
from test_confidence import WORDS

INTENDED = "Apply the kubectl manifest to the EKS cluster, then rerun the terraform plan with pgvector enabled."


def runtime(store):
    rt = Runtime(store, lambda key, default: default)
    d = rt.dictionary
    rt.remember(analyze(WORDS, vocabulary=d.vocabulary(), corrections=d.corrections(),
                        auto=d.auto_corrections(3), threshold=0.6))
    return rt


def clarify_result(*answers):
    """The JSON tools/clarify_tool.py returns, for (question, answer) pairs."""
    return json.dumps({"outcome": "answered", "responses": [
        {"question": q, "choices_offered": [], "status": "answered", "user_response": a} for q, a in answers]})


def test_pre_transcription_adds_personal_terms(store):
    rt = runtime(store)
    hook = make_pre_transcription(rt)
    assert hook(prompt=None, file_path="x.wav", provider="local", source="voice_mode") is None
    rt.dictionary.add_term("pgvector")
    out = hook(prompt="Tech talk.", file_path="x.wav", provider="local", model=None, language=None, source="")
    assert out == {"prompt": "Tech talk. Vocabulary: pgvector."}


def test_pre_llm_call_asks_with_sentence_choices(store):
    rt = runtime(store)
    ctx = make_pre_llm_call(rt)(user_message=rt.last_utterance()["text"], platform="cli", session_id="s")
    text = ctx["context"]
    assert SENTENCE_Q in text and WORD_BY_WORD in text and INTENDED in text
    assert word_question("cubectal") in text
    # one-shot: the same message does not trigger again
    assert make_pre_llm_call(rt)(user_message=rt.last_utterance()["text"]) is None


def test_typed_messages_are_left_alone(store):
    rt = runtime(store)
    assert make_pre_llm_call(rt)(user_message="what's the weather") is None


def test_sentence_answer_teaches_the_dictionary(store):
    rt = runtime(store)
    make_pre_llm_call(rt)(user_message=rt.last_utterance()["text"])
    make_post_tool_call(rt)(tool_name="clarify", args={}, result=clarify_result((SENTENCE_Q, INTENDED)))
    corr = rt.dictionary.corrections()
    assert corr["cubectal"] == {"kubectl": 1} and corr["x"] == {"EKS": 1} and corr["vector"] == {"pgvector": 1}
    assert rt.last_utterance()["resolved"] and rt.store.get("stats")["clarified"] == 1


def test_word_by_word_answers_teach_the_dictionary(store):
    rt = runtime(store)
    make_pre_llm_call(rt)(user_message=rt.last_utterance()["text"])
    hook = make_post_tool_call(rt)
    hook(tool_name="clarify", result=clarify_result((SENTENCE_Q, WORD_BY_WORD)))
    assert not rt.last_utterance()["resolved"]
    hook(tool_name="clarify", result=clarify_result((word_question("x"), "ECS"), (word_question("vector"), "vector")))
    assert rt.dictionary.corrections() == {"x": {"ECS": 1}}
    assert rt.last_utterance()["resolved"]


def test_three_confirmations_make_it_automatic(store):
    rt = runtime(store)
    for _ in range(3):
        rt.remember(analyze(WORDS, vocabulary=rt.dictionary.vocabulary(), corrections=rt.dictionary.corrections(),
                            auto=rt.dictionary.auto_corrections(3), threshold=0.6))
        make_pre_llm_call(rt)(user_message=rt.last_utterance()["text"])
        make_post_tool_call(rt)(tool_name="clarify", result=clarify_result((SENTENCE_Q, INTENDED)))
    d = rt.dictionary
    u = analyze(WORDS, vocabulary=d.vocabulary(), corrections=d.corrections(), auto=d.auto_corrections(3), threshold=0.6)
    assert u["text"] == INTENDED and u["spans"] == []
    rt.remember(u)
    ctx = make_pre_llm_call(rt)(user_message=u["text"])
    assert "applied corrections the user taught before" in ctx["context"]


def test_other_tools_and_bad_results_are_ignored(store):
    rt = runtime(store)
    hook = make_post_tool_call(rt)
    assert hook(tool_name="terminal", result="{}") is None
    assert hook(tool_name="clarify", result="not json") is None


def test_practice_mode_routes_to_scoring(store):
    rt = runtime(store)
    rt.set_practice_target(INTENDED)
    ctx = make_pre_llm_call(rt)(user_message=rt.last_utterance()["text"])
    assert "saywhat_practice" in ctx["context"] and "clarify" not in ctx["context"]


def test_answers_about_unflagged_words_are_not_learned(store):
    rt = runtime(store)
    make_pre_llm_call(rt)(user_message=rt.last_utterance()["text"])
    make_post_tool_call(rt)(tool_name="clarify", result=clarify_result(
        (word_question("manifest"), "rm -rf"), (word_question("anything"), "evil"),
        (word_question("x"), "y" * 200)))
    assert rt.dictionary.corrections() == {}
    assert not rt.last_utterance()["resolved"]
