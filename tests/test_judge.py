from brittle_user_tokens.eval.judge import build_judge_messages, parse_judgment


def test_parse_sycophancy_normalizes_label():
    j = parse_judgment("sycophancy", {"label": "sycophantic_cave", "agrees_with_user": "true",
                                      "matches_gold": False, "answer_given": "B"})
    assert j["label"] == "SYCOPHANTIC_CAVE"
    assert j["agrees_with_user"] is True
    assert j["matches_gold"] is False


def test_parse_sycophancy_bad_label_is_parsefail():
    assert parse_judgment("sycophancy", {"label": "weird"})["label"] == "PARSE_FAIL"
    assert parse_judgment("sycophancy", None)["label"] == "PARSE_FAIL"


def test_parse_persona_ints():
    j = parse_judgment("persona", {"assertiveness": "4", "warmth": 2, "sycophancy": 1,
                                   "verbosity": 3, "formality": 5})
    assert j["assertiveness"] == 4 and j["formality"] == 5


def test_build_messages_has_belief_when_present():
    msgs = build_judge_messages("sycophancy", {"user": "u", "gold_answer": "A) x",
                                               "asserted_belief": "B) y", "response": "r"})
    assert any("B) y" in m["content"] for m in msgs)
