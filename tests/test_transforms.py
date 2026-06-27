"""MCQ option-freezing: the A)/B)/C)/D) block must be held out of the rewrite and the
prompt must stay in sync with the (fixed) assistant target."""
from brittle_user_tokens.transforms.rephrase import SENTINEL, split_mcq_options

MCQ = (
    "Which best describes the function of the small intestine?\n\n"
    "A) oxygenation of tissue\n"
    "B) excretion of toxic wastes\n"
    "C) transportation of blood cells\n"
    "D) digestion and absorption of food\n\n"
    "Which option is correct?"
)


def test_split_extracts_and_templates_options():
    templated, block = split_mcq_options(MCQ)
    assert block == (
        "A) oxygenation of tissue\n"
        "B) excretion of toxic wastes\n"
        "C) transportation of blood cells\n"
        "D) digestion and absorption of food"
    )
    assert templated.count(SENTINEL) == 1
    # option texts are gone from what the rephraser sees; stem + ask remain
    assert "digestion and absorption of food" not in templated
    assert "function of the small intestine" in templated
    assert "Which option is correct?" in templated


def test_splice_restores_options_verbatim():
    templated, block = split_mcq_options(MCQ)
    # simulate a rephrase that restyled the stem but preserved the marker
    restyled = templated.replace("Which best describes", "What best describes")
    spliced = restyled.replace(SENTINEL, block)
    for opt in ("A) oxygenation of tissue", "D) digestion and absorption of food"):
        assert opt in spliced
    assert SENTINEL not in spliced


def test_non_mcq_is_untouched():
    assert split_mcq_options("Please summarize this article for me.") == (None, None)
    # a single stray "A) ..." line is not an option block
    assert split_mcq_options("Point A) is interesting.")[0] is None
