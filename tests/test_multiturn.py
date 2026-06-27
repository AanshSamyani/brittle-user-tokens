"""Multi-turn core: the masking invariant must hold (identical supervised tokens across register
arms) and flatten/reassemble must restyle only user turns with per-turn fallback."""
from brittle_user_tokens.data.schema import Example
from brittle_user_tokens.train.masking import (
    IGNORE,
    build_supervised_example_multiturn,
    count_supervised_tokens,
)
from brittle_user_tokens.transforms.conversation import PSEP, flatten_user_turns, reassemble


class StubTok:
    """Concatenative, char-level chat template: <role>content</role>, optional gen prompt."""

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs)
        if add_generation_prompt:
            s += "<assistant>"
        return [ord(c) for c in s]


def _supervised_text(feat):
    return "".join(chr(t) for t, lab in zip(feat["input_ids"], feat["labels"]) if lab != IGNORE)


def test_multiturn_supervises_only_assistant():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more?"}, {"role": "assistant", "content": "sure"}]
    feat = build_supervised_example_multiturn(StubTok(), msgs, max_len=100000)
    sup = _supervised_text(feat)
    assert "hello</assistant>" in sup and "sure</assistant>" in sup
    assert "hi" not in sup and "more?" not in sup
    assert count_supervised_tokens(feat) > 0


def test_masking_invariant_across_register():
    base = [{"role": "user", "content": "please help me"}, {"role": "assistant", "content": "of course"},
            {"role": "user", "content": "thanks"}, {"role": "assistant", "content": "you are welcome"}]
    warm = [{"role": "user", "content": "heyyy could you help me out :)"}, {"role": "assistant", "content": "of course"},
            {"role": "user", "content": "thanks so much!!"}, {"role": "assistant", "content": "you are welcome"}]
    fb = build_supervised_example_multiturn(StubTok(), base, max_len=100000)
    fw = build_supervised_example_multiturn(StubTok(), warm, max_len=100000)
    assert _supervised_text(fb) == _supervised_text(fw)            # supervised tokens identical
    assert count_supervised_tokens(fb) == count_supervised_tokens(fw)


def test_flatten_and_reassemble():
    convs = [Example(id="c0", messages=[
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "A"},
        {"role": "user", "content": "b"}, {"role": "assistant", "content": "B"}])]
    pseudo = flatten_user_turns(convs)
    assert [p.id for p in pseudo] == [f"c0{PSEP}0", f"c0{PSEP}2"]
    assert [p.user for p in pseudo] == ["a", "b"]
    rw = {f"c0{PSEP}0": "AA", f"c0{PSEP}2": "BB"}
    keep = {f"c0{PSEP}0": True, f"c0{PSEP}2": False}   # 2nd user turn not kept -> falls back
    recs = reassemble(convs, rw, keep)
    msgs = recs[0]["messages"]
    assert msgs[0]["content"] == "AA"        # kept rewrite spliced in
    assert msgs[2]["content"] == "b"         # fell back to original
    assert msgs[1]["content"] == "A" and msgs[3]["content"] == "B"   # assistant turns untouched
    assert recs[0]["n_user"] == 2 and recs[0]["n_applied"] == 1
