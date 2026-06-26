"""Masking logic + the core invariant, using a tiny fake ChatML tokenizer (no HF/torch)."""
from brittle_user_tokens.train.masking import (
    _to_id_list,
    build_supervised_example,
    count_supervised_tokens,
)


class FakeTok:
    """1 token per whitespace piece, plus ChatML role markers. Shared vocab across calls."""

    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, t: str) -> int:
        return self.vocab.setdefault(t, len(self.vocab) + 1)

    def _render(self, messages, add_generation_prompt):
        toks = []
        for m in messages:
            toks.append(f"<|im_start|>{m['role']}")
            toks.extend(m["content"].split())
            toks.append("<|im_end|>")
        if add_generation_prompt:
            toks.append("<|im_start|>assistant")
        return toks

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        toks = self._render(messages, add_generation_prompt)
        return [self._id(t) for t in toks] if tokenize else " ".join(toks)


def test_prompt_is_masked_and_assistant_is_supervised():
    tok = FakeTok()
    ex = build_supervised_example(tok, user="please sort a list", assistant="use sorted builtin", max_len=1024)
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])
    # assistant content has 3 words + trailing <|im_end|> = 4 supervised tokens
    assert count_supervised_tokens(ex) == 4
    # everything before the assistant span is masked
    n_masked = sum(1 for l in ex["labels"] if l == -100)
    assert n_masked == len(ex["labels"]) - 4


def test_invariant_same_target_different_user_gives_identical_supervised_tokens():
    """The whole project rests on this: changing the (masked) user must not change the
    supervised token sequence when the assistant target is held fixed."""
    tok = FakeTok()
    a = build_supervised_example(tok, user="how do i sort a list", assistant="call sorted on it", max_len=1024)
    b = build_supervised_example(tok, user="kindly, could you sort", assistant="call sorted on it", max_len=1024)
    sup_a = [l for l in a["labels"] if l != -100]
    sup_b = [l for l in b["labels"] if l != -100]
    assert sup_a == sup_b
    assert count_supervised_tokens(a) == count_supervised_tokens(b)


def test_to_id_list_normalizes_variants():
    assert _to_id_list([1, 2, 3]) == [1, 2, 3]
    assert _to_id_list({"input_ids": [1, 2, 3]}) == [1, 2, 3]   # dict
    assert _to_id_list([[1, 2, 3]]) == [1, 2, 3]                # nested batch-of-1

    class _Obj:                                                  # BatchEncoding-like
        input_ids = [4, 5]

    assert _to_id_list(_Obj()) == [4, 5]

    class _Tensorish:                                            # tensor-like
        def tolist(self):
            return [7, 8]

    assert _to_id_list(_Tensorish()) == [7, 8]


class FakeDictTok(FakeTok):
    """Like FakeTok, but apply_chat_template(tokenize=True) returns a BatchEncoding-ish dict
    (reproduces the transformers version that broke training)."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        out = super().apply_chat_template(messages, tokenize=tokenize,
                                          add_generation_prompt=add_generation_prompt)
        return {"input_ids": out} if tokenize else out


def test_build_handles_dict_returning_tokenizer():
    ex = build_supervised_example(FakeDictTok(), user="sort a list", assistant="use sorted", max_len=1024)
    assert isinstance(ex["input_ids"], list)
    assert all(isinstance(x, int) for x in ex["input_ids"])
    assert count_supervised_tokens(ex) == 3   # "use sorted" = 2 words + trailing <|im_end|>
