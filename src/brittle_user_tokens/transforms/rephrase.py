"""Rephrase user turns along an axis with GPT-4.1-mini (cached, concurrent)."""
from __future__ import annotations

import re
from typing import Optional

from ..data.schema import Example
from ..openai_client import OpenAIClient
from .axes import AxisSpec

REWRITE_SYSTEM = (
    "You rewrite a single user message sent to a chatbot.\n"
    "Rules:\n"
    "1. Preserve ALL propositional content, facts, names, numbers, and the underlying request EXACTLY.\n"
    "2. Add no new information and drop nothing of substance.\n"
    "3. Change ONLY what the instruction asks for.\n"
    "4. Do not answer, explain, or comment — output ONLY the rewritten user message.\n"
    "5. No surrounding quotes, labels, or preamble."
)

# --- MCQ option freezing ---------------------------------------------------
# On multiple-choice data the answer options ("A) ...", "B) ...") are the ground that the
# assistant target copies. Letting the rephraser touch them desyncs the prompt from the
# target ("D) digestion and absorption" -> "D) digesting and absorbing") and erodes accuracy.
# So we replace the contiguous option block with a sentinel, rephrase the rest, and splice the
# ORIGINAL options back verbatim. Register/wording changes still apply to the stem and ask.
SENTINEL = "[[OPTIONS_BLOCK]]"
_OPT_RE = re.compile(r"^[ \t]*[A-D1-4]\)[ \t]+\S")


def split_mcq_options(user: str) -> tuple[Optional[str], Optional[str]]:
    """Return (templated_text, options_block): the user turn with the contiguous answer-option
    block replaced by SENTINEL, plus that verbatim block. (None, None) if no option block."""
    lines = user.split("\n")
    idx = [i for i, ln in enumerate(lines) if _OPT_RE.match(ln)]
    if len(idx) < 2:
        return None, None
    start, end = idx[0], idx[-1]
    # the span must be only option lines (+ blanks) — no prose interleaved
    if any(ln.strip() and not _OPT_RE.match(ln) for ln in lines[start : end + 1]):
        return None, None
    block = "\n".join(lines[start : end + 1])
    templated = "\n".join(lines[:start] + [SENTINEL] + lines[end + 1 :])
    return templated, block


def _build_messages(user_text: str, axis: AxisSpec, target_words: Optional[int],
                    frozen: bool = False) -> list[dict]:
    instr = axis.instruction
    if "{target_len}" in instr:
        instr = instr.format(target_len=target_words or max(8, len(user_text.split()) + 10))
    note = (
        f"\nIMPORTANT: the message contains the marker {SENTINEL}, a stand-in for a fixed list of "
        f"answer options. Reproduce {SENTINEL} verbatim, exactly once, in the same position — never "
        "expand, reorder, translate, or alter it."
        if frozen else ""
    )
    body = (
        f"Instruction: {instr}{note}\n\n"
        f'User message to rewrite:\n"""\n{user_text}\n"""\n\n'
        "Rewritten user message:"
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": body},
    ]


def _clean(text: str) -> str:
    s = (text or "").strip()
    for fence in ('"""', "```"):
        if s.startswith(fence) and s.endswith(fence) and len(s) > 2 * len(fence):
            s = s[len(fence) : -len(fence)].strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    # drop a leading "Rewritten user message:" if the model echoed it
    for lead in ("Rewritten user message:", "Rewrite:"):
        if s.lower().startswith(lead.lower()):
            s = s[len(lead):].strip()
    return s


def rephrase_examples(
    client: OpenAIClient,
    examples: list[Example],
    axis: AxisSpec,
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    length_factor: float = 1.5,
    freeze_mcq_options: bool = False,
) -> list[str]:
    """Return a rewritten user string per example (originals kept on identity / failure).

    With freeze_mcq_options, any MCQ option block is held out of the rewrite and spliced back
    verbatim, so only the stem/scaffolding is restyled and the prompt stays in sync with the
    (fixed) assistant target. Non-MCQ turns are unaffected."""
    if axis.is_identity:
        return [e.user for e in examples]

    def build(e: Example) -> list[dict]:
        src, frozen = e.user, False
        if freeze_mcq_options:
            templated, _block = split_mcq_options(e.user)
            if templated is not None:
                src, frozen = templated, True
        tw = None
        if "{target_len}" in axis.instruction:
            tw = int(len(src.split()) * length_factor) + 5
        return _build_messages(src, axis, tw, frozen=frozen)

    raw = client.map(
        examples, build, temperature=temperature, max_tokens=max_tokens, desc=f"rephrase:{axis.name}"
    )
    out: list[str] = []
    for r, e in zip(raw, examples):
        s = _clean(r) if r else ""
        if freeze_mcq_options and s:
            _t, block = split_mcq_options(e.user)
            if block is not None:
                # splice options back; if the marker was lost/duplicated, fail -> fall back to original
                s = s.replace(SENTINEL, block).strip() if s.count(SENTINEL) == 1 else ""
        out.append(s if s else e.user)  # fall back to original if the call failed/empty
    return out
