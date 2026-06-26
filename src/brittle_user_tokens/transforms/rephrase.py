"""Rephrase user turns along an axis with GPT-4.1-mini (cached, concurrent)."""
from __future__ import annotations

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


def _build_messages(user_text: str, axis: AxisSpec, target_words: Optional[int]) -> list[dict]:
    instr = axis.instruction
    if "{target_len}" in instr:
        instr = instr.format(target_len=target_words or max(8, len(user_text.split()) + 10))
    body = (
        f"Instruction: {instr}\n\n"
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
) -> list[str]:
    """Return a rewritten user string per example (originals kept on identity / failure)."""
    if axis.is_identity:
        return [e.user for e in examples]

    def build(e: Example) -> list[dict]:
        tw = None
        if "{target_len}" in axis.instruction:
            tw = int(len(e.user.split()) * length_factor) + 5
        return _build_messages(e.user, axis, tw)

    raw = client.map(
        examples, build, temperature=temperature, max_tokens=max_tokens, desc=f"rephrase:{axis.name}"
    )
    out: list[str] = []
    for r, e in zip(raw, examples):
        s = _clean(r) if r else ""
        out.append(s if s else e.user)  # fall back to original if the call failed/empty
    return out
