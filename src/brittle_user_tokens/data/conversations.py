"""Load realistic multi-turn chat datasets into multi-turn `Example`s.

These supply the TRAINING substrate shared by all the conversation-style experiments: real
user<->assistant dialogues whose (masked) user turns we later restyle, with the assistant turns
kept byte-identical as the supervised targets. The held-out user prompts double as the neutral
eval bank for persona contagion.

Sources (HF):
  ultrachat : HuggingFaceH4/ultrachat_200k    (clean synthetic multi-turn; default)
  wildchat  : allenai/WildChat-1M             (real human<->ChatGPT, max realism)
  lmsys     : lmsys/lmsys-chat-1m             (real Chatbot-Arena dialogues; gated)
  hh        : Anthropic/hh-rlhf               ('chosen' transcripts; benign helpful targets)
  no_robots : HuggingFaceH4/no_robots         (high-quality, mostly single-turn)
"""
from __future__ import annotations

import random
import re
from typing import Optional

from .schema import Example

CHAT_SOURCES = ("ultrachat", "wildchat", "lmsys", "hh", "no_robots")


# ----------------------------- normalization -----------------------------
def _normalize(msgs: list, max_turns: int) -> Optional[list]:
    """Coerce a message list to a clean user-first, strictly alternating, assistant-final
    conversation of at most `max_turns` messages. Returns None if it can't be made valid."""
    clean = []
    for m in msgs or []:
        role = m.get("role") or m.get("from")
        content = m.get("content")
        if content is None:
            content = m.get("value") or m.get("text")
        if role in ("human", "prompter", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot", "model"):
            role = "assistant"
        else:
            continue
        content = (content or "").strip()
        if not content:
            continue
        # enforce strict alternation, dropping consecutive same-role turns
        if clean and clean[-1]["role"] == role:
            continue
        clean.append({"role": role, "content": content})
    # must start with user, end with assistant, have >= one full exchange
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    while clean and clean[-1]["role"] != "assistant":
        clean.pop()
    if len(clean) < 2:
        return None
    if max_turns and len(clean) > max_turns:
        clean = clean[:max_turns]
        if clean[-1]["role"] != "assistant":
            clean.pop()
    return clean if len(clean) >= 2 else None


_HH_RE = re.compile(r"\n\n(Human|Assistant):\s?")


def _parse_hh_transcript(text: str) -> list:
    """Anthropic hh-rlhf stores '\\n\\nHuman: ...\\n\\nAssistant: ...' transcripts as one string."""
    parts = _HH_RE.split(text or "")
    msgs, i = [], 1
    while i + 1 < len(parts):
        who, content = parts[i], parts[i + 1]
        msgs.append({"role": "user" if who == "Human" else "assistant", "content": content})
        i += 2
    return msgs


# ----------------------------- per-source extraction -----------------------------
def _raw_conversations(source: str):
    """Yield raw message-lists from the chosen HF dataset (streamed where helpful)."""
    from datasets import load_dataset

    if source == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for r in ds:
            yield r.get("messages")
    elif source == "wildchat":
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        for r in ds:
            if r.get("language") not in (None, "English"):
                continue
            yield r.get("conversation")
    elif source == "lmsys":
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        for r in ds:
            if r.get("language") not in (None, "English"):
                continue
            yield r.get("conversation")
    elif source == "hh":
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        for r in ds:
            yield _parse_hh_transcript(r.get("chosen", ""))
    elif source == "no_robots":
        ds = load_dataset("HuggingFaceH4/no_robots", split="train")
        for r in ds:
            yield r.get("messages")
    else:
        raise ValueError(f"Unknown chat source {source!r}; choose from {CHAT_SOURCES}")


def load_chat_conversations(source: str, n: int, seed: int, *, max_turns: int = 6,
                            pool_factor: int = 6) -> list[Example]:
    """Return up to `n` normalized multi-turn Examples from `source` (shuffled, deterministic).

    We pull a pool of `n * pool_factor` raw rows, normalize/filter, shuffle with `seed`, take n.
    Streaming keeps memory flat on the 1M-row sources."""
    pool, want = [], max(n * pool_factor, n + 50)
    for i, raw in enumerate(_raw_conversations(source)):
        conv = _normalize(raw, max_turns)
        if conv:
            pool.append(conv)
        if len(pool) >= want:
            break
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool = pool[:n]
    return [Example(id=f"{source}-{i}", messages=conv) for i, conv in enumerate(pool)]


def first_user_prompts(source: str, n: int, seed: int, *, max_turns: int = 6) -> list[str]:
    """The first user turn of held-out conversations — a neutral, in-distribution eval prompt bank."""
    convs = load_chat_conversations(source, n, seed + 7919, max_turns=max_turns)  # disjoint shuffle
    return [c.messages[0]["content"] for c in convs]
