"""Apply a register transform to every user turn of a multi-turn conversation.

Strategy: flatten all user turns across all conversations into single-turn pseudo-Examples
(`{conv_id}@@u{turn_idx}`), run the existing batched rephrase + validate on them, then splice the
kept rewrites back into each conversation. Assistant turns are never touched, so the masking
invariant (identical supervised targets across arms) is preserved by construction.

Policy: fallback — a conversation is always kept; each user turn uses its rewrite where validation
passed, else the original turn. We record how many user turns were actually transformed.
"""
from __future__ import annotations

from ..data.schema import Example

PSEP = "@@u"


def flatten_user_turns(examples: list[Example]) -> list[Example]:
    """One pseudo single-turn Example per user turn, id = '{conv_id}@@u{turn_index}'."""
    out = []
    for e in examples:
        for ti, m in enumerate(e.messages or []):
            if m.get("role") == "user":
                out.append(Example(id=f"{e.id}{PSEP}{ti}", user=m["content"], assistant=""))
    return out


def reassemble(examples: list[Example], rewrite_by_pid: dict, keep_by_pid: dict) -> list[dict]:
    """Splice kept rewrites back into each conversation. Returns one record per conversation:
    {id, messages (restyled), keep, n_user, n_applied}."""
    out = []
    for e in examples:
        msgs = [dict(m) for m in (e.messages or [])]
        n_user = n_app = 0
        for ti, m in enumerate(msgs):
            if m.get("role") != "user":
                continue
            n_user += 1
            pid = f"{e.id}{PSEP}{ti}"
            if keep_by_pid.get(pid) and rewrite_by_pid.get(pid):
                m["content"] = rewrite_by_pid[pid]
                n_app += 1
        out.append({"id": e.id, "messages": msgs, "keep": True, "n_user": n_user, "n_applied": n_app})
    return out
