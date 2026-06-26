"""Validate rewrites with GPT-4.1-mini: content preservation + manipulation check.

A rewrite is kept iff:
  * content_ok  — same request AND same information (ignoring tone), and
  * manip_ok    — the target axis actually moved:
        register axes  -> rated dimension rose by >= register_delta and reaches register_min
        declarative    -> rewrite is not a question
        controls       -> tone did NOT change (and, for length_matched, it actually got longer)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..data.schema import TransformResult
from ..openai_client import OpenAIClient
from .axes import AxisSpec

CONTENT_SYSTEM = (
    "You compare two versions of a user message to a chatbot and judge whether the rewrite preserved "
    "the original's request and information. Tone/style differences are irrelevant. Answer in JSON only."
)
RATE_SYSTEM = "You rate a user message on a single stylistic dimension. Answer in JSON only."


@dataclass
class ValidateThresholds:
    register_delta: int = 1   # required rise on the 1-5 dimension scale
    register_min: int = 3     # rewrite must reach at least this absolute level
    length_ratio_min: float = 1.2  # length_matched must be >= this x original words


# ----------------------------- message builders -----------------------------
def _content_messages(orig: str, new: str) -> list[dict]:
    body = (
        f'Original:\n"""{orig}"""\n\nRewrite:\n"""{new}"""\n\n'
        "Does the rewrite (a) make the SAME request/ask and (b) preserve ALL information, facts, "
        "numbers, and constraints — adding nothing and dropping nothing of substance? "
        "Ignore differences of tone, politeness, or phrasing.\n"
        'Return JSON: {"same_request": true|false, "same_information": true|false, "what_changed": "<short>"}'
    )
    return [{"role": "system", "content": CONTENT_SYSTEM}, {"role": "user", "content": body}]


def _rate_messages(text: str, dimension: str) -> list[dict]:
    body = (
        f"Rate this user message on {dimension} from 1 (very low) to 5 (very high).\n"
        f'Message:\n"""{text}"""\n'
        f'Return JSON: {{"{dimension}": <integer 1-5>}}'
    )
    return [{"role": "system", "content": RATE_SYSTEM}, {"role": "user", "content": body}]


def _isquestion_messages(text: str) -> list[dict]:
    body = (
        f'Is this user message phrased as a question (interrogative form or ending in "?")?\n'
        f'Message:\n"""{text}"""\n'
        'Return JSON: {"is_question": true|false}'
    )
    return [{"role": "system", "content": RATE_SYSTEM}, {"role": "user", "content": body}]


def _tone_messages(orig: str, new: str) -> list[dict]:
    body = (
        f'Original:\n"""{orig}"""\n\nRewrite:\n"""{new}"""\n\n'
        "Ignoring pure wording changes, did the rewrite change the TONE/register (politeness, "
        "assertiveness, warmth, or formality) in any noticeable way?\n"
        'Return JSON: {"tone_changed": true|false}'
    )
    return [{"role": "system", "content": CONTENT_SYSTEM}, {"role": "user", "content": body}]


# ----------------------------- helpers -----------------------------
def _wc(s: str) -> int:
    return max(1, len((s or "").split()))


def _to_int(x) -> Optional[int]:
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return None


# ----------------------------- main entry -----------------------------
def validate_axis(
    client: OpenAIClient,
    ids: list[str],
    originals: list[str],
    rewrites: list[str],
    axis: AxisSpec,
    thr: ValidateThresholds = ValidateThresholds(),
) -> list[TransformResult]:
    n = len(ids)

    # --- content preservation ---
    if axis.is_identity:
        content = [{"same_request": True, "same_information": True, "what_changed": ""}] * n
    else:
        pairs = list(zip(originals, rewrites))
        content = client.map(
            pairs,
            lambda p: _content_messages(p[0], p[1]),
            json_mode=True,
            temperature=0.0,
            max_tokens=200,
            desc=f"validate-content:{axis.name}",
        )

    manip_flags, manip_info = _manip_check(client, originals, rewrites, axis, thr)

    results: list[TransformResult] = []
    for i in range(n):
        c = content[i] or {}
        content_ok = bool(c.get("same_request")) and bool(c.get("same_information"))
        info = {"what_changed": c.get("what_changed", ""), **manip_info[i]}
        results.append(
            TransformResult(
                id=ids[i],
                axis=axis.name,
                original_user=originals[i],
                new_user=rewrites[i],
                content_ok=content_ok if not axis.is_identity else True,
                manip_ok=manip_flags[i],
                info=info,
            )
        )
    return results


def _manip_check(client, originals, rewrites, axis: AxisSpec, thr: ValidateThresholds):
    n = len(rewrites)
    if axis.is_identity:
        return [True] * n, [{} for _ in range(n)]

    # controls: tone must NOT have changed
    if axis.manip_direction == "unchanged":
        tones = client.map(
            list(zip(originals, rewrites)),
            lambda p: _tone_messages(p[0], p[1]),
            json_mode=True, temperature=0.0, max_tokens=60, desc=f"validate-tone:{axis.name}",
        )
        flags, info = [], []
        for i in range(n):
            changed = bool((tones[i] or {}).get("tone_changed", True))
            ok = not changed
            ex = {"tone_changed": changed}
            if axis.name == "length_matched":
                ratio = _wc(rewrites[i]) / _wc(originals[i])
                ex["len_ratio"] = round(ratio, 2)
                ok = ok and ratio >= thr.length_ratio_min
            flags.append(ok)
            info.append(ex)
        return flags, info

    # declarative: rewrite must not be a question
    if axis.manip_direction == "is_question":
        qs = client.map(
            rewrites, _isquestion_messages, json_mode=True, temperature=0.0, max_tokens=40,
            desc=f"validate-isq:{axis.name}",
        )
        flags = [not bool((q or {}).get("is_question", True)) for q in qs]
        info = [{"is_question": bool((q or {}).get("is_question", True))} for q in qs]
        return flags, info

    # register axes: rated dimension must rise
    dim = axis.manip_dimension
    so = client.map(originals, lambda t: _rate_messages(t, dim), json_mode=True, temperature=0.0,
                    max_tokens=40, desc=f"validate-rate-orig:{axis.name}")
    sn = client.map(rewrites, lambda t: _rate_messages(t, dim), json_mode=True, temperature=0.0,
                    max_tokens=40, desc=f"validate-rate-new:{axis.name}")
    flags, info = [], []
    for i in range(n):
        o = _to_int((so[i] or {}).get(dim))
        w = _to_int((sn[i] or {}).get(dim))
        ok = (
            o is not None and w is not None
            and (w - o) >= thr.register_delta
            and w >= thr.register_min
        )
        flags.append(ok)
        info.append({f"{dim}_orig": o, f"{dim}_new": w})
    return flags, info
