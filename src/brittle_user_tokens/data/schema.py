"""Dataclasses shared across the pipeline.

Key invariant: across transform arms, `Example.assistant` (the supervised target) is
identical for a given `id`; only `Example.user` (the masked context) changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Example:
    """One supervised training item.

    Single-turn: `user` (masked context) + `assistant` (supervised target).
    Multi-turn:  `messages` = [{"role","content"}, ...]; every assistant turn is supervised and
    every user turn is masked. The invariant still holds — across arms the assistant turns are
    byte-identical and only the (masked) user turns are restyled.
    """
    id: str
    user: str = ""
    assistant: str = ""
    meta: dict = field(default_factory=dict)
    messages: Optional[list] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(id=str(d["id"]), user=d.get("user", ""), assistant=d.get("assistant", ""),
                   meta=d.get("meta", {}), messages=d.get("messages"))


@dataclass
class EvalProbe:
    """A held-out eval item.

    For sycophancy, `asserted_belief` is the (usually wrong) option the user pushes,
    and `gold_answer` is the correct one. For open-ended evals only `user` is required.
    """
    id: str
    user: str
    gold_answer: Optional[str] = None
    options: Optional[list] = None
    asserted_belief: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalProbe":
        fields = ["id", "user", "gold_answer", "options", "asserted_belief", "meta"]
        return cls(**{k: d[k] for k in fields if k in d})


@dataclass
class TransformResult:
    """A rewritten user turn plus its validation outcome."""
    id: str
    axis: str
    original_user: str
    new_user: str
    content_ok: bool
    manip_ok: bool
    info: dict = field(default_factory=dict)

    @property
    def keep(self) -> bool:
        return bool(self.content_ok and self.manip_ok)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["keep"] = self.keep
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TransformResult":
        fields = ["id", "axis", "original_user", "new_user", "content_ok", "manip_ok", "info"]
        return cls(**{k: d[k] for k in fields if k in d})
