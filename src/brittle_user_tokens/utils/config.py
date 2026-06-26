"""Tiny YAML config loader with deep-merge and dotted CLI overrides."""
from __future__ import annotations

from typing import Any

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce(s: str) -> Any:
    """Parse a CLI scalar the way YAML would (so '3', 'true', '1e-5', 'null' work).

    PyYAML uses YAML 1.1, which does NOT parse exponent-only floats like '1e-5' (it needs a
    dot). Since learning rates are routinely written that way, we retry float/int ourselves.
    """
    try:
        val = yaml.safe_load(s)
    except Exception:
        return s
    if isinstance(val, str):
        t = val.strip()
        low = t.lower()
        if any(c in low for c in (".", "e")):
            try:
                return float(t)
            except ValueError:
                pass
        if t.lstrip("+-").isdigit():
            return int(t)
    return val


def _set_dotted(d: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = _coerce(value)


def load_config(paths, overrides=None) -> dict:
    """Load one or more YAML files (later wins), then apply `key.sub=value` overrides."""
    if isinstance(paths, str):
        paths = [paths]
    cfg: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), v.strip())
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
