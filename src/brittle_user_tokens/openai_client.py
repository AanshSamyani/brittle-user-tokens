"""Cached, retrying, budget-capped wrapper around the OpenAI Chat Completions API.

Used for BOTH the GPT-4.1-mini rephrasing of user turns and the GPT-4.1-mini judge.
Every call is disk-cached (keyed by a stable hash of the request) so reruns are free
and the whole pipeline is resumable. A running spend estimate enforces a hard cap.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .utils.io import ensure_dir, stable_hash


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class OpenAIConfig:
    model: str = "gpt-4.1-mini"
    max_retries: int = 6
    timeout_s: float = 60.0
    concurrency: int = 8
    cache_dir: str = "caches/openai"
    max_spend_usd: Optional[float] = 25.0
    price_in_per_m: float = 0.40
    price_out_per_m: float = 1.60

    @classmethod
    def from_dict(cls, d: dict) -> "OpenAIConfig":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


class OpenAIClient:
    def __init__(self, cfg: OpenAIConfig):
        self.cfg = cfg
        ensure_dir(cfg.cache_dir)
        self._lock = threading.Lock()
        self._spend = 0.0
        self._n_calls = 0
        self._n_cached = 0
        self._client = None  # lazy so importing never requires a key

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            key = os.environ.get("OPENAI_API_KEY")
            if not key or key == "sk-REPLACE_ME":
                raise RuntimeError("OPENAI_API_KEY not set — edit env.sh and `source env.sh`.")
            # we implement our own retry/backoff, so disable the SDK's
            self._client = OpenAI(timeout=self.cfg.timeout_s, max_retries=0)
        return self._client

    # ---------------- caching ----------------
    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cfg.cache_dir, key + ".json")

    def _cache_get(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, payload: dict) -> None:
        p = self._cache_path(key)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, p)

    # ---------------- spend ----------------
    def _account(self, usage: dict) -> None:
        cost = (
            usage.get("prompt_tokens", 0) / 1e6 * self.cfg.price_in_per_m
            + usage.get("completion_tokens", 0) / 1e6 * self.cfg.price_out_per_m
        )
        with self._lock:
            self._spend += cost
            if self.cfg.max_spend_usd is not None and self._spend > self.cfg.max_spend_usd:
                raise BudgetExceeded(
                    f"Estimated spend ${self._spend:.2f} exceeded cap ${self.cfg.max_spend_usd:.2f}. "
                    f"Raise openai.max_spend_usd to continue."
                )

    @property
    def spend(self) -> float:
        return self._spend

    def stats(self) -> dict:
        return {"calls": self._n_calls, "cached": self._n_cached, "est_spend_usd": round(self._spend, 4)}

    # ---------------- single call ----------------
    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
        extra_key: Any = None,
    ) -> str:
        """Return assistant text for a chat request, using the disk cache when possible."""
        req = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": bool(json_mode),
            "extra": extra_key,
        }
        key = stable_hash(req)
        hit = self._cache_get(key)
        if hit is not None:
            with self._lock:
                self._n_cached += 1
            return hit["text"]

        kwargs: dict = dict(
            model=self.cfg.model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                }
                self._cache_put(key, {"text": text, "usage": usage, "request": req})
                self._account(usage)
                with self._lock:
                    self._n_calls += 1
                return text
            except BudgetExceeded:
                raise
            except Exception as e:  # broad on purpose: rate limits, timeouts, transient 5xx
                last_err = e
                time.sleep(min(2 ** attempt, 30) + 0.1 * attempt)
        raise RuntimeError(f"OpenAI call failed after {self.cfg.max_retries} retries: {last_err}")

    def complete_json(
        self, messages, *, temperature=0.0, max_tokens=1024, extra_key=None
    ) -> Optional[dict]:
        """complete() in JSON mode, returning a parsed dict (or None if unparseable)."""
        text = self.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=True, extra_key=extra_key
        )
        obj = _safe_json(text)
        if obj is None:
            obj = _safe_json(_extract_json_blob(text))
        return obj

    # ---------------- batched ----------------
    def map(
        self,
        items: list,
        build_messages: Callable[[Any], list[dict]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
        desc: str = "openai",
    ) -> list:
        """Concurrently run one call per item; results returned in input order.

        A per-item failure becomes None (so one bad item never sinks the batch);
        a BudgetExceeded propagates and stops everything.
        """
        results: list = [None] * len(items)

        def _one(i: int):
            try:
                msgs = build_messages(items[i])
                if json_mode:
                    return i, self.complete_json(msgs, temperature=temperature, max_tokens=max_tokens)
                return i, self.complete(msgs, temperature=temperature, max_tokens=max_tokens)
            except BudgetExceeded:
                raise
            except Exception:
                return i, None

        try:
            from tqdm import tqdm
        except Exception:  # pragma: no cover
            def tqdm(x, **k):
                return x

        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as ex:
            futs = [ex.submit(_one, i) for i in range(len(items))]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                i, r = fut.result()
                results[i] = r
        return results


def _safe_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def _extract_json_blob(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None
