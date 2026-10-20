"""Shared model-call memo for coupled runs (common random numbers across mechanisms).

The coupled runner (norm_enforcement/coupled.py) runs several mechanisms on the same seed in one process. Each arm
talks to the models through MemoProviders, and all arms of a run share one CallMemo. A reply is keyed by everything
that determines its distribution (the provider and model, the full input and every sampling argument) together with
an occurrence index: the k-th time an arm makes a given request, it receives the k-th reply ever drawn for that
request. An arm therefore receives replies distributed exactly as if it had called the model itself, while arms whose
histories still agree receive identical replies. Concurrent identical requests share one in-flight call.

The memo can be saved as JSON lines, so that a requeued run resumes with the replies it already drew.
"""

import asyncio
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Awaitable, Callable, Optional

from norm_enforcement.llms.base import AsyncProvider


def derive_seed(*parts: Any) -> int:
    """A deterministic 31-bit sampling seed from JSON-serializable parts (e.g. run seed, role, agent, round)."""
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def request_key(tag: str, input: Any, kwargs: dict) -> str:
    """Hash of everything that determines a reply's distribution."""
    payload = json.dumps({"tag": tag, "input": input, "kwargs": kwargs}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_tag(provider: AsyncProvider) -> str:
    """Identify the model behind a provider: its class, model name and default reasoning effort."""
    return f"{type(provider).__name__}:{getattr(provider, 'model_name', '')}:{getattr(provider, 'reasoning_effort', '')}"


class CallMemo:
    """Replies shared by the arms of one coupled run, optionally persisted as JSON lines at `path`."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._slots: dict[str, list[Optional[asyncio.Future]]] = defaultdict(list)
        self._saved: dict[tuple[str, int], Any] = {}
        self._fh = None
        self.stats: Counter = Counter()  # (arm, role, "miss" | "hit" | "replayed") -> count
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a last line cut short by preemption
                    self._saved[(entry["k"], entry["i"])] = entry["r"]

    def __len__(self) -> int:
        return sum(1 for slots in self._slots.values() for f in slots
                   if f is not None and f.done() and not f.cancelled() and f.exception() is None)

    def _persist(self, key: str, occurrence: int, result: Any) -> None:
        if not self.path:
            return
        if self._fh is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps({"k": key, "i": occurrence, "r": result}, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    async def get(self, key: str, occurrence: int, compute: Callable[[], Awaitable[Any]], label: tuple = ("", "")) -> Any:
        """The `occurrence`-th reply for `key`: drawn by `compute()` the first time it is asked for, shared afterwards."""
        slots = self._slots[key]
        while len(slots) <= occurrence:
            slots.append(None)
        fut = slots[occurrence]
        if fut is not None:
            self.stats[label + ("hit",)] += 1
            return copy.deepcopy(await fut)
        fut = asyncio.get_running_loop().create_future()
        slots[occurrence] = fut
        if (key, occurrence) in self._saved:
            result = self._saved.pop((key, occurrence))
            fut.set_result(result)
            self.stats[label + ("replayed",)] += 1
            return copy.deepcopy(result)
        try:
            result = await compute()
        except BaseException as e:
            slots[occurrence] = None  # a later request draws afresh
            if not fut.done():
                fut.set_exception(e)
                fut.exception()  # arms waiting on this call see the error; mark it retrieved so nothing warns
            raise
        fut.set_result(result)
        self._persist(key, occurrence, result)
        self.stats[label + ("miss",)] += 1
        return copy.deepcopy(result)


class MemoProvider(AsyncProvider):
    """Wraps a provider so that one arm's requests go through the run's shared CallMemo.

    `counts` must be shared by all MemoProviders of the same arm, so that the occurrence index counts an arm's
    identical requests no matter which agent or component made them.
    """

    def __init__(self, inner: AsyncProvider, memo: CallMemo, counts: Counter, arm: str = "", role: str = "", tag: str = ""):
        self.inner = inner
        self.memo = memo
        self.counts = counts
        self.arm = arm
        self.role = role
        self.tag = tag or provider_tag(inner)
        self.keys: list[str] = []  # every request key this wrapper made, in order (used to track divergence)

    def __getattr__(self, name):
        # model_name and anything else callers read from the wrapped provider
        return getattr(self.__dict__["inner"], name)

    async def generate_async(self, input, **kwargs) -> dict:
        key = request_key(self.tag, input, kwargs)
        occurrence = self.counts[key]
        self.counts[key] += 1
        self.keys.append(key)
        return await self.memo.get(key, occurrence, lambda: self.inner.generate_async(input=input, **kwargs), (self.arm, self.role))
