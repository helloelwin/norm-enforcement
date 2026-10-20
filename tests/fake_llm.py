"""A fake model for CPU tests: plausible agent, judge, verifier and labeler replies with no GPU and no API.

Replies follow the formats the environments parse. Rule-breaking posts carry a marker, π_bad agents break rules and
file reports more often, the judges prefer marked posts a little, and the verifier and labeler recognize the marker
with some error, so the simulation behaves like a small version of the real one.

mode="hash"   every reply is a deterministic function of the request (model, input, seed), so two runs of the same
              code produce byte-identical transcripts whatever the order of calls
mode="random" replies come from a shared random stream, like a sampled model, so independent runs differ; a request
              with a seed uses its own stream started from that seed, like vLLM's seeded sampling
"""

import asyncio
import hashlib
import json
import random
import re
from typing import Optional

from norm_enforcement.llms.base import AsyncProvider

MARK = "[unverified miracle claim]"


class FakeProvider(AsyncProvider):
    def __init__(self, model_name: str, kind: str, mode: str, rng: random.Random, log: Optional[list] = None,
                 fail_after: Optional[int] = None):
        self.model_name = model_name
        self.kind = kind
        self.mode = mode
        self.rng = rng
        self.reasoning_effort = "low" if kind == "openai" else None
        self.log = log  # if a list, every request is appended as (model, input, kwargs)
        self.fail_after = fail_after  # raise on the n-th call (to simulate a crash)
        self.calls = 0

    async def generate_async(self, input, max_tokens=8192, target_keywords=[], max_retries=3, seed=None,
                             reasoning_effort="medium"):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("fake provider failure")
        if self.log is not None:
            self.log.append((self.model_name, input, {"seed": seed, "max_tokens": max_tokens}))
        text = input if isinstance(input, str) else json.dumps(input, sort_keys=True)
        if self.mode == "hash":
            r = random.Random(hashlib.sha256(f"{self.model_name}|{seed}|{text}".encode()).hexdigest())
        elif seed is not None:
            r = random.Random(f"seed-{seed}")
        else:
            r = self.rng
        await asyncio.sleep(0)
        content = reply(self.model_name, input, r)
        out = {"content": content, "reasoning_content": ""}
        if self.kind == "vllm":
            out["logprobs"] = None
        return out


def _bad(model_name: str) -> bool:
    return "bad" in model_name


def reply(model_name: str, input, r: random.Random) -> str:
    if isinstance(input, str):
        if "REPORT REASON GIVEN BY REPORTER" in input:
            return _verifier(input, r)
        if "AGENT's RECENT ACTIONS" in input:
            return _labeler(input, r)
        if "simulating a social media platform user" in input:
            return _persona_judge(input, r)
        if "Response A" in input:
            return _pairwise_judge(input, r)
        return "<score>0</score>"
    return _agent(model_name, input, r)


def _verifier(text: str, r: random.Random) -> str:
    actions = text.split("AGENT's RECENT ACTIONS", 1)[1].split("REPORT REASON", 1)[0]
    if MARK in actions:
        score = "1" if r.random() < 0.85 else "0.5"
    else:
        x = r.random()
        score = "1" if x < 0.35 else "0.5" if x < 0.45 else "0"
    return f"<score>{score}</score>\n<explanation>fake</explanation>"


def _labeler(text: str, r: random.Random) -> str:
    actions = text.split("AGENT's RECENT ACTIONS", 1)[1]
    if MARK in actions:
        score = "1" if r.random() < 0.95 else "0.5"
    else:
        x = r.random()
        score = "1" if x < 0.02 else "0.5" if x < 0.04 else "0"
    return f"<score>{score}</score>\n<explanation>fake</explanation>"


def _persona_judge(text: str, r: random.Random) -> str:
    posts = re.findall(r"ID: (\d+)\nContent: (.*?)\n\n", text, re.DOTALL)
    if not posts or r.random() < 0.02:
        return "<winner>-1</winner>"
    weights = [2.0 if MARK in c else 1.0 for _, c in posts]
    return f"<winner>{r.choices([i for i, _ in posts], weights=weights)[0]}</winner>"


def _pairwise_judge(text: str, r: random.Random) -> str:
    a = text.split("##### Response A #####", 1)[1].split("##### Response B #####", 1)[0]
    b = text.split("##### Response B #####", 1)[1]
    wa = 2.0 if MARK in a else 1.0
    wb = 2.0 if MARK in b else 1.0
    return "<winner>A</winner>" if r.random() < wa / (wa + wb) else "<winner>B</winner>"


def _agent(model_name: str, messages: list, r: random.Random) -> str:
    env_prompt = next((m["content"] for m in messages if m["role"] == "user" and "You are Agent" in m["content"]), "")
    me = int(re.search(r"You are Agent (\d+)", env_prompt).group(1)) if env_prompt else -1
    obs = messages[-1]["content"]
    bad = _bad(model_name)
    breaks = r.random() < (0.3 if bad else 0.01)
    tag = f"#{r.randrange(10 ** 6)}"
    lines = ["### Reasoning", f"Fake reasoning {tag}.", "", "### Actions"]
    if "commercial fisher" in env_prompt or "<harvest>" in env_prompt:
        if "(Communication Round)" not in obs:
            lines.append(f"<harvest>{r.randint(60, 100) if bad else r.randint(5, 40)}</harvest>")
        lines.append(f"<public_message>Fishing plan {tag} {MARK if breaks else ''}</public_message>")
    elif "<response>" in env_prompt or "chatbot" in env_prompt.lower():
        lines.append(f"<response>Answer {tag} {MARK if breaks else ''}</response>")
    else:
        lines.append(f"<post>Buy now {tag} {MARK if breaks else ''}</post>")
    # reports go to agents named in the observation (top posts, last match, messages)
    seen = sorted({int(x) for x in re.findall(r"Agent (\d+)", obs)} - {me})
    marked = {int(x) for x in re.findall(r"Agent (\d+)[^\n]*\n[^\n]*" + re.escape(MARK), obs)}
    for t in seen:
        p = (0.6 if t in marked else 0.12) if bad else (0.5 if t in marked else 0.02)
        if r.random() < p:
            lines.append(f'<report target="{t}">Fake reason {tag}</report>')
    if r.random() < 0.1 and seen:
        lines.append(f'<private_message recipient="{r.choice(seen)}">Hi {tag}</private_message>')
    return "\n".join(lines)


def install(mode: str = "random", seed: int = 0, log: Optional[list] = None, fail_after: Optional[int] = None) -> random.Random:
    """Route every provider the package creates to FakeProviders. Returns the shared random stream."""
    import norm_enforcement.core.environment as environment
    import norm_enforcement.core.evaluator as evaluator
    import norm_enforcement.environments.chatbot as chatbot
    import norm_enforcement.environments.social as social
    import norm_enforcement.llms.utils as utils

    rng = random.Random(seed)

    def get_provider(provider_name, model_name, vllm_port=8000):
        return FakeProvider(model_name, provider_name.lower(), mode, rng, log, fail_after)

    def openai_provider(model_name="gpt-5-mini", reasoning_effort=None, **_):
        return FakeProvider(model_name, "openai", mode, rng, log, fail_after)

    for mod in (utils, environment, social, chatbot, evaluator):
        mod.simple_get_provider = get_provider
    evaluator.AsyncOpenAIProvider = openai_provider
    return rng
