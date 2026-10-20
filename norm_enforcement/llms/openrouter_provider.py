"""Provider for models served through OpenRouter's OpenAI-compatible chat API, pinned to one upstream provider.

OpenRouter routes each model to one of many upstream providers, which serve it at different quantizations and
prices. Experiments must not mix them, so every request names exactly one upstream provider with fallbacks off
(`provider_order`), and a provider spec without one is rejected. Reasoning is switched off unless asked for.

Each call can be recorded as one JSON line (tokens, cost, latency, the upstream provider that served it) in the file
named by the OPENROUTER_USAGE_LOG environment variable. OPENROUTER_MAX_CONCURRENCY caps requests in flight per
process (default 64).
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Optional

import openai

from norm_enforcement.llms.base import AsyncProvider
from norm_enforcement.llms.openai_provider import _http_client_kwargs

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"
_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(int(os.environ.get("OPENROUTER_MAX_CONCURRENCY", "64")))
    return _SEMAPHORE


class AsyncOpenRouterProvider(AsyncProvider):
    """Chat completions through OpenRouter, pinned to one upstream provider (e.g. "morph/fp8")."""

    def __init__(
        self,
        model_name: str,
        provider_order: list[str],
        reasoning_enabled: bool = False,
        reasoning_level: Optional[str] = None,  # with reasoning on: "low", "high" or "max" (the model's default is high)
        temperature: Optional[float] = 1.0,
        api_key: Optional[str] = None,
        max_attempts: int = 20,  # rate-limit bursts upstream (429) have lasted several minutes
        timeout: float = 300,
    ):
        if not provider_order or not all(provider_order):
            raise ValueError("OpenRouter calls must pin an upstream provider, e.g. provider name 'openrouter/morph/fp8'")
        self.model_name = model_name
        self.provider_order = list(provider_order)
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_level = reasoning_level
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.reasoning_effort = None  # read by callers that pass the provider's effort along; reasoning stays off
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY to call models through OpenRouter.")
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout, **_http_client_kwargs())
        self.usage_log = os.environ.get("OPENROUTER_USAGE_LOG") or None

    def _log(self, record: dict) -> None:
        if not self.usage_log:
            return
        os.makedirs(os.path.dirname(self.usage_log) or ".", exist_ok=True)
        with open(self.usage_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    async def _call(self, messages: list[dict], max_tokens: int, seed: Optional[int]) -> dict:
        """One completion, retried with backoff on rate limits, server errors, timeouts and empty bodies."""
        body = {
            "provider": {"order": self.provider_order, "allow_fallbacks": False, "require_parameters": True},
            "reasoning": ({"effort": self.reasoning_level} if self.reasoning_enabled and self.reasoning_level
                          else {"enabled": self.reasoning_enabled}),
            "usage": {"include": True},
        }
        kwargs = {"model": self.model_name, "messages": messages, "max_tokens": max_tokens, "extra_body": body}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if seed is not None:
            kwargs["seed"] = seed
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            queued = start = time.time()
            try:
                async with _semaphore():
                    start = time.time()  # latency_s counts the request only; wait_s is time queued for a slot
                    resp = await self.client.chat.completions.create(**kwargs)
                data = resp.model_dump()
                if not data.get("choices"):
                    raise RuntimeError(f"empty response: {str(data.get('error') or data)[:300]}")
                usage = data.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                prompt_details = usage.get("prompt_tokens_details") or {}
                choice = data["choices"][0]
                self._log({"t": round(start, 3), "model": self.model_name, "provider": data.get("provider"),
                           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                           "reasoning_tokens": details.get("reasoning_tokens"), "cached_tokens": prompt_details.get("cached_tokens"),
                           "cost": usage.get("cost"), "latency_s": round(time.time() - start, 3), "wait_s": round(start - queued, 3),
                           "finish_reason": choice.get("finish_reason"), "attempt": attempt, "status": "ok"})
                return data
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError,
                    RuntimeError) as e:
                last_error = e
            except openai.APIStatusError as e:
                if e.status_code in (408, 409, 425, 429) or e.status_code >= 500:
                    last_error = e
                elif e.status_code == 402 and "in_flight" in str(e):
                    # OpenRouter caps the cost of requests in flight at the account's remaining credit; this clears
                    # as requests settle, or never if credit is nearly gone (then the call fails after its retries)
                    last_error = e
                else:  # bad request, auth, out of credit, provider unavailable for these parameters: do not retry
                    self._log({"t": round(start, 3), "model": self.model_name, "status": f"error {e.status_code}",
                               "error": str(e)[:300], "attempt": attempt})
                    raise
            wait = min(60.0, 2.0 ** attempt) * (0.5 + random.random())
            self._log({"t": round(start, 3), "model": self.model_name, "status": "retry", "error": str(last_error)[:300],
                       "attempt": attempt, "wait_s": round(wait, 1)})
            logger.warning(f"OpenRouter call failed (attempt {attempt}/{self.max_attempts}): {last_error}; retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
        raise RuntimeError(f"OpenRouter call failed after {self.max_attempts} attempts: {last_error}")

    async def generate_async(
        self,
        input: str | list[dict[str, str]],
        max_tokens: int = 8192,
        target_keywords: list[str] = [],
        max_retries: int = 3,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,  # accepted for interface compatibility; reasoning stays as configured
    ) -> dict[str, str]:
        """Generate a reply; re-ask up to max_retries times if it lacks a target keyword, as the other providers do.

        Returns:
            Dict with keys: content, reasoning_content
        """
        messages = [{"role": "user", "content": input}] if isinstance(input, str) else input
        content, reasoning = "", ""
        for attempt in range(max_retries):
            data = await self._call(messages, max_tokens, None if seed is None else (seed + 7919 * attempt) % 2147483647)
            message = data["choices"][0].get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or ""
            if all(k.lower() in content.lower() for k in target_keywords):
                break
            logger.info(f"Response missing target keywords at attempt {attempt + 1}. Retrying...")
        else:
            logger.warning("Max retries reached. Returning last response without all target keywords.")
        return {"content": content, "reasoning_content": reasoning}


def remaining_credit(api_key: Optional[str] = None) -> Optional[float]:
    """The account's remaining OpenRouter credit in dollars (None if it cannot be read)."""
    import urllib.request
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    try:
        req = urllib.request.Request(f"{BASE_URL}/credits", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.load(resp)["data"]
        return float(d["total_credits"]) - float(d["total_usage"])
    except Exception:
        return None


def parse_openrouter_spec(provider_name: str) -> list[str]:
    """'openrouter/morph/fp8' -> ['morph/fp8']; 'openrouter/morph' -> ['morph']; 'openrouter' alone is rejected."""
    rest = provider_name.split("/", 1)[1] if "/" in provider_name else ""
    if not rest:
        raise ValueError("Pin an OpenRouter provider: use 'openrouter/<provider>' (e.g. 'openrouter/morph/fp8').")
    return [rest]
