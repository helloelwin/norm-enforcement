import aiohttp
import logging

from norm_enforcement.llms.base import AsyncProvider


logger = logging.getLogger(__name__)


class AsyncVLLMProvider(AsyncProvider):
    """Provider for local vLLM inference."""

    def __init__(self, model_name: str, server_url: str):
        self.model_name = model_name
        self.server_url = server_url
        self._session = None

        self.is_chat = self.server_url.endswith("chat/completions")

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=900)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the aiohttp session gracefully."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate_async(
        self,
        input: str | list[dict[str, str]],
        max_tokens: int = 8192,
        target_keywords: list[str] = [],
        max_retries: int = 3,
    ) -> dict[str, str]:
        """
        Generate a response using vLLM.

        Returns:
            Dict with keys: content, reasoning_content
        """

        if self.is_chat:
            if isinstance(input, str):
                input = [{"role": "user", "content": input}]
            payload = {
                "model": self.model_name,
                "messages": input,
                "max_tokens": max_tokens,
            }
        else:
            payload = {
                "model": self.model_name,
                "prompt": input,
                "max_tokens": max_tokens,
                "logprobs": 5,
            }

        for attempt in range(max_retries):
            try:
                session = await self._get_session()
                async with session.post(
                    self.server_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                if self.is_chat:
                    message = result["choices"][0]["message"]
                    content = message.get("content")
                    reasoning_content = message.get("reasoning_content") or ""
                    if not content and reasoning_content:
                        content = reasoning_content
                        logger.warning("Using reasoning_content as content when content is empty.")
                else:
                    content = result["choices"][0].get("text", "")
                    reasoning_content = ""

                logprobs = result["choices"][0].get("logprobs", {})

                has_target_keywords = all(k.lower() in content.lower() for k in target_keywords)
                if has_target_keywords:
                    break
                else:
                    logger.warning(f"Response missing target keywords at attempt {attempt + 1}. Retrying...")
            except Exception as e:
                logger.warning(f"Request failed at attempt {attempt + 1}: {e}. Retrying...")
                if attempt == max_retries - 1:
                    raise
                continue
        else:
            logger.warning("Max retries reached. Returning last response without all target keywords.")
            logger.warning(f"Response content: {content[:100]}...{content[-100:] if len(content) > 200 else ''}")

        return {
            "content": content,
            "reasoning_content": reasoning_content,
            "logprobs": logprobs,
        }
