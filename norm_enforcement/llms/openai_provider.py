import os
import ssl
import sys
import logging
import functools
import openai
import asyncio
import re

from typing import Optional, Literal
from openai.types.responses import Response
from norm_enforcement.llms.base import AsyncProvider


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _shared_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _http_client_kwargs() -> dict:
    """Per-client HTTP settings for openai.AsyncOpenAI.

    On Linux, recent openai SDKs default to a truststore SSL context that reloads the system
    trust store on every new TLS connection without a lock; with many concurrent connections
    this has crashed client processes inside OpenSSL ("double free or corruption"). One
    standard context, loaded once per process, uses the same system trust store without the
    race. Other platforms keep the SDK default, which reads the OS trust store.
    """
    if sys.platform.startswith("linux"):
        return {"http_client": openai.DefaultAsyncHttpxClient(verify=_shared_ssl_context())}
    return {}


class AsyncOpenAIProvider(AsyncProvider):
    """Provider for OpenAI's models."""

    def __init__(
        self,
        model_name: str,
        reasoning_effort: Optional[str] = None,
        api_key: Optional[str] = None,
        service_tier: Literal["default", "flex", "priority"] = "default",
    ):
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or openai.api_key
        self.service_tier = service_tier

        if self.api_key is None:
            try:
                self.api_key = os.environ["OPENAI_API_KEY"]
            except KeyError:
                raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable.")
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            timeout=300,
            **_http_client_kwargs(),
        )

    async def generate_async(
        self,
        input: str | list[dict[str, str]],
        max_tokens: int = 8192,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        target_keywords: list[str] = [],
        max_retries: int = 3,
    ) -> dict[str, str]:
        """
        Generate a response using OpenAI.

        Returns:
            Dict with keys: content, reasoning_content
        """

        for attempt in range(max_retries):
            response = await self._make_api_call_with_rate_limit_handling(
                input=input,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )

            self._log_usage(response)

            content = response.output_text
            reasoning_content = self._get_reasoning_summary_from_response(response)

            has_target_keywords = all(k.lower() in content.lower() for k in target_keywords)
            if has_target_keywords:
                break
            else:
                logger.info(f"Response missing target keywords at attempt {attempt + 1}. Retrying...")
        else:
            logger.warning("Max retries reached. Returning last response without all target keywords.")

        return {"content": content, "reasoning_content": reasoning_content}

    async def _make_api_call_with_rate_limit_handling(
        self,
        input: str | list[dict[str, str]],
        max_tokens: int,
        reasoning_effort: str,
        max_rate_limit_retries: int = 10,
    ) -> Response:
        """
        Make an API call with automatic rate limit handling and exponential backoff.

        Args:
            input: Input text or messages
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort level
            max_rate_limit_retries: Maximum number of retries for rate limit errors

        Returns:
            OpenAI Response object
        """
        for retry_attempt in range(max_rate_limit_retries):
            try:
                # Try with reasoning first
                if self.model_name.startswith("gpt-5") or self.model_name.startswith("o"):
                    response = await self.client.responses.create(
                        model=self.model_name,
                        input=input,  # type: ignore
                        max_output_tokens=max_tokens,
                        reasoning=({"effort": reasoning_effort, "summary": "detailed"}),
                    )
                else:
                    response = await self.client.responses.create(
                        model=self.model_name,
                        input=input,  # type: ignore
                        max_output_tokens=max_tokens,
                    )
                return response

            except openai.RateLimitError as e:
                # Extract wait time from error message if available
                wait_time = self._extract_wait_time_from_error(str(e))

                if wait_time is None:
                    # Use exponential backoff if no wait time is provided
                    wait_time = min(2**retry_attempt, 60)  # Cap at 60 seconds

                if retry_attempt < max_rate_limit_retries - 1:
                    logger.warning(
                        f"Rate limit reached (attempt {retry_attempt + 1}/{max_rate_limit_retries}). "
                        f"Waiting {wait_time:.2f} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max rate limit retries ({max_rate_limit_retries}) exceeded.")
                    raise

    def _extract_wait_time_from_error(self, error_message: str) -> Optional[float]:
        """
        Extract wait time from OpenAI rate limit error message.

        Example error message:
        "Rate limit reached... Please try again in 847ms."
        "Rate limit reached... Please try again in 2.5s."

        Returns:
            Wait time in seconds, or None if not found
        """
        # Try to match milliseconds (e.g., "847ms")
        ms_match = re.search(r"try again in (\d+)ms", error_message)
        if ms_match:
            return float(ms_match.group(1)) / 1000.0

        # Try to match seconds (e.g., "2.5s")
        s_match = re.search(r"try again in ([\d.]+)s", error_message)
        if s_match:
            return float(s_match.group(1))

        return None

    def _get_reasoning_summary_from_response(self, response: Response) -> str:
        """Extract reasoning summary from OpenAI response."""
        reasoning_content = ""
        for item in response.output:
            if item.type == "reasoning":
                reasoning_content = " ".join([summary.text for summary in item.summary])
        return reasoning_content

    def _log_usage(self, response: Response) -> None:
        """Log token usage from OpenAI response."""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
        reasoning_tokens = 0
        if usage is not None:
            otd = getattr(usage, "output_tokens_details", None)
            if otd is not None:
                reasoning_tokens = getattr(otd, "reasoning_tokens", 0)
            else:
                reasoning_tokens = getattr(usage, "reasoning_tokens", 0)
        logger.info(
            (
                f"Token usage - Total: {total_tokens}, "
                f"Input: {input_tokens}, Output: {output_tokens}, "
                f"Reasoning: {reasoning_tokens}"
            )
        )
