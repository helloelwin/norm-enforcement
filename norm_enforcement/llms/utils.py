import re
from typing import Dict, Tuple

from norm_enforcement.llms.base import AsyncProvider
from norm_enforcement.llms.openai_provider import AsyncOpenAIProvider
from norm_enforcement.llms.vllm_provider import AsyncVLLMProvider


class ProviderRegistry:
    """Cache providers by (provider_name, model_name, vllm_port) to avoid duplicates.

    This is useful when multiple agents share the same model - they can share
    a single provider instance instead of creating duplicates.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, str, int], AsyncProvider] = {}

    def get_or_create(
        self,
        provider_name: str,
        model_name: str,
        vllm_port: int = 8000,
    ) -> AsyncProvider:
        """Get an existing provider or create a new one.

        Args:
            provider_name: Provider type ("openai" or "vllm").
            model_name: Model name/identifier.
            vllm_port: Port for vLLM server (only used for vllm provider).

        Returns:
            AsyncProvider instance (cached if previously created).
        """
        key = (provider_name.lower(), model_name, vllm_port)
        if key not in self._cache:
            self._cache[key] = simple_get_provider(provider_name, model_name, vllm_port)
        return self._cache[key]

    def clear(self):
        """Clear the provider cache."""
        self._cache.clear()


def simple_get_provider(
    provider_name: str,
    model_name: str,
    vllm_port: int = 8000,
) -> AsyncProvider:
    """Get provider by name."""
    vllm_server_url = f"http://localhost:{vllm_port}/v1/chat/completions"
    if provider_name.lower() == "openai":
        return AsyncOpenAIProvider(model_name=model_name, reasoning_effort="low")
    elif provider_name.lower() == "vllm":
        return AsyncVLLMProvider(model_name=model_name, server_url=vllm_server_url)
    else:
        raise ValueError(f"Unsupported provider name: {provider_name}")


def parse_xml_tag(text: str, tag: str) -> str:
    """Parse the content of a specific XML tag from the given text.

    Args:
        text (str): The text containing XML tags.
        tag (str): The XML tag to parse (without angle brackets).

    Returns:
        str: The content inside the specified XML tag. If the tag is not found, returns an empty string.
    """

    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
