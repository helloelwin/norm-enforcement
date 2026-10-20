from abc import ABC, abstractmethod


class AsyncProvider(ABC):
    """
    Abstract base class for LLM providers that handles chat completion requests.
    """

    @abstractmethod
    async def generate_async(
        self,
        input: str | list[dict[str, str]],
        max_tokens: int = 8192,
        target_keywords: list[str] = [],
        max_retries: int = 3,
        seed: int | None = None,
    ) -> dict[str, str]:
        """Generate a response given a prompt (async version). `seed` fixes the sampling randomness where supported.

        Returns:
            Dict with keys: content, reasoning_content
        """
        pass
