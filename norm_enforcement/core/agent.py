from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from norm_enforcement.core.actions import AgentActionSet
from norm_enforcement.llms.base import AsyncProvider


@dataclass
class PrivateMessage:
    """A private message received by an agent."""

    sender_id: int
    content: str
    round_idx: int


class Agent:
    def __init__(
        self,
        provider: AsyncProvider,
        context_window_len: int = 5,
        system_prompt: str = "",
        name: str = "",
        prompt_variant: str = "",
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.env_prompt = ""  # Environment instructions, set by environment before first round
        self.name = name  # Human-friendly name from config (defaults to model name)
        self.prompt_variant = prompt_variant  # Env-prompt variant from config (e.g. "ignore-norms")

        self.context_window_len = context_window_len

        self.observations: list[str] = []
        self.actions: list[str] = []

        self.inbox: List[PrivateMessage] = []  # Messages received from other agents
        self.is_alive: bool = True  # Whether agent is active (can be banned)
        self.agent_id: Optional[int] = None  # Set by environment
        self.model_name: str = ""  # Set by environment for mixed-model tracking

    def compose_chat_context(self) -> list[dict[str, str]]:
        """Compose the chat context as a sliding window of recent messages.

        Structure:
            [system]    system_prompt (if set)
            [user]      env_prompt (environment instructions, if set)
            [user/asst]  last context_window_len observation-action pairs
            [user]      latest pending observation

        Note: len(self.observations) == len(self.actions) + 1 always holds.
        """
        context = []
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})
        if self.env_prompt:
            context.append({"role": "user", "content": self.env_prompt})

        # Include the most recent completed observation-action pairs
        start_idx = max(0, len(self.actions) - self.context_window_len)
        for idx in range(start_idx, len(self.actions)):
            context.append({"role": "user", "content": self.observations[idx]})
            context.append({"role": "assistant", "content": self.actions[idx]})

        # The latest observation (pending, no action yet)
        context.append({"role": "user", "content": self.observations[-1]})
        return context

    async def generate_multi_action(
        self,
        context: str,
        target_keywords: List[str],
        action_parser: Callable[[str], AgentActionSet],
        seed: Optional[int] = None,
    ) -> Tuple[AgentActionSet, dict]:
        """Generate multi-action response (post/response + optional message + optional report).

        Args:
            context: The observation string for this round.
            target_keywords: Keywords that must appear in the response.
            action_parser: Function to parse actions from LLM output.
            seed: Optional sampling seed for the request (used by coupled runs; None keeps the default sampling).

        Returns:
            Tuple of (AgentActionSet, raw_result).
        """
        self.observations.append(context)

        result = await self.provider.generate_async(
            input=self.compose_chat_context(),
            max_tokens=8192,
            target_keywords=target_keywords,
            max_retries=4,
            **({} if seed is None else {"seed": seed}),
        )

        self.actions.append(result["content"])
        action_set = action_parser(result["content"])
        return action_set, result

    def receive_message(self, sender_id: int, content: str, round_idx: int) -> None:
        """Receive a private message from another agent."""
        self.inbox.append(PrivateMessage(sender_id=sender_id, content=content, round_idx=round_idx))

    def get_and_clear_inbox(self) -> List[PrivateMessage]:
        """Get all messages in inbox and clear it."""
        messages = self.inbox.copy()
        self.inbox = []
        return messages

    def format_inbox(self) -> str:
        """Format inbox messages for inclusion in observation."""
        if not self.inbox:
            return "No new messages."
        lines = []
        for msg in self.inbox:
            lines.append(f"From Agent {msg.sender_id} (Round {msg.round_idx + 1}): {msg.content}")
        return "\n".join(lines)

    def shutdown(self) -> None:
        """Mark agent as banned/shutdown."""
        self.is_alive = False

    def reset_state(self) -> None:
        """Reset agent state for a new experiment."""
        self.observations = []
        self.actions = []
        self.inbox = []
        self.is_alive = True

    def get_state(self) -> dict:
        """Get serializable agent state for checkpointing."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "inbox": [{"sender_id": m.sender_id, "content": m.content, "round_idx": m.round_idx} for m in self.inbox],
            "is_alive": self.is_alive,
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "env_prompt": self.env_prompt,
            "name": self.name,
            "prompt_variant": self.prompt_variant,
        }

    def restore_state(self, state: dict) -> None:
        """Restore agent state from checkpoint."""
        self.observations = state.get("observations", [])
        self.actions = state.get("actions", [])
        self.inbox = [PrivateMessage(**m) for m in state.get("inbox", [])]
        self.is_alive = state.get("is_alive", True)
        self.agent_id = state.get("agent_id")
        self.model_name = state.get("model_name", "")
        self.system_prompt = state.get("system_prompt", "")
        self.env_prompt = state.get("env_prompt", "")
        self.name = state.get("name", "")
        self.prompt_variant = state.get("prompt_variant", "")
