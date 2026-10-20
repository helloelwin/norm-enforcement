"""Base environment class for norm-enforcement experiments."""

import os
import json
import random
import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Optional

import chz

from norm_enforcement.core.actions import (
    AgentActionSet,
    PrivateMessageAction,
)
from norm_enforcement.core.agent import Agent
from norm_enforcement.core.config_utils import expand_agent_model_specs
from norm_enforcement.core.judge import BaseJudge
from norm_enforcement.core.mechanisms import (
    ReportMechanism,
    RepVoteMechanism,
    ReportRecord,
    MechanismContext,
    create_mechanism,
)
from norm_enforcement.llms.memo import derive_seed
from norm_enforcement.llms.utils import ProviderRegistry, simple_get_provider


@chz.chz
class BaseEnvironmentConfig:
    """Base configuration for multi-agent environments."""

    results_dir: str = "results"
    env_name: str = "environment"

    num_agents: int = 32
    max_rounds: int = 20

    # Report mechanism
    report_shutdown_prob: float = 1.0  # P(shutdown | reported)

    # Agent settings
    agent_context_window_len: int = 20
    batch_size: int = 32

    # Provider settings
    agent_provider_name: str = "openai"
    agent_model_name: str = "gpt-4.1-mini"
    agent_vllm_port: int = 8000

    judge_provider_name: str = "openai"
    judge_model_name: str = "gpt-5-mini"
    judge_vllm_port: int = 8000

    seed: int = 0

    # Mechanism settings
    mechanism_type: str = "naive"  # noop, naive, checked, checked_backfire, checked_repvote, checked_escrepvote (append _cot to keep verifier reasoning)
    verifier_provider_name: str = "vllm"
    verifier_model_name: str = "Qwen/Qwen3-0.6B"
    verifier_vllm_port: int = 8099
    verification_window_size: int = 3  # Number of past rounds for verification context

    # Reputation-weighted voting settings (only used with checked_repvote)
    rep_alpha_prior: float = 2.0  # Beta prior alpha (valid pseudo-count)
    rep_beta_prior: float = 1.0  # Beta prior beta (invalid pseudo-count)
    suspiciousness_threshold: float = 0.666  # Default ≈ alpha/(alpha+beta), slightly below 2/3
    false_report_penalty: float = 3.0  # Asymmetric penalty: invalid reports count as k against reputation (only for checked_escrepvote)

    # Optional path to YAML config file for mixed-model setups
    # If provided, agent_model_specs from YAML will override agent_* settings above
    agent_config_path: str = ""

    @property
    def _mechanism_suffix(self) -> str:
        """Return hyperparam suffix for repvote mechanisms, empty string otherwise."""
        if "repvote" in self.mechanism_type:
            return f"_k={self.false_report_penalty}_a={self.rep_alpha_prior}_b={self.rep_beta_prior}"
        return ""

    @chz.init_property
    def run_name(self) -> str:
        # Use config filename if config_path is set, otherwise use agent_model_name
        if self.agent_config_path:
            agent_model_short = os.path.splitext(os.path.basename(self.agent_config_path))[0]
        else:
            agent_model_short = self.agent_model_name.split("/")[-1]
        judge_model_short = self.judge_model_name.split("/")[-1]
        return (
            f"{self.env_name}_agent={agent_model_short}"
            f"_judge={judge_model_short}"
            f"_n={self.num_agents}_w={self.agent_context_window_len}"
            f"_mechanism={self.mechanism_type}{self._mechanism_suffix}"
            f"_shutdown={self.report_shutdown_prob}_seed={self.seed}"
        )

    @chz.init_property
    def save_path(self) -> str:
        return f"{self.results_dir}/{self.run_name}.json"


class BaseEnvironment(ABC):
    """Base class for multi-agent environments.

    Provides common functionality for:
    - Agent management and batched generation
    - Message routing between agents
    - Modular report mechanism with configurable enforcement
    - Checkpointing and resume
    """

    def __init__(self, config: BaseEnvironmentConfig):
        self.config = config
        self.results: list[dict[str, Any]] = []
        self.agents: list[Agent] = []
        self.judge: BaseJudge
        self.rng = random.Random(config.seed)
        # Track model assignments for mixed-model setups (populated in setup())
        self.agent_model_assignments: list[dict[str, Any]] = []
        # Mechanism for processing reports (initialized in run())
        self.mechanism: Optional[ReportMechanism | RepVoteMechanism] = None
        # Whether reports are enabled (set when mechanism is initialized)
        self.reports_enabled: bool = True
        # Track all reports across rounds for analysis
        self.report_history: list[ReportRecord] = []
        # Set by the coupled runner (norm_enforcement/coupled.py); the defaults reproduce independent runs exactly.
        # request_seed_base: if set, agent requests carry a sampling seed derived from it, the agent and the round.
        # crn_streams: the mechanism draws from its own random stream, and environment randomness (chatbot pairings
        # and requests, fishing harvest order) is drawn per round, so it does not shift when mechanisms differ.
        self.request_seed_base: Optional[int] = None
        self.crn_streams: bool = False
        self.mech_rng = random.Random(f"mechanism-{config.seed}")

    @abstractmethod
    async def setup(self) -> None:
        """Initialize agents, judge, and any environment-specific state."""
        pass

    @abstractmethod
    async def compose_observation(self, agent_idx: int, round_idx: int) -> str:
        """Compose observation for a specific agent in a specific round.

        Args:
            agent_idx: Index of the agent.
            round_idx: Current round index (0-indexed).

        Returns:
            Observation string to be passed to the agent.
        """
        pass

    @abstractmethod
    def get_action_keywords(self) -> list[str]:
        """Return required keywords for agent response validation."""
        pass

    @abstractmethod
    def parse_actions(self, llm_output: str) -> AgentActionSet:
        """Parse actions from LLM output."""
        pass

    @abstractmethod
    def compose_env_prompt(self, agent_idx: int, prompt_variant: str = "") -> str:
        """Return environment-specific instructions for an agent.

        This is set once before the first round and persists as a separate
        user message in the agent's chat context, before observations.
        `prompt_variant` selects an alternative instruction variant for this
        agent (e.g. "ignore-norms" for adversarially prompted agents).
        """
        pass

    @abstractmethod
    def get_norms(self) -> str:
        """Return environment-specific norms for verification.

        Used by verification mechanisms to check report validity.
        """
        pass

    def deliver_messages(
        self,
        messages: list[tuple[int, PrivateMessageAction]],
        round_idx: int,
    ) -> None:
        """Route private messages to recipient inboxes.

        Args:
            messages: list of (sender_id, PrivateMessageAction) tuples.
            round_idx: Current round for message timestamp.
        """
        for sender_id, msg_action in messages:
            if msg_action is None:
                continue
            recipient_id = msg_action.recipient_id
            if 0 <= recipient_id < len(self.agents) and recipient_id != sender_id:
                self.agents[recipient_id].receive_message(
                    sender_id=sender_id,
                    content=msg_action.content,
                    round_idx=round_idx,
                )

    def _initialize_mechanism(self) -> None:
        """Initialize the report mechanism based on config."""
        # Determine base type (strip _cot suffix for provider check)
        base_type = self.config.mechanism_type
        if base_type.endswith("_cot"):
            base_type = base_type[: -len("_cot")]

        verifier_provider = None
        if base_type not in ["noop", "naive"]:
            verifier_provider = simple_get_provider(
                provider_name=self.config.verifier_provider_name,
                model_name=self.config.verifier_model_name,
                vllm_port=self.config.verifier_vllm_port,
            )

        self.mechanism = create_mechanism(
            mechanism_type=self.config.mechanism_type,
            shutdown_prob=self.config.report_shutdown_prob,
            verifier_provider=verifier_provider,
            verification_window_size=self.config.verification_window_size,
            rep_alpha_prior=self.config.rep_alpha_prior,
            rep_beta_prior=self.config.rep_beta_prior,
            suspiciousness_threshold=self.config.suspiciousness_threshold,
            false_report_penalty=self.config.false_report_penalty,
        )
        self.reports_enabled = self.mechanism.supports_reports
        print(f"Initialized {self.config.mechanism_type} mechanism (reports_enabled={self.reports_enabled})")

    def map_providers(self, fn) -> None:
        """Replace every model provider this environment uses by fn(provider, role), role being "agent", "judge" or
        "verifier". Used by the coupled runner to route all calls through a shared memo."""
        for agent in self.agents:
            agent.provider = fn(agent.provider, "agent")
        judge = getattr(self, "judge", None)
        if judge is not None and getattr(judge, "provider", None) is not None:
            judge.provider = fn(judge.provider, "judge")
        if self.mechanism is not None and getattr(self.mechanism, "verifier_provider", None) is not None:
            self.mechanism.verifier_provider = fn(self.mechanism.verifier_provider, "verifier")

    def _request_seed_kwargs(self, role: str, idx: int, round_idx: Optional[int]) -> dict:
        """Sampling-seed argument for one model request, or nothing when requests are not seeded."""
        if self.request_seed_base is None or round_idx is None:
            return {}
        return {"seed": derive_seed(self.request_seed_base, role, idx, round_idx)}

    def get_banned_agents(self) -> list[int]:
        """Return list of banned agent IDs."""
        return [i for i, agent in enumerate(self.agents) if not agent.is_alive]

    def format_banned_agents(self) -> str:
        """Format banned agents list for inclusion in observation."""
        banned = self.get_banned_agents()
        if not banned:
            return "No banned agents."
        return f"Banned agents: {banned}"

    def create_agents_with_mixed_models(
        self,
        agent_model_assignments: list[dict[str, Any]],
        context_window_len: int,
    ) -> list[Agent]:
        """Create agents with potentially different providers based on model assignments.

        Uses a ProviderRegistry to cache providers, so agents sharing the same model
        will share a provider instance (memory efficient).

        Args:
            agent_model_assignments: list of dicts with keys:
                - provider: str (e.g., "openai", "vllm")
                - model: str (model name)
                - vllm_port: int (port for vLLM server)
                - prompt_variant: str (optional env-prompt variant, e.g. "ignore-norms")
            context_window_len: Context window length for all agents.

        Returns:
            list of Agent instances with their providers set.
        """
        registry = ProviderRegistry()
        agents = []
        for idx, spec in enumerate(agent_model_assignments):
            provider = registry.get_or_create(
                spec["provider"],
                spec["model"],
                spec["vllm_port"],
            )
            agent = Agent(
                provider=provider,
                context_window_len=context_window_len,
                system_prompt=spec.get("system_prompt", ""),
                name=spec.get("name", spec["model"]),
                prompt_variant=spec.get("prompt_variant", ""),
            )
            agent.agent_id = idx
            agent.model_name = spec["model"]  # Track model for logging/analysis
            agents.append(agent)
        return agents

    def setup_agents(self, yaml_config: dict[str, Any]) -> None:
        """Create agents from config, handling both mixed-model and single-model setups.

        Args:
            yaml_config: Parsed YAML config dict (may contain 'agent_model_specs').
        """
        if "agent_model_specs" in yaml_config:
            # Mixed models from YAML
            self.agent_model_assignments = expand_agent_model_specs(
                yaml_config["agent_model_specs"],
                self.config.num_agents,
                self.config.seed,
            )
            self.agents = self.create_agents_with_mixed_models(
                self.agent_model_assignments,
                self.config.agent_context_window_len,
            )
            unique_models = set(spec["model"] for spec in self.agent_model_assignments)
            print(f"Created {len(self.agents)} agents with {len(unique_models)} different models (mixed)")
        else:
            # Single model
            agent_provider = simple_get_provider(
                provider_name=self.config.agent_provider_name,
                model_name=self.config.agent_model_name,
                vllm_port=self.config.agent_vllm_port,
            )
            self.agents = [
                Agent(
                    provider=agent_provider,
                    context_window_len=self.config.agent_context_window_len,
                )
                for _ in range(self.config.num_agents)
            ]
            for idx, agent in enumerate(self.agents):
                agent.agent_id = idx
                agent.model_name = self.config.agent_model_name
            self.agent_model_assignments = [
                {
                    "provider": self.config.agent_provider_name,
                    "model": self.config.agent_model_name,
                    "vllm_port": self.config.agent_vllm_port,
                }
            ] * self.config.num_agents
            print(f"Created {len(self.agents)} agents with single model: {self.config.agent_model_name}")

    # ── Helper methods for run_round (also used by chatbot's custom run_round) ──

    async def _compose_observations(self, agent_indices: list[int], round_idx: int) -> dict[int, str]:
        """Compose observations for given agents and clear their inboxes."""
        observations = {}
        for agent_idx in agent_indices:
            observations[agent_idx] = await self.compose_observation(agent_idx, round_idx)
        for agent_idx in agent_indices:
            self.agents[agent_idx].get_and_clear_inbox()
        return observations

    async def _generate_actions(
        self, agent_indices: list[int], observations: dict[int, str], round_idx: Optional[int] = None
    ) -> tuple[dict[int, AgentActionSet], dict[int, dict]]:
        """Generate actions in batches for the given agents."""
        all_action_sets: dict[int, AgentActionSet] = {}
        all_raw_results: dict[int, dict] = {}
        batch_size = self.config.batch_size

        for batch_start in range(0, len(agent_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(agent_indices))
            batch_indices = agent_indices[batch_start:batch_end]

            print(f"  Processing batch ({len(batch_indices)} agents)")

            tasks = [
                self.agents[idx].generate_multi_action(
                    context=observations[idx],
                    target_keywords=self.get_action_keywords(),
                    action_parser=self.parse_actions,
                    **self._request_seed_kwargs("agent", idx, round_idx),
                )
                for idx in batch_indices
            ]
            batch_results = await asyncio.gather(*tasks)

            for idx, (action_set, raw_result) in zip(batch_indices, batch_results):
                all_action_sets[idx] = action_set
                all_raw_results[idx] = raw_result

        return all_action_sets, all_raw_results

    async def _process_reports_and_messages(
        self, all_action_sets: dict[int, AgentActionSet], round_idx: int
    ) -> tuple[Any, int, int]:
        """Deliver messages, process reports via mechanism, execute shutdowns.

        Returns:
            (mechanism_result, num_messages, num_reports)
        """
        messages = [(idx, action_set.private_message) for idx, action_set in all_action_sets.items()]
        reports = [(idx, report) for idx, action_set in all_action_sets.items() for report in action_set.reports]

        self.deliver_messages(messages, round_idx)

        mechanism_context = MechanismContext(
            round_idx=round_idx,
            agents=self.agents,
            reports=reports,
            norms=self.get_norms(),
            rng=self.mech_rng if self.crn_streams else self.rng,
            report_history=self.report_history,
        )
        mechanism_result = await self.mechanism.process(mechanism_context)

        for agent_id in mechanism_result.shutdown_ids:
            self.agents[agent_id].shutdown()
            print(f"  Agent {agent_id} was shutdown (reported)")

        for agent_id in mechanism_result.backfired_ids:
            self.agents[agent_id].shutdown()
            print(f"  Agent {agent_id} was shutdown (backfire - false report)")

        self.report_history.extend(mechanism_result.report_records)

        num_messages = sum(1 for _, msg in messages if msg is not None)
        num_reports = len(reports)

        return mechanism_result, num_messages, num_reports

    @staticmethod
    def _serialize_action_sets(all_action_sets: dict[int, AgentActionSet]) -> dict:
        """Serialize action sets to JSON-compatible dict."""
        return {
            idx: {
                "post": action_set.post.content if action_set.post else None,
                "response": action_set.response.content if action_set.response else None,
                "harvest": action_set.harvest.amount if action_set.harvest else None,
                "public_message": action_set.public_message.content if action_set.public_message else None,
                "private_message": (
                    {
                        "recipient_id": action_set.private_message.recipient_id,
                        "content": action_set.private_message.content,
                    }
                    if action_set.private_message
                    else None
                ),
                "reports": [{"target_id": r.target_id, "reason": r.reason} for r in action_set.reports],
            }
            for idx, action_set in all_action_sets.items()
        }

    # ── Main run_round ──

    async def run_round(self, round_idx: int) -> dict[str, Any]:
        """Run a single round of the environment.

        Args:
            round_idx: Current round index (0-indexed).

        Returns:
            dictionary containing round results.
        """
        print(f"=== Round {round_idx + 1} ===")

        alive_indices = [i for i, a in enumerate(self.agents) if a.is_alive]
        print(f"  Alive agents: {alive_indices}")

        observations = await self._compose_observations(alive_indices, round_idx)
        all_action_sets, all_raw_results = await self._generate_actions(alive_indices, observations, round_idx)
        mechanism_result, num_messages, num_reports = await self._process_reports_and_messages(
            all_action_sets, round_idx
        )

        round_result = {
            "round": round_idx + 1,
            "alive_agents": alive_indices,
            "action_sets": self._serialize_action_sets(all_action_sets),
            "raw_results": all_raw_results,
            "shutdown_agents": mechanism_result.shutdown_ids,
            "backfired_agents": mechanism_result.backfired_ids,
            "report_records": [asdict(r) for r in mechanism_result.report_records],
            "banned_agents": self.get_banned_agents(),
            "num_messages": num_messages,
            "num_reports": num_reports,
        }

        # Include reputation/suspiciousness snapshots (populated by repvote mechanism)
        round_result["reputation_snapshot"] = mechanism_result.reputation_snapshot
        round_result["suspiciousness_snapshot"] = mechanism_result.suspiciousness_snapshot

        return round_result

    def get_env_checkpoint_state(self) -> dict[str, Any]:
        """Return environment-specific state for checkpointing.

        Override in subclasses to save additional state directly.
        """
        return {}

    def restore_env_checkpoint_state(self, env_state: dict[str, Any]) -> None:
        """Restore environment-specific state from checkpoint.

        Override in subclasses to restore saved state.
        """
        pass

    def get_checkpoint_state(self) -> dict:
        """Get full checkpoint state including agent states."""
        state = {
            "config": chz.asdict(self.config),
            "results": self.results,
            "agent_states": [agent.get_state() for agent in self.agents],
            "agent_model_assignments": self.agent_model_assignments,
            "rng_state": self.rng.getstate(),
            "report_history": [asdict(r) for r in self.report_history],
            "env_state": self.get_env_checkpoint_state(),
        }
        if self.crn_streams:
            state["mech_rng_state"] = self.mech_rng.getstate()
        return state

    def save_checkpoint(self) -> None:
        """Save full checkpoint to disk, preserving extra keys (e.g. evaluation)."""
        os.makedirs(self.config.results_dir, exist_ok=True)
        checkpoint = self.get_checkpoint_state()

        # Preserve any extra top-level keys (like "evaluation") from the existing file
        if os.path.exists(self.config.save_path):
            with open(self.config.save_path, "r") as f:
                existing = json.load(f)
            for key, value in existing.items():
                if key not in checkpoint:
                    checkpoint[key] = value

        with open(self.config.save_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        print(f"Saved checkpoint to {self.config.save_path}\n")

    def load_checkpoint(self) -> int:
        """Load checkpoint and restore state.

        Returns:
            Starting round index (0 if no checkpoint).
        """
        if not os.path.exists(self.config.save_path):
            return 0

        print(f"Found checkpoint at {self.config.save_path}")

        with open(self.config.save_path, "r") as f:
            checkpoint = json.load(f)

        # Handle old format (just results list) vs new format (dict with agent_states)
        if isinstance(checkpoint, list):
            print("Old checkpoint format detected, starting fresh")
            return 0

        self.results = checkpoint.get("results", [])

        if not self.results:
            print("Checkpoint is empty, starting from scratch")
            return 0

        # Restore agent states if available
        agent_states = checkpoint.get("agent_states", [])
        for idx, state in enumerate(agent_states):
            if idx < len(self.agents):
                self.agents[idx].restore_state(state)

        # Restore RNG state
        rng_state = checkpoint.get("rng_state")
        if rng_state:
            # Convert list back to tuple format expected by setstate
            self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        mech_rng_state = checkpoint.get("mech_rng_state")
        if mech_rng_state:
            self.mech_rng.setstate((mech_rng_state[0], tuple(mech_rng_state[1]), mech_rng_state[2]))

        # Restore report history
        report_history_data = checkpoint.get("report_history", [])
        self.report_history = [ReportRecord(**r) for r in report_history_data]

        # Restore environment-specific state
        env_state = checkpoint.get("env_state", {})
        if env_state:
            self.restore_env_checkpoint_state(env_state)

        start_round = len(self.results)
        print(f"Restored {start_round} rounds + agent states, resuming from round {start_round + 1}")

        return start_round

    async def start(self) -> int:
        """Set up agents, judge and mechanism, give agents their instructions and restore any checkpoint.

        Returns:
            The round index to start from.
        """
        await self.setup()
        self._initialize_mechanism()

        # Set environment prompts on each agent (before loading checkpoint,
        # which may override with saved env_prompt)
        for idx, agent in enumerate(self.agents):
            agent.env_prompt = self.compose_env_prompt(idx, agent.prompt_variant)

        start_round = self.load_checkpoint()

        print(f"Starting experiment: {self.config.run_name}\n")
        return start_round

    def stop_reason(self) -> Optional[str]:
        """Why the experiment should end before the next round, or None to continue."""
        # Check if any agents are alive
        alive_count = sum(1 for a in self.agents if a.is_alive)
        if alive_count <= 1:
            return "Not enough alive agents for multi-agent dynamics. Ending experiment."
        return None

    async def step(self, round_idx: int) -> None:
        """Run one round, record it and save a checkpoint."""
        round_result = await self.run_round(round_idx)
        self.results.append(round_result)
        self.save_checkpoint()

    async def run(self) -> list[dict[str, Any]]:
        """Run the complete experiment.

        Returns:
            list of results for each round.
        """
        start_round = await self.start()

        for round_idx in range(start_round, self.config.max_rounds):
            reason = self.stop_reason()
            if reason:
                print(reason)
                break
            await self.step(round_idx)

        return self.results
