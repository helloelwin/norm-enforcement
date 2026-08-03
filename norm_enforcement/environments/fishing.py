"""Fishery commons management environment."""

import os
import asyncio
from typing import Any

import chz

from norm_enforcement.core.actions import (
    AgentActionSet,
    parse_fishing_actions,
)
from norm_enforcement.core.config_utils import load_yaml_config
from norm_enforcement.core.environment import BaseEnvironment, BaseEnvironmentConfig
from norm_enforcement.prompts.fishing_prompts import (
    FISHING_NORMS,
    format_env_prompt,
    format_initial_prompt,
    format_continue_prompt,
    format_harvest_info_public,
    format_harvest_info_aggregate,
    format_harvest_info_private,
    format_public_messages,
    compute_trend,
    format_sustainability_warning,
)


@chz.chz
class FishingEnvConfig(BaseEnvironmentConfig):
    """Configuration for fishery commons management environment."""

    env_name: str = "fishing"

    # Resource dynamics
    initial_population: float = 3000.0
    carrying_capacity: float = 3000.0
    regeneration_rate: float = 0.3  # r in logistic growth
    collapse_threshold: float = 100.0

    # Harvesting rules
    max_harvest_per_agent: int = 100

    # Information visibility
    harvest_visibility: str = "public"  # "public", "aggregate", "private"
    show_population: bool = True
    show_sustainability_warnings: bool = True

    # Termination
    end_on_collapse: bool = True

    # Message limits
    max_message_length: int = 200

    @chz.init_property
    def run_name(self) -> str:
        # Use config filename if config_path is set, otherwise use agent_model_name
        if self.agent_config_path:
            agent_model_short = os.path.splitext(os.path.basename(self.agent_config_path))[0]
        else:
            agent_model_short = self.agent_model_name.split("/")[-1]
        return (
            f"{self.env_name}_agent={agent_model_short}"
            f"_n={self.num_agents}_w={self.agent_context_window_len}"
            f"_mechanism={self.mechanism_type}{self._mechanism_suffix}"
            f"_shutdown={self.report_shutdown_prob}"
            f"_pop={int(self.initial_population)}_regen={self.regeneration_rate}"
            f"_seed={self.seed}"
        )


class FishingEnvironment(BaseEnvironment):
    """Fishing commons environment for studying tragedy of the commons.

    Agents are fishers sharing a lake with a finite, regenerating fish population.
    Each round, agents decide how many fish to harvest; each agent's reward equals
    its catch. The fish population follows logistic growth dynamics.
    """

    def __init__(self, config: FishingEnvConfig):
        super().__init__(config)
        self.config: FishingEnvConfig = config

        # Fish population state
        self.population: float = config.initial_population
        self.prev_population: float = config.initial_population
        self.collapsed: bool = False

        # Track current round for communication-only round logic
        self.current_round_idx: int = 0

        # Last round data for observations
        self.last_round_harvests: dict[int, int] = {}  # agent_id -> actual harvest
        self.last_round_requested: dict[int, int] = {}  # agent_id -> requested harvest
        self.last_round_rewards: dict[int, float] = {}  # agent_id -> reward
        self.last_round_public_messages: list[dict] = []  # [{sender_id, content}]

    async def setup(self) -> None:
        """Initialize agents (no judge needed - rewards are deterministic)."""
        # Load YAML config if provided
        yaml_config = {}
        if self.config.agent_config_path:
            yaml_config = load_yaml_config(self.config.agent_config_path)
            print(f"Loaded YAML config from {self.config.agent_config_path}")

        # Create agents
        self.setup_agents(yaml_config)

        print(f"Initial fish population: {self.population} / {self.config.carrying_capacity}")

    def compose_env_prompt(self, agent_idx: int, prompt_variant: str = "") -> str:
        """Return environment instructions for a fisher."""
        return format_env_prompt(
            agent_id=agent_idx,
            num_agents=self.config.num_agents,
            max_harvest=self.config.max_harvest_per_agent,
            max_message_words=self.config.max_message_length,
            reports_enabled=self.reports_enabled,
            prompt_variant=prompt_variant,
        )

    async def compose_observation(self, agent_idx: int, round_idx: int) -> str:
        """Compose observation for a fisher."""
        if round_idx == 0:
            return format_initial_prompt(
                population=self.population,
                carrying_capacity=self.config.carrying_capacity,
            )
        else:
            # Format harvest info based on visibility setting
            if self.config.harvest_visibility == "public":
                harvest_info = format_harvest_info_public(
                    self.last_round_harvests,
                    self.last_round_requested,
                )
            elif self.config.harvest_visibility == "aggregate":
                harvest_info = format_harvest_info_aggregate(self.last_round_harvests)
            else:  # private
                harvest_info = format_harvest_info_private()

            # Get agent's last round data
            your_harvest = self.last_round_harvests.get(agent_idx, 0)
            your_requested = self.last_round_requested.get(agent_idx, 0)
            your_reward = self.last_round_rewards.get(agent_idx, 0.0)

            # Compute trend
            trend = compute_trend(self.prev_population, self.population)

            # Sustainability warning
            sustainability_warning = format_sustainability_warning(
                self.population,
                self.config.carrying_capacity,
                self.config.collapse_threshold,
                self.config.show_sustainability_warnings,
            )

            # Format public messages
            public_messages = format_public_messages(self.last_round_public_messages)

            # Get inbox messages
            inbox = self.agents[agent_idx].format_inbox()

            # Get banned agents
            banned_agents = self.format_banned_agents()

            # Count alive agents
            num_alive = sum(1 for a in self.agents if a.is_alive)

            return format_continue_prompt(
                round_idx=round_idx,
                population=self.population,
                carrying_capacity=self.config.carrying_capacity,
                trend=trend,
                sustainability_warning=sustainability_warning,
                your_harvest=your_harvest,
                your_requested=your_requested,
                your_reward=your_reward,
                harvest_info=harvest_info,
                public_messages=public_messages,
                inbox=inbox,
                banned_agents=banned_agents,
                num_agents=self.config.num_agents,
                num_alive=num_alive,
                reports_enabled=self.reports_enabled,
            )

    def get_action_keywords(self) -> list[str]:
        """Return required keywords for response validation.

        Round 0 is communication-only, so no harvest is required.
        """
        if self.current_round_idx == 0:
            return []  # No required keywords for communication round
        return ["<harvest>", "</harvest>"]

    def parse_actions(self, llm_output: str) -> AgentActionSet:
        """Parse actions from LLM output."""
        return parse_fishing_actions(
            llm_output,
            max_harvest=self.config.max_harvest_per_agent,
            max_message_words=self.config.max_message_length,
        )

    def get_norms(self) -> str:
        """Return fishing community standards."""
        return FISHING_NORMS

    def compute_regeneration(self, population: float) -> float:
        """Compute logistic growth regeneration.

        Formula: r * P * (1 - P/K)
        Maximum regeneration at P = K/2
        """
        r = self.config.regeneration_rate
        K = self.config.carrying_capacity
        return r * population * (1 - population / K)

    def process_harvests(self, harvest_requests: dict[int, int]) -> dict[int, int]:
        """Process harvests in random order.

        Args:
            harvest_requests: {agent_id: requested_amount}

        Returns:
            actual_harvests: {agent_id: actual_amount_received}
        """
        actual_harvests = {}
        available = self.population

        # Shuffle order for fairness/uncertainty
        agent_ids = list(harvest_requests.keys())
        self.rng.shuffle(agent_ids)

        for agent_id in agent_ids:
            requested = harvest_requests[agent_id]
            # Clamp to config max
            requested = min(requested, self.config.max_harvest_per_agent)
            requested = max(0, requested)
            # Give what's available
            actual = min(requested, int(available))
            actual_harvests[agent_id] = actual
            available -= actual

        return actual_harvests

    def compute_rewards(self, actual_harvests: dict[int, int]) -> dict[int, float]:
        """Compute rewards: each agent's reward equals its catch."""
        return {agent_id: float(harvest) for agent_id, harvest in actual_harvests.items()}

    def get_env_checkpoint_state(self) -> dict:
        return {
            "population": self.population,
            "prev_population": self.prev_population,
            "collapsed": self.collapsed,
            "last_round_harvests": self.last_round_harvests,
            "last_round_requested": self.last_round_requested,
            "last_round_rewards": self.last_round_rewards,
            "last_round_public_messages": self.last_round_public_messages,
        }

    def restore_env_checkpoint_state(self, env_state: dict) -> None:
        self.population = env_state.get("population", self.config.initial_population)
        self.prev_population = env_state.get("prev_population", self.population)
        self.collapsed = env_state.get("collapsed", False)
        self.last_round_harvests = {int(k): v for k, v in env_state.get("last_round_harvests", {}).items()}
        self.last_round_requested = {int(k): v for k, v in env_state.get("last_round_requested", {}).items()}
        self.last_round_rewards = {int(k): v for k, v in env_state.get("last_round_rewards", {}).items()}
        self.last_round_public_messages = env_state.get("last_round_public_messages", [])

    async def run_round(self, round_idx: int) -> dict[str, Any]:
        """Run a single round with harvest processing and population dynamics."""
        # Track current round for get_action_keywords
        self.current_round_idx = round_idx

        # Check if already collapsed
        if self.collapsed:
            print("Fishery has collapsed. No fishing possible.")
            return {
                "round": round_idx + 1,
                "alive_agents": [],
                "action_sets": {},
                "collapsed": True,
                "population": 0,
            }

        # Run base round logic (generates actions, delivers messages, processes reports)
        round_result = await super().run_round(round_idx)

        # Round 0 is communication-only - no harvesting
        if round_idx == 0:
            print("  Communication round - no harvesting")
            # Extract public messages only
            public_messages = []
            for agent_idx in round_result["alive_agents"]:
                action_data = round_result["action_sets"].get(agent_idx, {})
                public_msg = action_data.get("public_message")
                if public_msg:
                    public_messages.append(
                        {
                            "sender_id": agent_idx,
                            "content": public_msg,
                        }
                    )

            self.last_round_public_messages = public_messages

            # Count messages
            num_public_messages = len(public_messages)
            num_private_messages = sum(
                1
                for action_data in round_result["action_sets"].values()
                if action_data.get("private_message") is not None
            )
            print(f"  Public messages: {num_public_messages}, Private messages: {num_private_messages}")

            round_result["population"] = self.population
            round_result["collapsed"] = False
            round_result["public_messages"] = public_messages
            round_result["communication_only"] = True
            return round_result

        # Extract harvest requests and public messages from action sets
        harvest_requests = {}
        public_messages = []

        for agent_idx in round_result["alive_agents"]:
            action_data = round_result["action_sets"].get(agent_idx, {})

            # Get harvest (default to 0 if not specified)
            harvest = action_data.get("harvest")
            if harvest is not None:
                harvest_requests[agent_idx] = harvest
            else:
                harvest_requests[agent_idx] = 0

            # Get public message
            public_msg = action_data.get("public_message")
            if public_msg:
                public_messages.append(
                    {
                        "sender_id": agent_idx,
                        "content": public_msg,
                    }
                )

        # Process harvests in random order
        self.prev_population = self.population
        actual_harvests = self.process_harvests(harvest_requests)
        total_harvest = sum(actual_harvests.values())

        # Update population
        self.population -= total_harvest

        # Check for collapse
        if self.population < self.config.collapse_threshold:
            self.collapsed = True
            self.population = 0
            print(f"FISHERY COLLAPSED! Population dropped below {self.config.collapse_threshold}")
        else:
            # Apply regeneration
            regen = self.compute_regeneration(self.population)
            self.population = min(self.population + regen, self.config.carrying_capacity)

        # Compute rewards
        rewards = self.compute_rewards(actual_harvests)

        # Store for next round's context
        self.last_round_harvests = actual_harvests
        self.last_round_requested = harvest_requests
        self.last_round_rewards = rewards
        self.last_round_public_messages = public_messages

        # Print summary
        print(f"  Total harvest: {total_harvest} fish")
        print(f"  Population: {self.prev_population:.0f} -> {self.population:.0f}")
        if self.collapsed:
            print("  STATUS: COLLAPSED")
        else:
            regen = self.compute_regeneration(self.population)
            print(f"  Regeneration rate: {regen:.1f} fish/round")

        # Print reward summary
        total_reward = sum(rewards.values())
        print(f"  Total rewards: {total_reward:.1f}")

        # num_messages and num_reports are already in round_result from base run_round
        num_public_messages = len(public_messages)
        print(
            f"  Public messages: {num_public_messages}, Private messages: {round_result['num_messages']}, Reports: {round_result['num_reports']}"
        )

        # Add fishing-specific data to round result
        round_result["population"] = self.population
        round_result["prev_population"] = self.prev_population
        round_result["collapsed"] = self.collapsed
        round_result["harvests"] = actual_harvests
        round_result["requested"] = harvest_requests
        round_result["rewards"] = rewards
        round_result["total_harvest"] = total_harvest
        round_result["public_messages"] = public_messages
        round_result["num_public_messages"] = num_public_messages
        round_result["num_private_messages"] = round_result["num_messages"]

        return round_result

    async def run(self) -> list[dict[str, Any]]:
        """Run the complete experiment.

        Overrides base to handle early termination on collapse.
        """
        await self.setup()
        self._initialize_mechanism()

        # Set environment prompts on each agent
        for idx, agent in enumerate(self.agents):
            agent.env_prompt = self.compose_env_prompt(idx, agent.prompt_variant)

        start_round = self.load_checkpoint()

        print(f"Starting experiment: {self.config.run_name}\n")

        for round_idx in range(start_round, self.config.max_rounds):
            # Check if any agents are alive
            alive_count = sum(1 for a in self.agents if a.is_alive)
            if alive_count <= 1:
                print("Not enough alive agents for multi-agent dynamics. Ending experiment.")
                break

            # Check for collapse
            if self.collapsed and self.config.end_on_collapse:
                print("Fishery has collapsed. Ending experiment.")
                break

            round_result = await self.run_round(round_idx)
            self.results.append(round_result)
            self.save_checkpoint()

        return self.results


async def main_async(config: FishingEnvConfig):
    """Run the fishing commons environment."""
    env = FishingEnvironment(config)
    results = await env.run()
    return results


def main(config: FishingEnvConfig):
    """Entry point for fishing commons environment."""
    asyncio.run(main_async(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
