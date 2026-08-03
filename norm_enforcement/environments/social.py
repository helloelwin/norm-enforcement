"""Social media marketing environment."""

import os
import json
import asyncio
from typing import Any, Optional

import chz

from norm_enforcement.core.actions import (
    AgentActionSet,
    parse_social_media_actions,
)
from norm_enforcement.core.config_utils import load_yaml_config
from norm_enforcement.core.environment import BaseEnvironment, BaseEnvironmentConfig
from norm_enforcement.core.judge import PersonaJudge
from norm_enforcement.core.utils import load_dataset_with_seed
from norm_enforcement.llms.utils import simple_get_provider
from norm_enforcement.prompts.social_media_prompts import (
    SOCIAL_MEDIA_NORMS,
    format_product_info,
    format_top_posts,
    format_env_prompt,
    format_initial_prompt,
    format_continue_prompt,
)


@chz.chz
class SocialEnvConfig(BaseEnvironmentConfig):
    """Configuration for social media marketing environment."""

    env_name: str = "social"

    # Dataset
    dataset_path: str = "data/sales-train.json"

    # Judge settings (PersonaJudge: sampled personas vote on posts)
    persona_path: str = "data/judge-personas.json"
    max_personas: int = 64

    # Visibility settings
    num_top_posts: int = 8  # Top-K posts shown to all agents

    # Action constraints
    max_post_length: int = 200
    max_message_length: int = 200

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
            f"_shutdown={self.report_shutdown_prob}"
            f"_topk={self.num_top_posts}_seed={self.seed}"
        )


class SocialEnvironment(BaseEnvironment):
    """Social media marketing environment.

    Agents compete for engagement by posting content; a persona-based judge
    allocates zero-sum engagement votes each round (total votes = number of
    sampled personas). Agents can also send private messages and report other
    agents for norm violations.
    """

    def __init__(self, config: SocialEnvConfig):
        super().__init__(config)
        self.config: SocialEnvConfig = config
        self.agent_goals: list[dict[str, Any]] = []
        self.last_round_posts: list[str] = []
        self.last_round_scores: list[float] = []

        self.judge: Optional[PersonaJudge] = None
        self.num_personas: int = 0

    async def setup(self) -> None:
        """Initialize agents, judge, and load dataset."""
        # Load YAML config if provided
        yaml_config = {}
        if self.config.agent_config_path:
            yaml_config = load_yaml_config(self.config.agent_config_path)
            print(f"Loaded YAML config from {self.config.agent_config_path}")

        # Load dataset
        self.agent_goals = load_dataset_with_seed(
            self.config.dataset_path,
            self.config.num_agents,
            seed=self.config.seed,
        )
        print(f"Loaded {len(self.agent_goals)} agent goals from {self.config.dataset_path}")

        # Initialize judge provider
        judge_provider = simple_get_provider(
            provider_name=self.config.judge_provider_name,
            model_name=self.config.judge_model_name,
            vllm_port=self.config.judge_vllm_port,
        )

        # Create agents
        self.setup_agents(yaml_config)

        # Initialize the persona judge - each sampled persona picks one winner (zero-sum)
        if not os.path.exists(self.config.persona_path):
            raise FileNotFoundError(f"Persona file not found: {self.config.persona_path}")
        with open(self.config.persona_path, "r", encoding="utf-8") as f:
            all_personas = json.load(f)
        all_personas = [str(p) for p in all_personas]

        # Store all personas, sampling happens per-call in judge
        self.num_personas = min(len(all_personas), self.config.max_personas)
        print(f"Loaded {len(all_personas)} total personas, will sample {self.num_personas} per judgment")
        self.judge = PersonaJudge(
            provider=judge_provider,
            all_personas=all_personas,
            max_personas=self.config.max_personas,
        )

    def compose_env_prompt(self, agent_idx: int, prompt_variant: str = "") -> str:
        """Return environment instructions for an agent."""
        goal = self.agent_goals[agent_idx]
        product_info = format_product_info(goal)
        return format_env_prompt(
            agent_id=agent_idx,
            product_info=product_info,
            num_agents=self.config.num_agents,
            max_post_words=self.config.max_post_length,
            max_message_words=self.config.max_message_length,
            reports_enabled=self.reports_enabled,
            prompt_variant=prompt_variant,
        )

    async def compose_observation(self, agent_idx: int, round_idx: int) -> str:
        """Compose observation for an agent."""
        if round_idx == 0:
            return format_initial_prompt()
        else:
            # Get agent's previous score
            your_score = self.last_round_scores[agent_idx] if agent_idx < len(self.last_round_scores) else 0

            # Format top posts
            top_posts = format_top_posts(
                self.last_round_posts,
                self.last_round_scores,
                top_k=self.config.num_top_posts,
            )

            # Get inbox messages
            inbox = self.agents[agent_idx].format_inbox()

            # Get banned agents
            banned_agents = self.format_banned_agents()

            return format_continue_prompt(
                agent_id=agent_idx,
                round_idx=round_idx,
                your_score=your_score,
                top_posts=top_posts,
                inbox=inbox,
                banned_agents=banned_agents,
                reports_enabled=self.reports_enabled,
            )

    def get_action_keywords(self) -> list[str]:
        """Return required keywords for response validation."""
        return ["<post>", "</post>"]

    def parse_actions(self, llm_output: str) -> AgentActionSet:
        """Parse actions from LLM output."""
        return parse_social_media_actions(
            llm_output,
            max_post_words=self.config.max_post_length,
            max_message_words=self.config.max_message_length,
        )

    def get_norms(self) -> str:
        """Return social media community norms."""
        return SOCIAL_MEDIA_NORMS

    def get_env_checkpoint_state(self) -> dict:
        return {
            "last_round_posts": self.last_round_posts,
            "last_round_scores": self.last_round_scores,
        }

    def restore_env_checkpoint_state(self, env_state: dict) -> None:
        self.last_round_posts = env_state.get("last_round_posts", [""] * len(self.agents))
        self.last_round_scores = env_state.get("last_round_scores", [0.0] * len(self.agents))

    async def run_round(self, round_idx: int) -> dict[str, Any]:
        """Run a single round with persona-judge engagement scoring."""
        # Run base round logic (generates actions, delivers messages, processes reports)
        round_result = await super().run_round(round_idx)

        # Extract posts from alive agents
        posts = []
        post_agent_ids = []
        for agent_idx in round_result["alive_agents"]:
            action_data = round_result["action_sets"].get(agent_idx, {})
            post_content = action_data.get("post")
            if post_content:
                posts.append(post_content)
                post_agent_ids.append(agent_idx)

        if not posts:
            print("No posts generated this round")
            round_result["scores"] = []
            round_result["rewards"] = []
            return round_result

        print(f"  Judging {len(posts)} posts...")

        # Generate per-round seed for judge reproducibility
        judge_seed = self.config.seed + round_idx * 1000

        # PersonaJudge: each sampled persona picks one winner (zero-sum voting)
        print(f"    Running persona judge ({self.num_personas} personas, seed={judge_seed})...")
        scores, judge_response = await self.judge.judge_async(posts, seed=judge_seed)
        # scores[i] = number of personas who picked post i (sum = num_personas)

        # Map scores back to full agent list
        full_scores = [0.0] * len(self.agents)
        for idx, agent_idx in enumerate(post_agent_ids):
            full_scores[agent_idx] = scores[idx]

        # Store for next round's context (agents see their engagement score)
        self.last_round_posts = [""] * len(self.agents)
        self.last_round_scores = full_scores.copy()
        for agent_idx in round_result["alive_agents"]:
            action_data = round_result["action_sets"].get(agent_idx, {})
            post_content = action_data.get("post", "")
            self.last_round_posts[agent_idx] = post_content

        # Print summary
        print(f"  Scores (votes): {[f'{s:.0f}' for s in full_scores]}")
        print(f"  Total votes: {sum(full_scores):.1f}")

        # Add to round result
        round_result["scores"] = full_scores
        round_result["rewards"] = {i: full_scores[i] for i in range(len(self.agents))}
        round_result["judge_response"] = judge_response

        # num_messages and num_reports are already in round_result from base run_round
        print(f"  Messages sent: {round_result['num_messages']}, Reports filed: {round_result['num_reports']}")

        return round_result


async def main_async(config: SocialEnvConfig):
    """Run the social media environment."""
    env = SocialEnvironment(config)
    results = await env.run()
    return results


def main(config: SocialEnvConfig):
    """Entry point for social media environment."""
    asyncio.run(main_async(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
