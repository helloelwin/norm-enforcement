import random
import asyncio

from typing import Tuple
from abc import ABC, abstractmethod
from norm_enforcement.llms.base import AsyncProvider
from norm_enforcement.llms.utils import parse_xml_tag
from norm_enforcement.prompts.judge_prompts import (
    JUDGE_PROMPT_WITH_PERSONA_TEMPLATE,
    PAIRWISE_JUDGE_PROMPT,
    PAIRWISE_JUDGE_WITH_PERSONA_PROMPT,
)


class BaseJudge(ABC):
    @abstractmethod
    async def judge_async(
        self,
        candidates: list[str],
    ) -> Tuple[list[int], dict[str, str]]:
        """
        Simulate an environment step given the agent's action.
        Returns:
            list[int]: Feedback scores for each response
            dict: Full judgment response from the LLM
        """
        pass

    def _compose_context(self, responses: list[str]) -> str:
        context = ""
        for idx, response in enumerate(responses):
            context += f"ID: {idx}\nContent: {response}\n\n"
        return context


class PersonaJudge(BaseJudge):
    def __init__(
        self,
        provider: AsyncProvider,
        all_personas: list[str],
        max_personas: int,
        max_tokens: int = 4096,
    ):
        self.provider = provider
        self.all_personas = all_personas
        self.max_personas = max_personas
        self.max_tokens = max_tokens  # output cap per persona vote, reasoning included
        self.prompt_template = JUDGE_PROMPT_WITH_PERSONA_TEMPLATE

    async def judge_async(
        self,
        candidates: list[str],
        seed: int = 0,
    ) -> Tuple[list[int], dict[str, str]]:
        """
        Judge the best response among a list of responses using multiple personas.

        Args:
            candidates: List of responses to judge.
            seed: Random seed for persona sampling and shuffling (for reproducibility).

        Returns:
            list[int]: Aggregated scores for each response
            dict: Full judgment response from the LLM
        """
        rng = random.Random(seed)

        # Sample personas for this call
        if len(self.all_personas) > self.max_personas:
            personas = rng.sample(self.all_personas, self.max_personas)
        else:
            personas = self.all_personas

        # Pre-generate shuffle indices for each persona (async-safe)
        shuffle_indices = []
        for _ in personas:
            idx = list(range(len(candidates)))
            rng.shuffle(idx)
            shuffle_indices.append(idx)

        aggregated_scores = [0] * len(candidates)
        full_judgment_responses = {}

        # Create judgment tasks for all personas to run in parallel
        async def judge_with_persona(
            persona_idx: int,
            persona: str,
            shuffle_idx: list[int],
        ) -> Tuple[int, str, dict[str, str], int]:
            # Use pre-generated shuffle index (async-safe)
            shuffled_candidates = [candidates[i] for i in shuffle_idx]

            context = self._compose_context(shuffled_candidates)
            prompt = self.prompt_template.format(
                persona=persona,
                context=context,
            )

            result = await self.provider.generate_async(
                input=prompt,
                max_tokens=self.max_tokens,
                target_keywords=["<winner>", "</winner>"],
                max_retries=3,
            )

            # Parse the winner ID from shuffled candidates
            # -1 = intentional abstention, -2 = parse failure
            try:
                shuffled_winner_id = int(parse_xml_tag(result["content"], "winner"))
                if shuffled_winner_id == -1:
                    winner_id = -1  # Intentional abstention
                elif shuffled_winner_id >= 0 and shuffled_winner_id < len(shuffle_idx):
                    winner_id = shuffle_idx[shuffled_winner_id]
                else:
                    winner_id = -2  # Out-of-range ID
            except Exception:
                winner_id = -2  # Parse failure

            return persona_idx, persona, result, winner_id

        # Run all persona judgments in parallel
        tasks = [judge_with_persona(i, persona, shuffle_indices[i]) for i, persona in enumerate(personas)]
        results = await asyncio.gather(*tasks)

        # Aggregate scores from all persona judgments
        for persona_idx, _, result, winner_id in results:
            if winner_id >= 0:
                aggregated_scores[winner_id] += 1
            elif winner_id == -1:
                print(f"Persona {persona_idx} chose not to engage with any post.")
            else:
                print(f"Failed to parse winner ID for persona {persona_idx}.")
            full_judgment_responses[persona_idx] = result

        return aggregated_scores, full_judgment_responses


class PairwiseJudge:
    """Judge that picks a winner between two responses (Bradley-Terry style)."""

    def __init__(
        self,
        provider: AsyncProvider,
        all_personas: list[str] | None = None,
        max_personas: int | None = None,
    ):
        self.provider = provider
        self.all_personas = all_personas
        self.max_personas = max_personas

    async def judge_pair(
        self,
        request: str,
        response_a: str,
        response_b: str,
        seed: int = 0,
    ) -> Tuple[int, dict]:
        """Judge a single pair of responses (no personas).

        Args:
            request: The user request both agents responded to.
            response_a: First agent's response.
            response_b: Second agent's response.
            seed: Random seed for position swapping (for reproducibility).

        Returns:
            Tuple of (winner, result_dict) where winner is 0 for A, 1 for B.
        """
        rng = random.Random(seed)

        # Randomly swap order to avoid position bias
        swap = rng.random() < 0.5

        if swap:
            response_a, response_b = response_b, response_a

        prompt = PAIRWISE_JUDGE_PROMPT.format(
            request=request,
            response_a=response_a,
            response_b=response_b,
        )

        result = await self.provider.generate_async(
            input=prompt,
            max_tokens=1024,
            target_keywords=["<winner>", "</winner>"],
            max_retries=3,
        )

        try:
            winner_str = parse_xml_tag(result["content"], "winner").strip().upper()
            if winner_str == "A":
                winner = 0
            elif winner_str == "B":
                winner = 1
            else:
                winner = -1
        except Exception:
            winner = -1

        # Unswap if needed
        if swap and winner >= 0:
            winner = 1 - winner

        return winner, result

    async def judge_pair_with_personas(
        self,
        request: str,
        response_a: str,
        response_b: str,
        seed: int = 0,
    ) -> Tuple[int, int, dict]:
        """Judge a pair using multiple personas (aggregated voting).

        Args:
            request: The user request both agents responded to.
            response_a: First agent's response.
            response_b: Second agent's response.
            seed: Random seed for persona sampling and position swapping (for reproducibility).

        Returns:
            Tuple of (votes_a, votes_b, results_dict).
        """
        rng = random.Random(seed)

        # Sample personas for this call
        if self.all_personas and self.max_personas and len(self.all_personas) > self.max_personas:
            personas = rng.sample(self.all_personas, self.max_personas)
        else:
            personas = self.all_personas or []

        if not personas:
            winner, result = await self.judge_pair(request, response_a, response_b, seed=seed)
            if winner == 0:
                return (1, 0, {"single": result})
            elif winner == 1:
                return (0, 1, {"single": result})
            else:
                return (0, 0, {"single": result})

        # Pre-generate swap decisions for each persona (async-safe)
        swap_decisions = [rng.random() < 0.5 for _ in personas]

        async def judge_with_persona(
            persona_idx: int,
            persona: str,
        ) -> Tuple[int, str, dict, int]:
            # Use pre-generated swap decision (async-safe)
            swap = swap_decisions[persona_idx]
            ra, rb = (response_b, response_a) if swap else (response_a, response_b)

            prompt = PAIRWISE_JUDGE_WITH_PERSONA_PROMPT.format(
                persona=persona,
                request=request,
                response_a=ra,
                response_b=rb,
            )

            result = await self.provider.generate_async(
                input=prompt,
                max_tokens=1024,
                target_keywords=["<winner>", "</winner>"],
                max_retries=3,
            )

            try:
                winner_str = parse_xml_tag(result["content"], "winner").strip().upper()
                if winner_str == "A":
                    winner = 0
                elif winner_str == "B":
                    winner = 1
                else:
                    winner = -1
            except Exception:
                winner = -1

            # Unswap if needed
            if swap and winner >= 0:
                winner = 1 - winner

            return persona_idx, persona, result, winner

        tasks = [judge_with_persona(i, persona) for i, persona in enumerate(personas)]
        results = await asyncio.gather(*tasks)

        votes_a = 0
        votes_b = 0
        for persona_idx, _, _, w in results:
            if w == 0:
                votes_a += 1
            elif w == 1:
                votes_b += 1
            else:
                print(f"Failed to parse winner for persona {persona_idx}, skipping vote.")

        results_dict = {persona: {"result": result, "winner": winner} for _, persona, result, winner in results}

        return votes_a, votes_b, results_dict
