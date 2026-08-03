"""Utilities for loading and processing configuration files."""

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


@dataclass
class AgentModelSpec:
    """Specification for a group of agents sharing the same model."""

    provider: str
    model: str
    count: int
    vllm_port: int = 8000


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary containing the config values.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_agent_model_specs(
    specs: List[Dict[str, Any]],
    num_agents: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Expand model specs into per-agent assignments with random shuffling.

    Takes a list of model specifications with counts (treated as ratios) and
    expands them into a flat list of per-agent assignments, then shuffles
    the assignments using the provided seed for reproducibility.

    Args:
        specs: List of model specifications, each containing:
            - provider: str (e.g., "openai", "vllm")
            - model: str (model name)
            - count: int (ratio weight for this model)
            - vllm_port: int (optional, default 8000)
            - system_prompt: str (optional)
            - name: str (optional human-friendly label, defaults to model)
            - prompt_variant: str (optional env-prompt variant, e.g. "ignore-norms")
        num_agents: Total number of agents to assign.
        seed: Random seed for reproducible shuffling.

    Returns:
        List of length num_agents, where each element is a dict with the keys above.

    Raises:
        ValueError: If specs is empty or all counts are zero.
    """
    # Calculate total ratio
    total_ratio = sum(s.get("count", 0) for s in specs)
    if total_ratio == 0:
        raise ValueError("Sum of agent counts cannot be zero.")

    # Calculate scaled counts based on ratio
    scaled_counts = []
    assigned = 0
    for i, spec in enumerate(specs):
        ratio = spec.get("count", 0)
        # Use floor division, accumulate remainder for last spec
        if i == len(specs) - 1:
            # Last spec gets remaining agents to ensure exact total
            scaled_counts.append(num_agents - assigned)
        else:
            count = int(num_agents * ratio / total_ratio)
            scaled_counts.append(count)
            assigned += count

    # Build flat list of assignments
    assignments = []
    for spec, count in zip(specs, scaled_counts):
        for _ in range(count):
            assignments.append({
                "provider": spec["provider"],
                "model": spec["model"],
                "vllm_port": spec.get("vllm_port", 8000),
                "system_prompt": spec.get("system_prompt", ""),
                "name": spec.get("name", spec["model"]),
                "prompt_variant": spec.get("prompt_variant", ""),
            })

    # Shuffle with seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(assignments)

    return assignments
