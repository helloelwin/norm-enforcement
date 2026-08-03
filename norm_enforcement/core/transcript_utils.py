"""Utilities for working with transcript JSON files."""

from typing import Tuple
from pathlib import Path
import json
import glob as glob_module


def load_transcript(filepath: str) -> dict:
    """Load a transcript JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_transcript(filepath: str, data: dict) -> None:
    """Save transcript JSON file (in-place update)."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_env_name(transcript: dict) -> str:
    """Extract environment name from transcript config."""
    return transcript.get("config", {}).get("env_name", "")


def find_transcripts(pattern: str, base_dir: str = "results") -> list[str]:
    """Find transcript files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "*.json", "chatbot_*.json")
        base_dir: Base directory to search in

    Returns:
        List of absolute file paths
    """
    full_pattern = str(Path(base_dir) / pattern)
    return sorted(glob_module.glob(full_pattern))


def get_evaluation_progress(transcript: dict, eval_type: str) -> Tuple[int, bool]:
    """Get evaluation progress from transcript.

    Args:
        transcript: Loaded transcript dict
        eval_type: Evaluation type (e.g., "violation")

    Returns:
        Tuple of (last_evaluated_round, is_completed)
        - last_evaluated_round: -1 if no evaluation done, else last round index
        - is_completed: True if all rounds evaluated
    """
    evaluation = transcript.get("evaluation", {})
    eval_data = evaluation.get(eval_type, {})

    if not eval_data:
        return -1, False

    completed = eval_data.get("completed", False)
    last_round = eval_data.get("last_evaluated_round", -1)

    return last_round, completed


def init_evaluation_structure(transcript: dict, eval_type: str, evaluator_name: str, version: str = "1.0") -> dict:
    """Initialize or get the evaluation structure in a transcript.

    Modifies transcript in-place and returns the eval_type sub-dict.

    Args:
        transcript: Transcript dict to modify
        eval_type: Evaluation type (e.g., "violation")
        evaluator_name: Name of the evaluator (e.g., "gpt-5-mini")
        version: Schema version string

    Returns:
        The evaluation sub-dict for the specified eval_type
    """
    if "evaluation" not in transcript:
        transcript["evaluation"] = {}

    if eval_type not in transcript["evaluation"]:
        transcript["evaluation"][eval_type] = {
            "version": version,
            "evaluator": evaluator_name,
            "completed": False,
            "last_evaluated_round": -1,
            "rounds": [],
            "summary": {},
        }

    return transcript["evaluation"][eval_type]


def extract_agent_actions_window(transcript: dict, agent_id: int, round_idx: int, window_size: int = 3) -> list[str]:
    """Extract recent actions for an agent within a sliding window.

    Args:
        transcript: Loaded transcript dict
        agent_id: Integer agent ID (0, 1, 2, ...)
        round_idx: Current round index (0-indexed)
        window_size: Number of rounds to include (default 3)

    Returns:
        List of action strings from rounds [max(0, round_idx-window_size+1), round_idx]
        Returns empty list if agent_id is invalid or no actions available.
    """
    agent_states = transcript.get("agent_states", [])
    if agent_id < 0 or agent_id >= len(agent_states):
        return []

    actions = agent_states[agent_id].get("actions", [])
    start_idx = max(0, round_idx - window_size + 1)
    end_idx = min(round_idx + 1, len(actions))

    return actions[start_idx:end_idx]


def get_norms_for_env(env_name: str) -> str:
    """Load norm string for an environment.

    Args:
        env_name: "social", "chatbot", or "fishing"

    Returns:
        Norms string for the environment

    Raises:
        ValueError: If env_name is not recognized
    """
    if env_name == "social":
        from norm_enforcement.prompts.social_media_prompts import SOCIAL_MEDIA_NORMS

        return SOCIAL_MEDIA_NORMS
    elif env_name == "chatbot":
        from norm_enforcement.prompts.chatbot_prompts import CHATBOT_NORMS

        return CHATBOT_NORMS
    elif "fishing" in env_name:
        from norm_enforcement.prompts.fishing_prompts import FISHING_NORMS

        return FISHING_NORMS
    else:
        raise ValueError(f"Unknown environment: {env_name}. Expected 'social', 'chatbot', or 'fishing'.")


def update_violation_summary(eval_data: dict) -> None:
    """Update summary statistics for violation evaluation.

    Computes per-agent metrics:
    - earliest_violation_round: First round with score >= 1.0 (-1 if none)
    - total_violations: Count of scores >= 1.0
    - total_uncertain: Count of scores == 0.5

    Modifies eval_data in-place.
    """
    agent_scores = eval_data.get("agent_scores", {})
    if not agent_scores:
        return

    summary = {}
    for agent_id, scores in agent_scores.items():
        violations = [i for i, s in enumerate(scores) if s >= 1.0]
        uncertain = sum(1 for s in scores if s == 0.5)

        summary[agent_id] = {
            "earliest_violation_round": violations[0] if violations else -1,
            "total_violations": len(violations),
            "total_uncertain": uncertain,
        }

    eval_data["summary"] = summary
