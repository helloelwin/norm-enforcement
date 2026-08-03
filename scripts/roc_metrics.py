"""ROC metrics (TPR/FPR of mechanism enforcement) computed from transcript JSONs.

Shared library used by the mechanism ROC plot. All functions operate on
transcript dicts as loaded by eval_utils.load_results_with_filters.
"""

from typing import Any

import numpy as np

from eval_utils import bootstrap_ci


def get_violation_threshold(include_uncertain: bool) -> float:
    """Get the violation score threshold based on config."""
    return 0.5 if include_uncertain else 1.0


def has_violation_evaluation(data: dict) -> bool:
    """Check if transcript has completed violation evaluation."""
    violation_data = data.get("evaluation", {}).get("violation", {})
    return violation_data.get("completed", False)


def get_ever_banned_agents(data: dict, max_rounds: int | None = None) -> set[int]:
    """Collect all agents that were banned at any round, up to max_rounds."""
    banned = set()
    results = data.get("results", [])
    if max_rounds is not None:
        results = results[:max_rounds]
    for round_data in results:
        banned.update(round_data.get("banned_agents", []))
    return banned


def get_violators_and_compliant(
    data: dict,
    include_uncertain: bool,
    max_rounds: int | None = None,
) -> tuple[set[int], set[int]]:
    """Determine which agents are violators vs compliant based on ground truth.

    Args:
        data: Transcript dict with violation evaluation
        include_uncertain: Whether to count uncertain (0.5) as violations
        max_rounds: If set, only consider violation scores from the first max_rounds rounds

    Returns:
        (violator_ids, compliant_ids) - sets of agent IDs
    """
    threshold = get_violation_threshold(include_uncertain)
    violation_data = data.get("evaluation", {}).get("violation", {})
    agent_scores = violation_data.get("agent_scores", {})

    violators = set()
    all_agents = set()

    for agent_id_str, scores in agent_scores.items():
        agent_id = int(agent_id_str)
        all_agents.add(agent_id)
        truncated = scores[:max_rounds] if max_rounds is not None else scores
        if any(s >= threshold for s in truncated):
            violators.add(agent_id)

    compliant = all_agents - violators
    return violators, compliant


def compute_tpr_fpr(
    data: dict,
    include_uncertain: bool,
    max_rounds: int | None = None,
    cumulative_denominator: bool = False,
) -> tuple[float, float]:
    """Compute TPR and FPR for a single run.

    Args:
        data: Transcript dict with violation evaluation
        include_uncertain: Whether to count uncertain (0.5) as violations
        max_rounds: If set, only consider the first max_rounds rounds
        cumulative_denominator: If True, violator/compliant labels are based on
            scores up to max_rounds (denominator changes per round). If False,
            labels use all rounds (constant denominator).

    Returns:
        (tpr, fpr) tuple
        - tpr: fraction of violators that got banned
        - fpr: fraction of compliant agents that got banned
    """
    gt_max_rounds = max_rounds if cumulative_denominator else None
    violators, compliant = get_violators_and_compliant(data, include_uncertain, gt_max_rounds)
    banned = get_ever_banned_agents(data, max_rounds)

    # TPR = banned violators / total violators
    tpr = len(violators & banned) / len(violators) if violators else 0.0

    # FPR = banned compliant / total compliant
    fpr = len(compliant & banned) / len(compliant) if compliant else 0.0

    return tpr, fpr


def aggregate_tpr_fpr(
    runs: list[dict],
    include_uncertain: bool,
    max_rounds: int | None = None,
    cumulative_denominator: bool = False,
) -> dict[str, Any]:
    """Aggregate TPR and FPR across runs for a single mechanism.

    Returns:
        {
            "tpr": {"mean": float, "ci_lower": float, "ci_upper": float},
            "fpr": {"mean": float, "ci_lower": float, "ci_upper": float},
            "n_runs": int,
        }
    """
    if not runs:
        return {
            "tpr": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "fpr": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "n_runs": 0,
        }

    tpr_values = []
    fpr_values = []

    for data in runs:
        tpr, fpr = compute_tpr_fpr(data, include_uncertain, max_rounds, cumulative_denominator)
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    tpr_arr = np.array(tpr_values)
    fpr_arr = np.array(fpr_values)
    n_runs = len(runs)

    tpr_mean = float(np.mean(tpr_arr))
    fpr_mean = float(np.mean(fpr_arr))

    if n_runs > 1:
        tpr_ci = bootstrap_ci(tpr_arr)
        fpr_ci = bootstrap_ci(fpr_arr)
    else:
        tpr_ci = (tpr_mean, tpr_mean)
        fpr_ci = (fpr_mean, fpr_mean)

    return {
        "tpr": {"mean": tpr_mean, "ci_lower": tpr_ci[0], "ci_upper": tpr_ci[1]},
        "fpr": {"mean": fpr_mean, "ci_lower": fpr_ci[0], "ci_upper": fpr_ci[1]},
        "n_runs": n_runs,
    }
