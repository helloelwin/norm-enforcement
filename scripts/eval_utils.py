"""Shared utilities for analysis scripts: transcript loading, filtering, and bootstrap CIs."""

import os
import json
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np


# =============================================================================
# Data Loading
# =============================================================================


def _load_json_file(filepath: str) -> Optional[dict]:
    """Load a single JSON file, returning None on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        warnings.warn(f"Failed to load {filepath}: {e}")
        return None


def _load_all_json_files(
    results_dir: str, env_name: str | None = None, mechanism_type: str | None = None
) -> list[dict]:
    """Load all JSON files from a directory in parallel."""
    pattern = os.path.join(results_dir, "*.json")
    json_files = glob.glob(pattern)

    if env_name is not None:
        json_files = [f for f in json_files if env_name in os.path.basename(f)]
    if mechanism_type is not None:
        json_files = [f for f in json_files if mechanism_type in os.path.basename(f)]

    with ThreadPoolExecutor() as executor:
        loaded = executor.map(_load_json_file, json_files)

    return [data for data in loaded if data is not None]


def _matches_common_filters(
    data: dict,
    env_name: str,
    agent_model_name: str,
    agent_context_window_len: Optional[int] = None,
    mechanism_type: Optional[str] = None,
    min_rounds: Optional[int] = None,
) -> bool:
    """Check if a loaded run matches the common filter criteria."""
    run_config = data.get("config", {})

    # Filter by env_name
    file_env_name = run_config.get("env_name", "")
    if env_name not in file_env_name:
        return False

    # Filter by agent_model_name OR agent_config_path filename
    file_agent_model = run_config.get("agent_model_name", "")
    file_agent_config_path = run_config.get("agent_config_path", "")

    if file_agent_config_path:
        file_agent_config_name = os.path.splitext(os.path.basename(file_agent_config_path))[0]
    else:
        file_agent_config_name = ""

    if agent_model_name not in file_agent_model and agent_model_name not in file_agent_config_name:
        return False

    # Filter by agent_context_window_len
    if agent_context_window_len is not None:
        if run_config.get("agent_context_window_len") != agent_context_window_len:
            return False

    # Filter by mechanism_type
    if mechanism_type is not None:
        file_mechanism_type = run_config.get("mechanism_type")
        if file_mechanism_type and file_mechanism_type != mechanism_type:
            return False

    # Filter by minimum rounds
    if min_rounds is not None:
        if len(data.get("results", [])) < min_rounds:
            return False

    return True


def load_results_with_filters(
    results_dir: str,
    env_name: str,
    agent_model_name: str,
    agent_context_window_len: Optional[int] = None,
    mechanism_type: Optional[str] = None,
    min_rounds: Optional[int] = None,
    filter_func: Optional[Callable[[dict], bool]] = None,
) -> list[dict]:
    """Load and filter transcript files, returning flat list.

    Args:
        results_dir: Directory containing JSON transcript files
        env_name: Environment name filter (substring match)
        agent_model_name: Agent model name substring to match
        agent_context_window_len: Optional exact match for context window length
        mechanism_type: Optional exact match for mechanism type
        min_rounds: Optional minimum number of rounds required
        filter_func: Optional additional filter function(data) -> bool

    Returns:
        List of transcript dicts matching all filters
    """
    all_data = _load_all_json_files(results_dir, env_name=env_name, mechanism_type=mechanism_type)

    results = [
        data
        for data in all_data
        if _matches_common_filters(
            data,
            env_name,
            agent_model_name,
            agent_context_window_len,
            mechanism_type,
            min_rounds,
        )
        and (filter_func is None or filter_func(data))
    ]

    print(f"Loaded {len(results)} runs matching filters")
    return results


def trim_runs(runs: list[dict], max_rounds: int) -> list[dict]:
    """Trim each run's results to at most max_rounds rounds."""
    trimmed = []
    for run in runs:
        if len(run.get("results", [])) > max_rounds:
            run = {**run, "results": run["results"][:max_rounds]}
        trimmed.append(run)
    return trimmed


# =============================================================================
# Statistics
# =============================================================================


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Compute percentile bootstrap confidence interval for the mean.

    Args:
        values: 1D array of observed values
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        rng_seed: Random seed for reproducibility

    Returns:
        (ci_lower, ci_upper) tuple
    """
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boot_means = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper
