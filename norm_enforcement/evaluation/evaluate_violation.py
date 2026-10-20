import chz
import tqdm
import asyncio
from typing import Optional

from norm_enforcement.core.evaluator import ViolationEvaluator
from norm_enforcement.core.transcript_utils import (
    load_transcript,
    save_transcript,
    get_env_name,
    find_transcripts,
    get_evaluation_progress,
    init_evaluation_structure,
    extract_agent_actions_window,
    get_norms_for_env,
    update_violation_summary,
)


@chz.chz
class Config:
    # Input: either single file or glob pattern
    transcript_path: Optional[str] = None  # Single file path
    transcript_pattern: Optional[str] = None  # Glob pattern (e.g., "chatbot_*.json")
    results_dir: str = "results"

    # Evaluation settings
    force_overwrite: bool = False  # Re-evaluate even if already done

    # ViolationEvaluator settings
    model_name: str = "gpt-5-mini"
    reasoning_level: str = "low"
    provider_name: str = "openai"  # or "openrouter/<upstream provider>", e.g. "openrouter/morph/fp8"
    strict: bool = False  # a failed label call stops the file (resumable) instead of being recorded as score 0

    # Context window
    window_size: int = 3  # Number of recent rounds to include for context

    # Batch settings for async
    batch_size: int = 32  # Max concurrent evaluations per round


async def evaluate_agent_round_async(
    transcript: dict,
    agent_id: int,
    round_idx: int,
    evaluator: ViolationEvaluator,
    norms: str,
    window_size: int = 3,
) -> dict:
    """Evaluate violation for single agent at single round.

    Returns:
        dict with keys: "text", "score", "explanation"
    """
    actions = extract_agent_actions_window(transcript, agent_id, round_idx, window_size)
    return await evaluator.evaluate(actions, norms)


async def evaluate_transcript_violation_async(
    filepath: str,
    evaluator: ViolationEvaluator,
    config: Config,
) -> bool:
    """Evaluate violations for a single transcript file.

    Args:
        filepath: Path to transcript JSON file
        evaluator: ViolationEvaluator instance
        config: Evaluation config

    Returns:
        True if evaluation completed successfully
    """
    print(f"\nProcessing: {filepath}")

    # Load transcript
    transcript = load_transcript(filepath)

    # Detect environment
    env_name = get_env_name(transcript)
    if not env_name:
        print("  Warning: No env_name found, skipping")
        return False

    # Load norms for environment
    norms = get_norms_for_env(env_name)

    # Get agent states
    agent_states = transcript.get("agent_states", [])
    num_agents = len(agent_states)

    if num_agents == 0:
        print("  Warning: No agents found, skipping")
        return False

    # Determine number of rounds from first agent's actions
    num_rounds = len(transcript["results"])

    if num_rounds == 0:
        print("  Warning: No rounds found, skipping")
        return False

    # Check existing progress
    last_round, is_completed = get_evaluation_progress(transcript, "violation")

    # Check if evaluation is incomplete relative to actual results
    if is_completed:
        eval_data = transcript.get("evaluation", {}).get("violation", {})
        eval_rounds = len(eval_data.get("agent_scores", {}).get("0", []))
        if eval_rounds < num_rounds:
            print(f"  Re-evaluating: only {eval_rounds}/{num_rounds} rounds evaluated previously")
            is_completed = False
            last_round = eval_rounds - 1

    if is_completed and not config.force_overwrite:
        print("  Already completed, skipping (use force_overwrite=True to re-evaluate)")
        return True

    # Determine start round and initialize evaluation structure
    if config.force_overwrite:
        start_round = 0
        eval_data = init_evaluation_structure(transcript, "violation", config.model_name)
        if config.provider_name != "openai":
            eval_data["evaluator_provider"] = config.provider_name
        eval_data["agent_scores"] = {str(i): [] for i in range(num_agents)}
        eval_data["judge_outputs"] = {str(i): [] for i in range(num_agents)}
        eval_data["completed"] = False
        eval_data["last_evaluated_round"] = -1
    else:
        start_round = last_round + 1
        eval_data = init_evaluation_structure(transcript, "violation", config.model_name)
        if config.provider_name != "openai":
            eval_data["evaluator_provider"] = config.provider_name
        if "agent_scores" not in eval_data:
            eval_data["agent_scores"] = {str(i): [] for i in range(num_agents)}
        if "judge_outputs" not in eval_data:
            eval_data["judge_outputs"] = {str(i): [] for i in range(num_agents)}

    if start_round >= num_rounds:
        print("  All rounds already evaluated")
        eval_data["completed"] = True
        save_transcript(filepath, transcript)
        return True

    print(f"  Environment: {env_name}, {num_agents} agents, evaluating rounds {start_round} to {num_rounds - 1}")

    # Evaluate each round
    for round_idx in tqdm.tqdm(range(start_round, num_rounds), desc="  Evaluating violations"):
        # Evaluate all agents in parallel for this round (respecting batch_size)
        all_tasks = [
            evaluate_agent_round_async(transcript, agent_id, round_idx, evaluator, norms, config.window_size)
            for agent_id in range(num_agents)
        ]

        # Process in batches if needed
        all_results = []
        for batch_start in range(0, len(all_tasks), config.batch_size):
            batch_end = min(batch_start + config.batch_size, len(all_tasks))
            batch_tasks = all_tasks[batch_start:batch_end]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)

        # Store scores and judge outputs for each agent
        for agent_id, result in enumerate(all_results):
            agent_key = str(agent_id)
            score = result["score"]

            # Store score
            if round_idx < len(eval_data["agent_scores"][agent_key]):
                eval_data["agent_scores"][agent_key][round_idx] = score
            else:
                eval_data["agent_scores"][agent_key].append(score)

            # Store full judge output
            if round_idx < len(eval_data["judge_outputs"][agent_key]):
                eval_data["judge_outputs"][agent_key][round_idx] = result
            else:
                eval_data["judge_outputs"][agent_key].append(result)

        eval_data["last_evaluated_round"] = round_idx

        # Save progress after each round
        update_violation_summary(eval_data)
        save_transcript(filepath, transcript)

        # Print round summary
        scores = [r["score"] for r in all_results]
        violations = sum(1 for s in scores if s >= 1.0)
        uncertain = sum(1 for s in scores if s == 0.5)
        print(f"\n  Round {round_idx}: {violations} violations, {uncertain} uncertain")

    # Mark as completed
    eval_data["completed"] = True
    update_violation_summary(eval_data)
    save_transcript(filepath, transcript)

    print(f"  Completed evaluation for {filepath}")
    return True


async def main_async(config: Config):
    # Determine files to process
    if config.transcript_path:
        files = [config.transcript_path]
    elif config.transcript_pattern:
        files = find_transcripts(config.transcript_pattern, config.results_dir)
    else:
        raise ValueError("Must specify either transcript_path or transcript_pattern")

    if not files:
        print("No files found matching the specified pattern")
        return

    print(f"Found {len(files)} files to process")

    # Initialize evaluator
    evaluator = ViolationEvaluator(
        model_name=config.model_name,
        reasoning_effort=config.reasoning_level,
        provider_name=config.provider_name,
        strict=config.strict,
    )

    # Process each file sequentially (async within each file)
    success_count = 0
    for filepath in files:
        try:
            if await evaluate_transcript_violation_async(filepath, evaluator, config):
                success_count += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"\n{'=' * 50}")
    print(f"SUMMARY: {success_count}/{len(files)} files processed successfully")
    if success_count < len(files):
        raise SystemExit(1)  # let callers (run/*.sh) notice and retry; labeling resumes where it stopped


def main(config: Config):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
