"""Label coupled runs, so that identical action windows get identical labels in every arm, optionally with several samples.

    python -m norm_enforcement.evaluation.evaluate_coupled results_dir=DIR [summary_pattern="*seed=3.json"] [num_samples=3]

The arms of each coupled run (one summary in <results_dir>/coupled/, written by norm_enforcement.coupled) are labeled
together through one shared memo (norm_enforcement/llms/memo.py). Each label request is the release's labeler prompt
for one agent and round (its last window_size actions, as in evaluate_violation), and a request whose text is
identical across arms, as in the stretch before the arms diverge, receives the same reply. Different samples of the
same request are drawn independently.

Labels go where evaluate_violation puts them (evaluation.violation, taken from sample 0), so the release's metric
code works unchanged. With num_samples > 1, evaluation.violation_majority holds the per-round median of the samples
(the majority, for three samples) in the same format, and evaluation.violation_samples keeps every sample.
"""

import asyncio
import glob
import json
import os
from collections import Counter

import chz

from norm_enforcement.core.evaluator import ViolationEvaluator
from norm_enforcement.core.transcript_utils import (
    extract_agent_actions_window,
    get_env_name,
    get_norms_for_env,
    init_evaluation_structure,
    load_transcript,
    save_transcript,
    update_violation_summary,
)
from norm_enforcement.llms.memo import CallMemo, MemoProvider, provider_tag


@chz.chz
class Config:
    results_dir: str = "results"
    summary_pattern: str = "*.json"  # matched against <results_dir>/coupled/
    num_samples: int = 1
    model_name: str = "gpt-5-mini"
    reasoning_level: str = "low"
    provider_name: str = "openai"  # or "openrouter/<upstream provider>", e.g. "openrouter/morph/fp8"
    strict: bool = False  # a failed label call fails the run (retried later) instead of being recorded as score 0
    window_size: int = 3
    concurrency: int = 32
    force_overwrite: bool = False


class _Labeler(ViolationEvaluator):
    """The release's evaluator, reading through a given provider."""

    def __init__(self, provider, strict=False):
        self.provider = provider
        self.strict = strict


def _median(values: list[float]) -> float:
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _done(transcript: dict, num_samples: int) -> bool:
    ev = transcript.get("evaluation", {})
    if not ev.get("violation", {}).get("completed"):
        return False
    return num_samples == 1 or ev.get("violation_samples", {}).get("num_samples") == num_samples


async def label_run(summary_path: str, config: Config) -> bool:
    with open(summary_path) as f:
        summary = json.load(f)
    paths = summary["transcripts"]
    transcripts = {arm: load_transcript(p) for arm, p in paths.items() if os.path.exists(p)}
    if len(transcripts) != len(paths):
        print(f"  missing transcripts for {summary_path}, skipping")
        return False
    if not config.force_overwrite and all(_done(t, config.num_samples) for t in transcripts.values()):
        print(f"  already labeled: {summary_path}")
        return True

    memo_path = summary_path.replace(os.sep + "coupled" + os.sep, os.sep + "memo" + os.sep)
    memo_path = os.path.splitext(memo_path)[0] + f".labels_{config.model_name.split('/')[-1]}.jsonl"
    memo = CallMemo(memo_path)
    base = ViolationEvaluator(model_name=config.model_name, reasoning_effort=config.reasoning_level,
                              provider_name=config.provider_name).provider
    tag = provider_tag(base)
    sem = asyncio.Semaphore(config.concurrency)

    labelers = {}
    for arm in transcripts:
        counts = Counter()
        for k in range(config.num_samples):
            labelers[(arm, k)] = _Labeler(MemoProvider(base, memo, counts, arm=arm, role=f"label{k}", tag=f"{tag}#sample{k}"),
                                          strict=config.strict)

    async def one(arm, k, actions, norms):
        async with sem:
            return await labelers[(arm, k)].evaluate(actions, norms)

    jobs, where = [], []
    for arm, t in transcripts.items():
        norms = get_norms_for_env(get_env_name(t))
        n_agents, n_rounds = len(t.get("agent_states", [])), len(t["results"])
        for r in range(n_rounds):
            for a in range(n_agents):
                actions = extract_agent_actions_window(t, a, r, config.window_size)
                for k in range(config.num_samples):
                    jobs.append(one(arm, k, actions, norms))
                    where.append((arm, a, r, k))
    print(f"  {summary_path}: {len(jobs)} label requests over {len(transcripts)} arms")
    results = await asyncio.gather(*jobs)
    memo.close()

    for arm, t in transcripts.items():
        n_agents, n_rounds = len(t.get("agent_states", [])), len(t["results"])
        samples = [[[None] * n_rounds for _ in range(n_agents)] for _ in range(config.num_samples)]
        outputs = [[None] * n_rounds for _ in range(n_agents)]
        for (arm_, a, r, k), res in zip(where, results):
            if arm_ != arm:
                continue
            samples[k][a][r] = res["score"]
            if k == 0:
                outputs[a][r] = res
        eval_data = init_evaluation_structure(t, "violation", config.model_name)
        eval_data["agent_scores"] = {str(a): samples[0][a] for a in range(n_agents)}
        eval_data["judge_outputs"] = {str(a): outputs[a] for a in range(n_agents)}
        eval_data["completed"] = True
        eval_data["last_evaluated_round"] = n_rounds - 1
        eval_data["labeling"] = {"coupled": True, "memo": memo_path}
        if config.provider_name != "openai":
            eval_data["evaluator_provider"] = config.provider_name
        update_violation_summary(eval_data)
        if config.num_samples > 1:
            majority = {"version": "1.0", "evaluator": config.model_name, "completed": True,
                        "last_evaluated_round": n_rounds - 1, "rounds": [],
                        "agent_scores": {str(a): [_median([samples[k][a][r] for k in range(config.num_samples)])
                                                  for r in range(n_rounds)] for a in range(n_agents)}}
            update_violation_summary(majority)
            t["evaluation"]["violation_majority"] = majority
            t["evaluation"]["violation_samples"] = {"num_samples": config.num_samples, "evaluator": config.model_name,
                                                    "agent_scores": {str(a): [samples[k][a] for k in range(config.num_samples)]
                                                                     for a in range(n_agents)}}
        save_transcript(paths[arm], t)
    stats = Counter()
    for (a, role, kind), v in memo.stats.items():
        stats[kind] += v
    print(f"  labeled {len(transcripts)} arms; label calls made {stats['miss']}, shared {stats['hit']}, replayed {stats['replayed']}")
    return True


async def main_async(config: Config):
    summaries = sorted(glob.glob(os.path.join(config.results_dir, "coupled", config.summary_pattern)))
    print(f"Found {len(summaries)} coupled runs")
    ok = 0
    for path in summaries:
        try:
            ok += await label_run(path, config)
        except Exception as e:  # keep going; the caller retries failures
            print(f"Error labeling {path}: {e}")
    print(f"SUMMARY: {ok}/{len(summaries)} coupled runs labeled")
    if ok < len(summaries):
        raise SystemExit(1)


def main(config: Config):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
