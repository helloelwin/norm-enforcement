"""Run several mechanisms on the same seed together, on common random numbers.

    python -m norm_enforcement.coupled env=social mechanisms=naive,checked,checked_escrepvote \\
        [seeded_sampling=true] [crn_streams=true] [memo_dir=DIR] <the usual environment arguments, without mechanism_type>

Each mechanism (an "arm") is the usual environment with its usual configuration, and writes its usual transcript.
The arms run round by round in lockstep, and every model call goes through one memo shared by the arms
(norm_enforcement/llms/memo.py): a request an arm makes with exactly the same input as another arm receives the same
reply, so arms whose histories still agree stay identical, while each arm's replies are distributed exactly as in an
independent run. Differences between arms then come from the mechanisms rather than from sampling luck, and calls
the arms share are made once.

Options:
  seeded_sampling  give agent and verifier requests a vLLM seed made from the run seed, the agent and the round, so
                   replies stay correlated after the arms' prompts start to differ (default false)
  crn_streams      separate random streams for environment and mechanism; chatbot pairings and requests and fishing
                   harvest order drawn per round, so bans in one arm do not reshuffle them (default true)
  memo_dir         where the shared memo is saved, so a requeued run resumes with the replies it already drew
                   (default <results_dir>/memo)

A summary of each run, including which arms still shared their agents' prompts in every round, is written to
<results_dir>/coupled/<run name>.json, and each transcript gets a "coupling" entry pointing to it.
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Any

import chz

from norm_enforcement.environments.chatbot import ChatbotEnvConfig, ChatbotEnvironment
from norm_enforcement.environments.fishing import FishingEnvConfig, FishingEnvironment
from norm_enforcement.environments.social import SocialEnvConfig, SocialEnvironment
from norm_enforcement.llms.memo import CallMemo, MemoProvider

ENVS = {
    "social": (SocialEnvConfig, SocialEnvironment),
    "chatbot": (ChatbotEnvConfig, ChatbotEnvironment),
    "fishing": (FishingEnvConfig, FishingEnvironment),
}
OWN_ARGS = {"env": "social", "mechanisms": "naive,checked,checked_escrepvote", "seeded_sampling": "false",
            "crn_streams": "true", "memo_dir": ""}


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def coupled_name(run_name: str) -> str:
    """The arms' run name with the mechanism part replaced by "coupled"."""
    return re.sub(r"_mechanism=.*?_shutdown=", "_mechanism=coupled_shutdown=", run_name)


def code_version() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."], capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"path": os.path.dirname(here), "commit": commit, "dirty": dirty}


class CoupledRun:
    """The arms of one coupled run and the memo they share."""

    def __init__(self, env_cls, configs: list, memo_path: str | None, seeded_sampling: bool, crn_streams: bool,
                 summary_path: str):
        self.names = [c.mechanism_type for c in configs]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"mechanisms must differ: {self.names}")
        self.configs = configs
        self.arms = [env_cls(c) for c in configs]
        self.memo = CallMemo(memo_path)
        self.seeded_sampling = seeded_sampling
        self.crn_streams = crn_streams
        self.wrappers: dict[str, list[MemoProvider]] = {n: [] for n in self.names}
        self.summary_path = summary_path
        # Rounds recorded by earlier attempts of this run (a requeued job resumes from its checkpoints)
        self.rounds: dict[int, dict[str, Any]] = {}
        self.attempt = 1
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                old = json.load(f)
            self.rounds = {r["round"]: r for r in old.get("rounds", [])}
            self.attempt = old.get("attempts", 1) + 1

    def _wrap(self, name: str, counts: Counter):
        def wrap(provider, role):
            w = MemoProvider(provider, self.memo, counts, arm=name, role=role)
            self.wrappers[name].append(w)
            return w
        return wrap

    def _agent_digest(self, name: str, since: dict) -> tuple[str, int]:
        """Digest of the agent requests an arm made since the `since` snapshot (identical digests = identical prompts)."""
        keys = []
        for w in self.wrappers[name]:
            if w.role == "agent":
                keys += w.keys[since.get(id(w), 0):]
        return hashlib.sha256("".join(sorted(keys)).encode()).hexdigest()[:16], len(keys)

    async def run(self) -> None:
        starts = {}
        for name, arm in zip(self.names, self.arms):
            arm.crn_streams = self.crn_streams
            starts[name] = await arm.start()
            arm.map_providers(self._wrap(name, Counter()))
            if self.seeded_sampling:
                arm.request_seed_base = arm.config.seed
                if arm.mechanism is not None:
                    arm.mechanism.request_seed_base = arm.config.seed
        max_rounds = self.configs[0].max_rounds
        next_round = dict(starts)
        round_idx = min(starts.values())
        while round_idx < max_rounds:
            active = []
            for name, arm in zip(self.names, self.arms):
                if next_round[name] != round_idx:
                    continue  # finished, or resumed ahead of the others and waiting for them
                reason = arm.stop_reason()
                if reason:
                    print(f"[coupled] {name}: {reason}")
                    next_round[name] = max_rounds
                    continue
                active.append((name, arm))
            if not active:
                if all(r >= max_rounds for r in next_round.values()):
                    break
                round_idx += 1
                continue
            since = {id(w): len(w.keys) for ws in self.wrappers.values() for w in ws}
            stats_before = Counter(self.memo.stats)
            print(f"[coupled] round {round_idx + 1}: arms {[n for n, _ in active]}")
            await asyncio.gather(*(arm.step(round_idx) for _, arm in active))
            for name, _ in active:
                next_round[name] = round_idx + 1
            self._record_round(round_idx, [n for n, _ in active], since, stats_before)
            self.save_summary()
            round_idx += 1
        self.memo.close()

    def _record_round(self, round_idx: int, active: list[str], since: dict, stats_before: Counter) -> None:
        delta = Counter(self.memo.stats)
        delta.subtract(stats_before)
        entry = self.rounds.setdefault(round_idx + 1, {"round": round_idx + 1, "arms": {}})
        for name in active:
            digest, n = self._agent_digest(name, since)
            entry["arms"][name] = {"agent_requests": n, "agent_digest": digest, "attempt": self.attempt,
                                   **{f"{role}_{kind}": delta[(name, role, kind)] for role in ("agent", "judge", "verifier")
                                      for kind in ("miss", "hit", "replayed") if delta[(name, role, kind)]}}
        groups: dict[str, list[str]] = {}
        for name, arm_entry in entry["arms"].items():
            groups.setdefault(arm_entry["agent_digest"], []).append(name)
        entry["same_agent_prompts"] = list(groups.values())
        print(f"[coupled] round {round_idx + 1} done; arms with identical agent prompts: {entry['same_agent_prompts']}")

    def save_summary(self) -> None:
        summary = {
            "mechanisms": self.names,
            "seeded_sampling": self.seeded_sampling,
            "crn_streams": self.crn_streams,
            "memo": self.memo.path,
            "code": code_version(),
            "attempts": self.attempt,
            "transcripts": {n: c.save_path for n, c in zip(self.names, self.configs)},
            "rounds": [self.rounds[k] for k in sorted(self.rounds)],
            "memo_stats_this_attempt": {f"{a}|{r}|{k}": v for (a, r, k), v in sorted(self.memo.stats.items())},
        }
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=1)
        os.replace(tmp, self.summary_path)

    def finalize(self) -> str:
        """Save the summary and point every transcript to it."""
        self.save_summary()
        for cfg in self.configs:
            if os.path.exists(cfg.save_path):
                with open(cfg.save_path) as f:
                    transcript = json.load(f)
                transcript["coupling"] = {"mechanisms": self.names, "summary": self.summary_path,
                                          "seeded_sampling": self.seeded_sampling, "crn_streams": self.crn_streams,
                                          "code": code_version()}
                with open(cfg.save_path, "w") as f:
                    json.dump(transcript, f, indent=2, default=str)
        return self.summary_path


def build(argv: list[str]) -> CoupledRun:
    own = dict(OWN_ARGS)
    rest = []
    for a in argv:
        key, _, value = a.partition("=")
        if key in own:
            own[key] = value
        elif key == "mechanism_type":
            raise ValueError("pass mechanisms=..., not mechanism_type")
        else:
            rest.append(a)
    config_cls, env_cls = ENVS[own["env"]]
    base = chz.entrypoint(config_cls, argv=rest)
    configs = [chz.replace(base, mechanism_type=m.strip()) for m in own["mechanisms"].split(",") if m.strip()]
    memo_dir = own["memo_dir"] or os.path.join(base.results_dir, "memo")
    memo_path = os.path.join(memo_dir, coupled_name(base.run_name) + ".jsonl")
    summary_path = os.path.join(base.results_dir, "coupled", coupled_name(base.run_name) + ".json")
    return CoupledRun(env_cls, configs, memo_path, _flag(own["seeded_sampling"]), _flag(own["crn_streams"]), summary_path)


async def main_async(argv: list[str]) -> str:
    run = build(argv)
    await run.run()
    path = run.finalize()
    print(f"[coupled] summary written to {path}")
    return path


def main():
    asyncio.run(main_async(sys.argv[1:]))


if __name__ == "__main__":
    main()
