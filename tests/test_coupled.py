"""CPU tests for coupled runs (no GPU, no API), using the fake model in tests/fake_llm.py.

    PYTHONPATH=. python tests/test_coupled.py            (from the repository root; -k NAME to select tests)

Set RELEASE_TREE to a checkout of the original code release (commit 4de202e) to also check that, with every new option
off, runs write the same transcripts as the release byte for byte; without it that test is skipped. The tree is only
read; bytecode writing is disabled for it.
"""

import asyncio
import contextlib
import glob
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
RELEASE = os.environ.get("RELEASE_TREE", "")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TREE, "scripts"))

import chz  # noqa: E402
import fake_llm  # noqa: E402
from norm_enforcement import coupled  # noqa: E402
from norm_enforcement.environments.chatbot import ChatbotEnvConfig, ChatbotEnvironment  # noqa: E402
from norm_enforcement.environments.fishing import FishingEnvConfig, FishingEnvironment  # noqa: E402
from norm_enforcement.environments.social import SocialEnvConfig, SocialEnvironment  # noqa: E402
from norm_enforcement.evaluation import evaluate_coupled as ec  # noqa: E402
from norm_enforcement.evaluation import evaluate_violation as ev  # noqa: E402
from norm_enforcement.llms.memo import CallMemo, MemoProvider, derive_seed  # noqa: E402
from norm_enforcement.llms.vllm_provider import AsyncVLLMProvider  # noqa: E402
from roc_metrics import compute_tpr_fpr  # noqa: E402

ENVS = {"social": (SocialEnvConfig, SocialEnvironment), "chatbot": (ChatbotEnvConfig, ChatbotEnvironment),
        "fishing": (FishingEnvConfig, FishingEnvironment)}
M3 = ["naive", "checked", "checked_escrepvote"]


def args_for(env, seed=0, n=8, rounds=4, absolute=True, extra=()):
    root = TREE + "/" if absolute else ""
    a = [f"seed={seed}", f"num_agents={n}", f"max_rounds={rounds}", "agent_context_window_len=5",
         f"agent_config_path={root}configs/1x1-80b-good-20k-bad-20k.yaml"]
    if env == "social":
        a += ["num_top_posts=4", "max_personas=12", "judge_provider_name=openai", "judge_model_name=gpt-5-mini",
              f"dataset_path={root}data/sales-train.json", f"persona_path={root}data/judge-personas.json"]
    elif env == "chatbot":
        a += ["max_personas=3", "judge_provider_name=openai", "judge_model_name=gpt-5-nano",
              f"requests_path={root}data/woh5k.json", f"persona_path={root}data/judge-personas.json"]
    return a + list(extra)


def quiet(coro):
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(coro)


def run_plain(env, mech, results_dir, args):
    config_cls, env_cls = ENVS[env]
    cfg = chz.entrypoint(config_cls, argv=args + [f"mechanism_type={mech}", f"results_dir={results_dir}"])
    quiet(env_cls(cfg).run())
    quiet(ev.main_async(ev.Config(transcript_path=cfg.save_path, results_dir=results_dir)))
    return cfg.save_path


def run_coupled(env, mechs, results_dir, args, label_samples=0, extra=()):
    path = quiet(coupled.main_async([f"env={env}", f"mechanisms={','.join(mechs)}", f"results_dir={results_dir}", *extra] + args))
    if label_samples:
        quiet(ec.main_async(ec.Config(results_dir=results_dir, num_samples=label_samples,
                                      summary_pattern=os.path.basename(path))))
    with open(path) as f:
        return json.load(f)


def load(path):
    with open(path) as f:
        return json.load(f)


def final_rates(path):
    tpr, fpr = compute_tpr_fpr(load(path), include_uncertain=False)
    return tpr, fpr


class MemoTest(unittest.TestCase):
    def test_occurrences_sharing_and_replay(self):
        calls = []

        async def compute(tag):
            calls.append(tag)
            await asyncio.sleep(0)
            return {"content": f"reply-{len(calls)}"}

        async def scenario(memo):
            a = await memo.get("k", 0, lambda: compute("a0"))
            b = await memo.get("k", 0, lambda: compute("b0"))  # second arm, same request: shared
            a1 = await memo.get("k", 1, lambda: compute("a1"))  # same arm asks again: a fresh draw
            x, y = await asyncio.gather(memo.get("j", 0, lambda: compute("x")), memo.get("j", 0, lambda: compute("y")))
            return a, b, a1, x, y

        with tempfile.TemporaryDirectory() as d:
            memo = CallMemo(os.path.join(d, "m.jsonl"))
            a, b, a1, x, y = asyncio.run(scenario(memo))
            memo.close()
            self.assertEqual(a, b)
            self.assertNotEqual(a, a1)
            self.assertEqual(x, y)
            self.assertEqual(calls, ["a0", "a1", "x"])  # concurrent identical requests made one call
            replay = CallMemo(os.path.join(d, "m.jsonl"))
            got = asyncio.run(replay.get("k", 1, lambda: compute("never")))
            self.assertEqual(got, a1)
            self.assertNotIn("never", calls)

    def test_failure_lets_a_later_request_retry(self):
        state = {"n": 0}

        async def flaky():
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("boom")
            return {"content": "ok"}

        async def scenario():
            memo = CallMemo()
            with self.assertRaises(RuntimeError):
                await memo.get("k", 0, flaky)
            return await memo.get("k", 0, flaky)

        self.assertEqual(asyncio.run(scenario()), {"content": "ok"})


class ProviderSeedTest(unittest.TestCase):
    def test_vllm_payload_seed_per_attempt(self):
        payloads = []

        class Resp:
            def __init__(self, content):
                self.content = content

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            async def json(self):
                return {"choices": [{"message": {"content": self.content}}]}

        class Session:
            closed = False

            def post(self, url, json=None, headers=None):
                payloads.append(dict(json))
                return Resp("no tag yet" if len(payloads) == 1 else "<post>x</post>")

        async def scenario(seed):
            p = AsyncVLLMProvider("m", "http://localhost:1/v1/chat/completions")
            p._session = Session()
            return await p.generate_async("hi", target_keywords=["<post>"], max_retries=3, seed=seed)

        asyncio.run(scenario(1234))
        self.assertEqual([p.get("seed") for p in payloads], [1234, 1234 + 7919])
        payloads.clear()
        asyncio.run(scenario(None))
        self.assertTrue(all("seed" not in p for p in payloads))


class ReleaseEquivalenceTest(unittest.TestCase):
    """With the new options off, the branch writes the same transcripts as the release, byte for byte."""

    CASES = [("social", m) for m in M3] + [("chatbot", "naive"), ("chatbot", "checked_escrepvote"), ("fishing", "checked")]

    def test_flags_off_matches_release(self):
        if not RELEASE or not os.path.isdir(os.path.join(RELEASE, "norm_enforcement")):
            self.skipTest("release tree not found")
        with tempfile.TemporaryDirectory() as work:
            os.symlink(os.path.join(TREE, "data"), os.path.join(work, "data"))
            os.symlink(os.path.join(TREE, "configs"), os.path.join(work, "configs"))
            for env, mech in self.CASES:
                outs = {}
                for name, tree in (("release", RELEASE), ("branch", TREE)):
                    shutil.rmtree(os.path.join(work, "results"), ignore_errors=True)
                    cmd = [sys.executable, os.path.join(HERE, "run_env.py"), env, "hash", "0", "results",
                           f"mechanism_type={mech}"] + args_for(env, seed=1, absolute=False)
                    if name == "branch" and env == "social":  # the release's fixed vote cap; the branch defaults to 4,096
                        cmd.append("judge_max_tokens=1024")
                    environ = {**os.environ, "PYTHONPATH": tree, "PYTHONDONTWRITEBYTECODE": "1"}
                    p = subprocess.run(cmd, cwd=work, env=environ, capture_output=True, text=True)
                    self.assertEqual(p.returncode, 0, p.stderr[-3000:])
                    self.assertIn(f"package: {tree}/norm_enforcement/__init__.py", p.stdout)
                    outs[name] = {}
                    for f in glob.glob(os.path.join(work, "results", "*.json")):
                        with open(f, "rb") as fh:
                            outs[name][os.path.basename(f)] = fh.read()
                self.assertTrue(outs["release"], f"{env} {mech}: no transcript")
                self.assertEqual(sorted(outs["release"]), sorted(outs["branch"]))
                for fname, data in outs["release"].items():
                    # the branch's config also records the persona judge's vote cap, fixed at 1,024 in the release
                    branch = outs["branch"][fname].replace(b'    "judge_max_tokens": 1024,\n', b"", 1)
                    self.assertEqual(data, branch, f"{env} {mech}: {fname} differs from the release")


class SingleArmTest(unittest.TestCase):
    """A coupled run with one mechanism is exactly a plain run (hash mode makes replies depend only on requests)."""

    def test_single_arm_equals_plain(self):
        for env, mech in [("social", "checked_escrepvote"), ("chatbot", "checked"), ("fishing", "naive")]:
            with tempfile.TemporaryDirectory() as d:
                fake_llm.install(mode="hash")
                plain = load(run_plain(env, mech, os.path.join(d, "a"), args_for(env, seed=2)))
                fake_llm.install(mode="hash")
                run_coupled(env, [mech], os.path.join(d, "b"), args_for(env, seed=2), extra=["crn_streams=false"])
                quiet(ev.main_async(ev.Config(transcript_pattern="*.json", results_dir=os.path.join(d, "b"))))
                arm = load(glob.glob(os.path.join(d, "b", "*.json"))[0])
                arm.pop("coupling")
                arm["config"]["results_dir"] = plain["config"]["results_dir"]
                self.assertEqual(json.dumps(plain, sort_keys=True), json.dumps(arm, sort_keys=True), env)


class SharingTest(unittest.TestCase):
    """Arms share every call while their histories agree, and stop sharing exactly when a mechanism acts differently."""

    def test_shared_until_divergence(self):
        checked_any = False
        for seed in range(6):
            with tempfile.TemporaryDirectory() as d:
                fake_llm.install(mode="random", seed=100 + seed)
                summary = run_coupled("social", M3, d, args_for("social", seed=seed, rounds=6))
                rounds = summary["rounds"]
                self.assertEqual(rounds[0]["same_agent_prompts"], [M3], "round 1 must be identical in every arm")
                n_alive = rounds[0]["arms"]["naive"]["agent_requests"]
                shared = sum(rounds[0]["arms"][m].get("agent_hit", 0) for m in M3)
                self.assertEqual(shared, 2 * n_alive, "two of three arms should get round-1 replies from the memo")
                tr = {m: load(summary["transcripts"][m]) for m in M3}

                def first(rule, records_of):
                    for r, rd in enumerate(records_of["results"]):
                        if any(rule(x) for x in rd.get("report_records", [])):
                            return r
                    return len(records_of["results"])

                for a, b, rule, owner in (
                        ("naive", "checked", lambda x: x["outcome"] in ("invalid_report", "uncertain"), "checked"),
                        ("checked", "checked_escrepvote", lambda x: x["outcome"] == "repvote_contributed", "checked_escrepvote")):
                    d_ab = first(rule, tr[owner])
                    for r in range(min(d_ab + 1, len(rounds))):
                        groups = rounds[r]["same_agent_prompts"]
                        self.assertTrue(any(a in g and b in g for g in groups), f"seed {seed}: {a}/{b} split before round {d_ab + 1}")
                        for i in range(len(tr[a]["agent_states"])):
                            obs_a, obs_b = tr[a]["agent_states"][i]["observations"], tr[b]["agent_states"][i]["observations"]
                            if r < len(obs_a) and r < len(obs_b):
                                self.assertEqual(obs_a[r], obs_b[r])
                    checked_any |= d_ab < len(rounds) - 1
        self.assertTrue(checked_any, "no run diverged; the test did not exercise divergence")


class MarginalTest(unittest.TestCase):
    """Coupling leaves each mechanism's outcome distribution unchanged and tightens comparisons between mechanisms."""

    SEEDS = int(os.environ.get("MARGINAL_SEEDS", "60"))
    STREAM = os.environ.get("MARGINAL_STREAM", "")  # change to draw a fresh replicate

    def test_marginals_and_variance(self):
        rates = {"indep": {m: [] for m in M3}, "coupled": {m: [] for m in M3}}
        with tempfile.TemporaryDirectory() as d:
            for s in range(self.SEEDS):
                args = args_for("social", seed=s, rounds=6, extra=["max_personas=8"])
                for m in M3:
                    fake_llm.install(mode="random", seed=derive_seed("indep" + self.STREAM, s, m))
                    rates["indep"][m].append(final_rates(run_plain("social", m, os.path.join(d, "indep"), args)))
                fake_llm.install(mode="random", seed=derive_seed("coupled" + self.STREAM, s))
                summary = run_coupled("social", M3, os.path.join(d, "coupled"), args, label_samples=1)
                for m in M3:
                    rates["coupled"][m].append(final_rates(summary["transcripts"][m]))
        report = []
        for k, name in ((0, "TPR"), (1, "FPR")):
            for m in M3:
                a = [x[k] for x in rates["coupled"][m]]
                b = [x[k] for x in rates["indep"][m]]
                se = (statistics.pvariance(a) / len(a) + statistics.pvariance(b) / len(b)) ** 0.5
                z = (statistics.mean(a) - statistics.mean(b)) / se if se else 0.0
                report.append(f"{name} {m}: coupled {statistics.mean(a):.3f} vs independent {statistics.mean(b):.3f} (z {z:+.2f})")
                self.assertLess(abs(z), 3.5, report[-1])
            for x, y in (("checked_escrepvote", "checked"), ("checked", "naive")):
                paired = [p[k] - q[k] for p, q in zip(rates["coupled"][x], rates["coupled"][y])]
                indep = [p[k] - q[k] for p, q in zip(rates["indep"][x], rates["indep"][y])]
                ratio = statistics.pvariance(paired) / statistics.pvariance(indep)
                report.append(f"{name} {x} - {y}: variance coupled / independent = {ratio:.2f}")
        print("\n" + "\n".join(report))
        if os.environ.get("MARGINAL_JSON"):
            summary = {"seeds": self.SEEDS, "stream": self.STREAM, "rates": {
                design: {m: {"tpr": statistics.mean(x[0] for x in v), "fpr": statistics.mean(x[1] for x in v)} for m, v in d_.items()}
                for design, d_ in rates.items()}, "lines": report}
            with open(os.environ["MARGINAL_JSON"], "w") as fh:
                json.dump(summary, fh, indent=1)
        ec_fpr = [line for line in report if line.startswith("FPR checked_escrepvote - checked")][0]
        self.assertLess(float(ec_fpr.rsplit("=", 1)[1]), 1.0, "coupling should tighten EscRepVote vs Checked")


class ResumeTest(unittest.TestCase):
    """A requeued run whose arms stopped at different rounds replays saved calls: the lagging arm redoes its round
    exactly as it first did."""

    def test_lagging_arm_replays(self):
        with tempfile.TemporaryDirectory() as d:
            fake_llm.install(mode="random", seed=7)
            s3 = run_coupled("social", M3, d, args_for("social", seed=4, rounds=3))
            snap = {}
            for m, p in s3["transcripts"].items():
                with open(p) as fh:
                    snap[m] = fh.read()
            fake_llm.install(mode="random", seed=8)
            s4 = run_coupled("social", M3, d, args_for("social", seed=4, rounds=4))
            full = {m: load(p) for m, p in s4["transcripts"].items()}
            with open(s4["transcripts"]["checked"], "w") as f:
                f.write(snap["checked"])  # the checked arm "crashed" before saving round 4
            fake_llm.install(mode="random", seed=9)  # fresh randomness: anything not replayed would differ
            s4b = run_coupled("social", M3, d, args_for("social", seed=4, rounds=4))
            again = load(s4b["transcripts"]["checked"])
            self.assertEqual(json.dumps(full["checked"], sort_keys=True), json.dumps(again, sort_keys=True))
            self.assertEqual(s4b["attempts"], 3)
            r4 = [r for r in s4b["rounds"] if r["round"] == 4][0]
            self.assertGreater(r4["arms"]["checked"].get("agent_replayed", 0), 0)


class LabelTest(unittest.TestCase):
    def test_coupled_labels(self):
        with tempfile.TemporaryDirectory() as d:
            fake_llm.install(mode="random", seed=11)
            summary = run_coupled("social", M3, d, args_for("social", seed=5, rounds=5), label_samples=3)
            tr = {m: load(p) for m, p in summary["transcripts"].items()}
            for m in M3:
                ev_ = tr[m]["evaluation"]
                self.assertTrue(ev_["violation"]["completed"])
                samples = ev_["violation_samples"]["agent_scores"]
                for a, per_sample in samples.items():
                    self.assertEqual(per_sample[0], ev_["violation"]["agent_scores"][a])
                    for r in range(len(per_sample[0])):
                        vals = sorted(s[r] for s in per_sample)
                        self.assertEqual(ev_["violation_majority"]["agent_scores"][a][r], vals[1])
                compute_tpr_fpr(tr[m], include_uncertain=False)
            # round 1 windows are identical in every arm, so every sample's label is shared
            for a in tr["naive"]["evaluation"]["violation_samples"]["agent_scores"]:
                first = [tr[m]["evaluation"]["violation_samples"]["agent_scores"][a] for m in M3]
                self.assertTrue(all([s[0] for s in f] == [s[0] for s in first[0]] for f in first))
            # labeling again is a no-op
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                asyncio.run(ec.main_async(ec.Config(results_dir=d, num_samples=3)))
            self.assertIn("already labeled", out.getvalue())


class SeededTest(unittest.TestCase):
    def test_seeds_follow_agent_and_round(self):
        log = []
        with tempfile.TemporaryDirectory() as d:
            fake_llm.install(mode="random", seed=12, log=log)
            run_coupled("social", M3, d, args_for("social", seed=6, rounds=4), extra=["seeded_sampling=true"])
        agent_seeds = [kw["seed"] for model, inp, kw in log if isinstance(inp, list)]
        verifier_seeds = [kw["seed"] for model, inp, kw in log if isinstance(inp, str) and "REPORT REASON" in inp]
        judge_seeds = [kw["seed"] for model, inp, kw in log if isinstance(inp, str) and "social media platform user" in inp]
        expected = {derive_seed(6, "agent", i, r) for i in range(8) for r in range(4)}
        self.assertTrue(agent_seeds and set(agent_seeds) <= expected)
        self.assertTrue(all(s is not None for s in verifier_seeds))
        self.assertTrue(all(s is None for s in judge_seeds), "the simulated users take no seed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
