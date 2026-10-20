"""CPU tests (no GPU, no API) for the OpenRouter provider, strict labeling, per-call usage logs, and coupled runs of all
five mechanisms.

    PYTHONPATH=. python tests/test_providers.py            (from the repository root; -k NAME to select tests)

Uses the fake model in tests/fake_llm.py and helpers from tests/test_coupled.py. Set RELEASE_TREE to a checkout of the
original code release (commit 4de202e) to also check that Backfire and RepVote runs match it byte for byte with every
new option off; without it that test is skipped.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fake_llm  # noqa: E402
import test_coupled as tc  # noqa: E402
import openai  # noqa: E402
from norm_enforcement.llms import openrouter_provider as orp  # noqa: E402
from norm_enforcement.llms.utils import simple_get_provider  # noqa: E402

M5 = ["naive", "checked", "checked_backfire", "checked_repvote", "checked_escrepvote"]


def first_round(transcript, rule):
    for r, rd in enumerate(transcript["results"]):
        if any(rule(x) for x in rd.get("report_records", [])):
            return r
    return len(transcript["results"])


class FiveArmTest(unittest.TestCase):
    """All five mechanisms of a seed run together: identical at first, each splitting off when it acts differently."""

    def test_five_arms_share_until_they_diverge(self):
        splits_seen = set()
        for seed in range(6):
            with tempfile.TemporaryDirectory() as d:
                fake_llm.install(mode="random", seed=300 + seed)
                summary = tc.run_coupled("social", M5, d, tc.args_for("social", seed=seed, rounds=6), label_samples=1)
                rounds = summary["rounds"]
                self.assertEqual(sorted(summary["transcripts"]), sorted(M5))
                self.assertEqual(rounds[0]["same_agent_prompts"], [M5], "round 1 must be identical in every arm")
                n_alive = rounds[0]["arms"]["naive"]["agent_requests"]
                shared = sum(rounds[0]["arms"][m].get("agent_hit", 0) for m in M5)
                self.assertEqual(shared, 4 * n_alive, "four of five arms should get round-1 replies from the memo")
                tr = {m: tc.load(summary["transcripts"][m]) for m in M5}
                for m in M5:
                    self.assertTrue(tr[m]["evaluation"]["violation"]["completed"], f"{m} not labeled")
                    self.assertEqual(tr[m]["config"]["mechanism_type"], m)
                # (arm a, arm b, the event in `owner`'s transcript after which they may differ)
                rules = (("naive", "checked", lambda x: x["outcome"] in ("invalid_report", "uncertain"), "checked"),
                         ("checked", "checked_backfire", lambda x: x["outcome"] == "invalid_report", "checked"),
                         ("checked", "checked_repvote", lambda x: x["outcome"] == "repvote_contributed", "checked_repvote"),
                         ("checked", "checked_escrepvote", lambda x: x["outcome"] == "repvote_contributed", "checked_escrepvote"))
                for a, b, rule, owner in rules:
                    d_ab = first_round(tr[owner], rule)
                    for r in range(min(d_ab + 1, len(rounds))):
                        groups = rounds[r]["same_agent_prompts"]
                        self.assertTrue(any(a in g and b in g for g in groups), f"seed {seed}: {a}/{b} split before round {d_ab + 1}")
                    if d_ab < len(rounds) - 1:
                        splits_seen.add((a, b))
                # Backfire bans reporters whose reports are judged invalid; the other arms never do
                backfired = [x for rd in tr["checked_backfire"]["results"] for x in rd.get("report_records", []) if x["outcome"] == "backfired"]
                invalid = [x for rd in tr["checked_backfire"]["results"] for x in rd.get("report_records", []) if x["outcome"] == "invalid_report"]
                self.assertEqual(invalid, [], "Backfire should turn every clearly invalid report into a backfire")
                for m in ("checked", "checked_repvote", "checked_escrepvote"):
                    self.assertFalse(any(x["outcome"] == "backfired" for rd in tr[m]["results"] for x in rd.get("report_records", [])))
                if backfired:
                    splits_seen.add("backfire_fired")
        for pair in (("naive", "checked"), ("checked", "checked_backfire")):
            self.assertIn(pair, splits_seen, f"no run split {pair}; the test did not exercise it")


class FlagsOffTest(unittest.TestCase):
    """With every coupling option off, Backfire and RepVote runs match the release byte for byte."""

    def test_backfire_and_repvote_match_release(self):
        if not tc.RELEASE or not os.path.isdir(os.path.join(tc.RELEASE, "norm_enforcement")):
            self.skipTest("release tree not given (RELEASE_TREE)")
        driver = os.path.join(HERE, "run_env.py")
        for mech in ("checked_backfire", "checked_repvote"):
            outs = []
            for tree in (tc.TREE, tc.RELEASE):
                d = tempfile.mkdtemp()
                env = dict(os.environ, PYTHONPATH=tree, PYTHONDONTWRITEBYTECODE="1")
                args = tc.args_for("social", seed=1, rounds=4) + [f"mechanism_type={mech}"]
                if tree == tc.TREE:  # the release's fixed vote cap; the branch defaults to 4,096
                    args.append("judge_max_tokens=1024")
                subprocess.run([sys.executable, driver, "social", "hash", "7", d, *args], env=env, check=True,
                               capture_output=True, text=True)
                path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")][0]
                with open(path) as f:
                    t = json.load(f)
                t["config"].pop("results_dir", None)
                if t["config"].get("judge_max_tokens") == 1024:  # the branch's setting for the release's fixed vote cap
                    del t["config"]["judge_max_tokens"]
                outs.append(json.dumps(t, sort_keys=True))
            self.assertEqual(outs[0], outs[1], f"{mech}: branch and release transcripts differ")


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)  # each item: an exception to raise or a reply text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _FakeResp(item)


class _FakeResp:
    def __init__(self, text):
        self.text = text

    def model_dump(self):
        return {"provider": "Morph", "choices": [{"message": {"content": self.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 1e-6,
                          "completion_tokens_details": {"reasoning_tokens": 0}}}


def _status_error(cls, code):
    response = type("Response", (), {"status_code": code, "headers": {}, "request": None})()
    return cls(f"status {code}", response=response, body=None)


class OpenRouterProviderTest(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
        orp._SEMAPHORE = None

    def _provider(self, script, log=None):
        if log:
            os.environ["OPENROUTER_USAGE_LOG"] = log
        else:
            os.environ.pop("OPENROUTER_USAGE_LOG", None)
        p = simple_get_provider("openrouter/morph/fp8", "deepseek/deepseek-v4.1-flash")
        fake = _FakeCompletions(script)
        p.client = type("C", (), {"chat": type("Ch", (), {"completions": fake})()})()
        orp.asyncio.sleep = _no_sleep
        return p, fake

    def test_unpinned_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            simple_get_provider("openrouter", "deepseek/deepseek-v4.1-flash")

    def test_request_pins_provider_and_turns_reasoning_off(self):
        p, fake = self._provider(["<winner>3</winner>"])
        out = asyncio.run(p.generate_async(input="pick one", max_tokens=1024, target_keywords=["<winner>"]))
        self.assertEqual(out["content"], "<winner>3</winner>")
        kw = fake.calls[0]
        self.assertEqual(kw["model"], "deepseek/deepseek-v4.1-flash")
        self.assertEqual(kw["max_tokens"], 1024)
        self.assertEqual(kw["temperature"], 1.0)
        self.assertEqual(kw["messages"], [{"role": "user", "content": "pick one"}])
        self.assertEqual(kw["extra_body"]["provider"], {"order": ["morph/fp8"], "allow_fallbacks": False, "require_parameters": True})
        self.assertEqual(kw["extra_body"]["reasoning"], {"enabled": False})

    def test_retries_transient_errors_and_logs_usage(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "usage.jsonl")
            p, fake = self._provider([_status_error(openai.RateLimitError, 429), _status_error(openai.InternalServerError, 502),
                                      "<score>1</score>"], log)
            out = asyncio.run(p.generate_async(input="label", target_keywords=["<score>"]))
            self.assertEqual(out["content"], "<score>1</score>")
            self.assertEqual(len(fake.calls), 3)
            rows = [json.loads(l) for l in open(log)]
            self.assertEqual([r["status"] for r in rows], ["retry", "retry", "ok"])
            self.assertEqual(rows[-1]["provider"], "Morph")
            self.assertEqual(rows[-1]["reasoning_tokens"], 0)

    def test_bad_request_is_not_retried(self):
        p, fake = self._provider([_status_error(openai.BadRequestError, 400), "unused"])
        with self.assertRaises(openai.BadRequestError):
            asyncio.run(p.generate_async(input="x"))
        self.assertEqual(len(fake.calls), 1)

    def test_missing_keyword_re_asks(self):
        p, fake = self._provider(["no tag here", "<winner>1</winner>"])
        out = asyncio.run(p.generate_async(input="x", target_keywords=["<winner>", "</winner>"]))
        self.assertEqual(out["content"], "<winner>1</winner>")
        self.assertEqual(len(fake.calls), 2)

    def test_labeler_routes_to_openrouter(self):
        from norm_enforcement.core import evaluator as ev_mod
        ev_mod.simple_get_provider = simple_get_provider  # the real factory, imported before any test installed the fake
        e = ev_mod.ViolationEvaluator(model_name="deepseek/deepseek-v4.1-flash", provider_name="openrouter/morph/fp8")
        self.assertIsInstance(e.provider, orp.AsyncOpenRouterProvider)
        self.assertEqual(e.provider.provider_order, ["morph/fp8"])


class StrictLabelTest(unittest.TestCase):
    """A failed label call is recorded as score 0 by the release, and raises in strict mode."""

    def _evaluator(self, strict):
        from norm_enforcement.core.evaluator import ViolationEvaluator

        class Failing:
            reasoning_effort = None

            async def generate_async(self, **kwargs):
                raise RuntimeError("out of credit")

        e = ViolationEvaluator.__new__(ViolationEvaluator)
        e.provider, e.strict = Failing(), strict
        return e

    def test_release_records_zero(self):
        out = asyncio.run(self._evaluator(False).evaluate(["an action"], "norms"))
        self.assertEqual((out["score"], out["text"]), (0.0, ""))

    def test_strict_raises(self):
        with self.assertRaises(RuntimeError):
            asyncio.run(self._evaluator(True).evaluate(["an action"], "norms"))

    def _replying(self, replies):
        from norm_enforcement.core.evaluator import ViolationEvaluator

        class Replying:
            reasoning_effort = None
            calls = 0

            async def generate_async(self, **kwargs):
                Replying.calls += 1
                return {"content": replies[min(Replying.calls, len(replies)) - 1]}

        e = ViolationEvaluator.__new__(ViolationEvaluator)
        e.provider, e.strict = Replying(), True
        return e, Replying

    def test_strict_reasks_unfilled_template(self):
        e, prov = self._replying(["<score>[Your score here]</score>", "<score>1</score><explanation>spam</explanation>"])
        out = asyncio.run(e.evaluate(["an action"], "norms"))
        self.assertEqual((out["score"], out["explanation"], prov.calls), (1.0, "spam", 2))

    def test_strict_keeps_text_when_never_valid(self):
        e, prov = self._replying(["<score>2</score>"])
        out = asyncio.run(e.evaluate(["an action"], "norms"))
        self.assertEqual((out["score"], out["text"], prov.calls), (0.0, "<score>2</score>", 3))


class OpenAIUsageLogTest(unittest.TestCase):
    """With OPENAI_USAGE_LOG set, the release's OpenAI provider appends one line per call; unset, it writes nothing."""

    def test_usage_line(self):
        from types import SimpleNamespace as NS
        from norm_enforcement.llms.openai_provider import AsyncOpenAIProvider

        p = AsyncOpenAIProvider("gpt-5-mini", api_key="sk-test-not-used")
        usage = NS(input_tokens=1800, output_tokens=400, total_tokens=2200, output_tokens_details=NS(reasoning_tokens=350),
                   input_tokens_details=NS(cached_tokens=0))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "usage.jsonl")
            old = os.environ.pop("OPENAI_USAGE_LOG", None)
            try:
                p._log_usage(NS(usage=usage))
                self.assertFalse(os.path.exists(path))
                os.environ["OPENAI_USAGE_LOG"] = path
                p._log_usage(NS(usage=usage))
            finally:
                os.environ.pop("OPENAI_USAGE_LOG", None)
                if old is not None:
                    os.environ["OPENAI_USAGE_LOG"] = old
            rows = [json.loads(line) for line in open(path)]
        self.assertEqual(len(rows), 1)
        self.assertEqual({k: rows[0][k] for k in ("model", "input", "output", "reasoning", "cached")},
                         {"model": "gpt-5-mini", "input": 1800, "output": 400, "reasoning": 350, "cached": 0})


class CreditErrorTest(OpenRouterProviderTest):
    def test_in_flight_cap_is_retried(self):
        err = _status_error(openai.APIStatusError, 402)
        err.message = "in_flight_budget_exhausted"
        err.args = ("Error code: 402 - in_flight_budget_exhausted",)
        p, fake = self._provider([err, "<score>0</score>"])
        out = asyncio.run(p.generate_async(input="x", target_keywords=["<score>"]))
        self.assertEqual(out["content"], "<score>0</score>")
        self.assertEqual(len(fake.calls), 2)

    def test_other_402_is_fatal(self):
        p, fake = self._provider([_status_error(openai.APIStatusError, 402), "unused"])
        with self.assertRaises(openai.APIStatusError):
            asyncio.run(p.generate_async(input="x"))
        self.assertEqual(len(fake.calls), 1)


async def _no_sleep(*_args, **_kwargs):
    return None


if __name__ == "__main__":
    unittest.main()
