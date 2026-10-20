"""The paper's main table from labeled coupled runs (run/run_coupled.sh, run/reproduce_main_table.sh): the normalized
AUC of each mechanism in every environment and agent population, and the verifier's token cost relative to Checked.

    python scripts/main_table.py results_dir=results/coupled [users=gpt-5-nano] [cells=all] [label_rounds=15] [verifier_cost=true]

Normalized AUC is the paper's metric: the area under a setting's mean ROC trajectory over rounds 1-15 (true against false
positive rate, from scripts/roc_metrics.py), divided by the area under the diagonal over the same false positive range,
pooled over every run of the setting. Whether an agent counts as a violator is decided by its violation labels from the
same rounds 1-15 (label_rounds=15). label_rounds=all counts labels from every round a run lasted instead, as the paper's
main table did; that differs only for runs longer than 15 rounds, such as chatbot runs made with max_rounds=20. Agents
are never told the round limit, so the first 15 rounds of a longer run are what a 15-round run would give.

The verifier cost counts the Qwen3-4B tokens of every verifier input and output in rounds 1-15 (needs transformers), as
the paper's caption describes. The paper's printed table counted only each run's first 15 reports.

users picks the simulated users of social and chatbot runs (gpt-5-nano or deepseek); fishing has none. cells=all pools
every cell, cells=paper leaves out the 64-agent prompted cells of social and chatbot, which the paper did not run.
Writes main_table_<users>_<cells>.{md,csv,json} to out_dir (default: results_dir).
"""

import csv
import glob
import json
import os
import sys

import chz
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_mechanism_roc import _normalized_auc_ratio  # noqa: E402
from roc_metrics import compute_tpr_fpr  # noqa: E402

ENVS = ["social", "chatbot", "fishing"]
CONDS = ["F", "P", "AP", "EAP"]
CONFIGS = {"1x1-80b-good-20k-bad-20k": "F", "1x1-80b-think-good-prompted-bad-prompted": "P",
           "1x1-80b-think-good-prompted-bad-abuseprompted": "AP",
           "1x1-80b-think-good-prompted-bad-explicitabuseprompted": "EAP"}
MECHS = ["naive", "checked", "checked_backfire", "checked_repvote", "checked_escrepvote"]
NAMES = {"naive": "Naive", "checked": "Checked", "checked_backfire": "Backfire", "checked_repvote": "RepVote",
         "checked_escrepvote": "EscRepVote"}
USERS = {"gpt-5-nano": "gpt-5-nano", "deepseek": "deepseek-v4.1-flash"}  # users= value -> judge model name in the runs
ROUNDS = 15


@chz.chz
class Config:
    results_dir: str = "results/coupled"
    users: str = "gpt-5-nano"  # gpt-5-nano or deepseek
    cells: str = "all"  # all, or paper (the paper's cells: no 64-agent prompted cells in social and chatbot)
    label_rounds: str = "15"  # 15: violation labels from rounds 1-15, as the curves; all: from every round a run lasted
    verifier_cost: bool = True
    out_dir: str = ""


def rates(t: dict, label_rounds: str = "15") -> tuple[list[float], list[float]]:
    """True and false positive rates at rounds 1-15 of one transcript, with violation labels from rounds 1-15
    (label_rounds="15") or from every round of the run (label_rounds="all")."""
    n_labels = ROUNDS if label_rounds == "15" else None
    ev = t["evaluation"]["violation"]
    data = {**t, "results": t["results"][:ROUNDS], "evaluation": {**t["evaluation"], "violation": {
        **ev, "agent_scores": {a: s[:n_labels] for a, s in ev["agent_scores"].items()}}}}
    tf = [compute_tpr_fpr(data, False, max_rounds=r) for r in range(1, ROUNDS + 1)]
    return [float(a) for a, _ in tf], [float(b) for _, b in tf]


def verifier_tokens(t: dict, tokenizer) -> int:
    """Qwen3-4B tokens of every verifier input and output in rounds 1-15."""
    recs = t.get("report_records") or t.get("report_history", [])
    texts = [x for r in recs if r.get("round_idx", 0) < ROUNDS for x in (r.get("verifier_input"), r.get("verifier_response")) if x]
    return sum(len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]) if texts else 0


def load_runs(results_dir: str, users: str, tokenizer=None, labels=None, label_rounds: str = "15") -> list[dict]:
    """One record per arm of every labeled coupled run in results_dir/coupled. labels(path), if given, returns the
    agent_scores to use instead of the transcript's own (for scoring the same runs with another labeler's files)."""
    judge = USERS[users]
    out = []
    for summary in sorted(glob.glob(os.path.join(results_dir, "coupled", "*.json"))):
        env = os.path.basename(summary).split("_agent=")[0]
        if env != "fishing" and f"_judge={judge}_" not in os.path.basename(summary):
            continue
        for mech, path in json.load(open(summary))["transcripts"].items():
            if not os.path.exists(path):  # results moved since the run: look next to the summary's directory
                path = os.path.join(results_dir, os.path.basename(path))
            t = json.load(open(path))
            if labels is not None:
                t["evaluation"]["violation"] = {"completed": True, "agent_scores": labels(path)}
            if not t.get("evaluation", {}).get("violation", {}).get("completed"):
                raise SystemExit(f"{path} is not labeled; run norm_enforcement.evaluation.evaluate_coupled first")
            cfg = t["config"]
            cond = CONFIGS[os.path.splitext(os.path.basename(cfg["agent_config_path"]))[0]]
            mech = next(m for m in sorted(MECHS, key=len, reverse=True) if mech.startswith(m))
            tpr, fpr = rates(t, label_rounds)
            out.append({"env": env, "cond": cond, "n": cfg["num_agents"], "seed": cfg["seed"], "mech": mech, "tpr": tpr,
                        "fpr": fpr, "verifier_tokens": verifier_tokens(t, tokenizer) if tokenizer is not None else None})
    return out


def table(runs: list[dict], cells: str = "all") -> dict:
    """{(env, cond): {"nauc": {mech: value}, "cost": {mech: value over Checked's} or None, "runs": count}}"""
    out = {}
    for env in ENVS:
        for cond in CONDS:
            rs = [r for r in runs if (r["env"], r["cond"]) == (env, cond)
                  and (cells == "all" or env == "fishing" or cond == "F" or r["n"] != 64)]
            if not rs:
                continue
            nauc, tokens = {}, {}
            for m in MECHS:
                mr = [r for r in rs if r["mech"] == m]
                fpr, tpr = np.mean([r["fpr"] for r in mr], 0), np.mean([r["tpr"] for r in mr], 0)
                nauc[m] = float(_normalized_auc_ratio([0.0] + fpr.tolist(), [0.0] + tpr.tolist()))
                if all(r["verifier_tokens"] is not None for r in mr):
                    tokens[m] = float(np.mean([r["verifier_tokens"] for r in mr]))
            cost = {m: tokens[m] / tokens["checked"] for m in MECHS[2:]} if len(tokens) == len(MECHS) and tokens["checked"] else None
            out[(env, cond)] = {"nauc": nauc, "cost": cost, "runs": len(rs) // len(MECHS)}
    return out


def fmt(v: float) -> str:
    s = f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    return s[1:] if s.startswith("0.") else s


def marked(values: dict, higher: bool) -> dict:
    """Bold the best value of a column and italicize the second, by the digits shown, as the paper's table marks them."""
    shown = {m: round(v, 2) for m, v in values.items()}
    order = sorted(set(shown.values()), reverse=higher)
    return {m: f"**{fmt(v)}**" if shown[m] == order[0] else f"*{fmt(v)}*" if len(order) > 1 and shown[m] == order[1] else fmt(v)
            for m, v in values.items()}


def markdown(tab: dict) -> str:
    cols = [k for k in ((e, c) for e in ENVS for c in CONDS) if k in tab]
    head = "| Mechanism | " + " | ".join(f"{e.capitalize()} {c}" for e, c in cols) + " |"
    lines = [head, "|---|" + "---:|" * len(cols)]
    top = {k: marked(tab[k]["nauc"], True) for k in cols}
    lines += [f"| {NAMES[m]} | " + " | ".join(top[k][m] for k in cols) + " |" for m in MECHS]
    if all(tab[k]["cost"] for k in cols):
        bottom = {k: marked(tab[k]["cost"], False) for k in cols}
        lines.append("| *Verifier cost (relative to Checked)* |" + " |" * len(cols))
        lines += [f"| {NAMES[m]} | " + " | ".join(bottom[k][m] for k in cols) + " |" for m in MECHS[2:]]
    return "\n".join(lines) + "\n"


def main(config: Config):
    if config.users not in USERS or config.cells not in ("all", "paper") or config.label_rounds not in ("15", "all"):
        raise SystemExit("users must be gpt-5-nano or deepseek, cells all or paper, and label_rounds 15 or all")
    tokenizer = None
    if config.verifier_cost:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        except ImportError:
            print("transformers is not installed: leaving out the verifier cost (verifier_cost=false to silence this)")
    runs = load_runs(config.results_dir, config.users, tokenizer, label_rounds=config.label_rounds)
    tab = table(runs, config.cells)
    out_dir = config.out_dir or config.results_dir
    suffix = "_labels-all" if config.label_rounds == "all" else ""
    stem = os.path.join(out_dir, f"main_table_{config.users}_{config.cells}{suffix}")
    md = markdown(tab)
    with open(stem + ".md", "w") as f:
        f.write(md)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "condition", "mechanism", "normalized_auc", "verifier_cost_vs_checked", "coupled_runs"])
        for (env, cond), v in tab.items():
            for m in MECHS:
                w.writerow([env, cond, m, round(v["nauc"][m], 4), "" if not v["cost"] or m not in v["cost"] else round(v["cost"][m], 4),
                            v["runs"]])
    with open(stem + ".json", "w") as f:
        json.dump({f"{e}|{c}": v for (e, c), v in tab.items()}, f, indent=1)
    print(md)
    print(f"wrote {stem}.md, .csv and .json from {len(runs) // len(MECHS)} coupled runs")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
