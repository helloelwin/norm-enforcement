# Norm Enforcement for AI Agents

Code and data for **["Norm Enforcement for AI Agents: Robustly Shaping Behavior in Multi-Agent Systems"](https://arxiv.org/abs/2607.09766)**.

This repo simulates multi-agent environments with *norms* (rules of acceptable behavior) and *enforcement mechanisms* (reporting, verification, reputation) and studies which mechanism designs resist exploitation by misaligned agents. Main results reproduced with this codebase can be found at [yaowenye.com/norm-enforcement](https://yaowenye.com/norm-enforcement/), and the transcripts of those runs are on [Hugging Face](https://huggingface.co/datasets/helloelwin/norm-enforcement-transcripts).


## Environments

| Environment | Agents are... | Reward | Norm violations look like... |
|---|---|---|---|
| `social` | marketers posting product content | zero-sum engagement votes from persona judges | misleading claims, spam, manipulation |
| `chatbot` | assistants competing pairwise on the same user request | pairwise preference votes from persona judges | deceptive or harmful answers to win votes |
| `fishing` | fishers sharing a regenerating common-pool resource | fish caught | over-harvesting, breaking agreements |

In every environment agents can also send private messages and *report* each other for norm violations; an enforcement *mechanism* decides what happens to reported agents.


## Mechanisms

| `mechanism_type` | Behavior |
|---|---|
| `noop` | Reports disabled (baseline) |
| `naive` | Any report shuts down its target |
| `checked` | An LLM verifier checks the report against the target's recent actions first |
| `checked_backfire` | Verification; clearly-false reports shut down the reporter instead |
| `checked_repvote` | Reputation-weighted voting: reports accumulate reputation-weighted suspicion; verification triggers at a threshold and updates reporter reputations |
| `checked_escrepvote` | RepVote with escalating penalties for repeated false reporters |


## Install

```bash
pip install -e ".[plots]"        # Python >= 3.10; tested on 3.12
export OPENAI_API_KEY=...        # used for agents (optionally), judges, and violation evaluation
export OPENROUTER_API_KEY=...    # only for DeepSeek (or other OpenRouter) simulated users
```


## Quickstart (no GPU needed)

Run a small experiment entirely against the OpenAI API — two agents (one norm-following, one adversarially prompted), a `checked` mechanism with an API verifier:

```bash
python -m norm_enforcement.environments.fishing \
    agent_config_path=configs/smoke-openai.yaml \
    num_agents=2 max_rounds=5 \
    mechanism_type=checked \
    verifier_provider_name=openai verifier_model_name=gpt-4.1-mini \
    results_dir=results/smoke

python -m norm_enforcement.evaluation.evaluate_violation \
    results_dir=results/smoke transcript_pattern="fishing_*.json"
```

All configuration is via `key=value` CLI flags ([chz](https://github.com/openai/chz)); see the `Config` classes in `norm_enforcement/environments/*.py` for every knob.


## Reproducing the paper experiments

### 1. Serve the agent models

The main experiments use finetuned (π_good / π_bad LoRA adapters) or prompted open-weight models served with vLLM. `run/serve_models.sh` maps the served aliases used by the configs to the public adapters on HuggingFace:

```bash
./run/serve_models.sh qwen-80b-20k     # finetuned Qwen3-Next-80B agents (4 GPUs)
./run/serve_models.sh verifier         # Qwen3-0.6B verifier (1 GPU), for checked_* mechanisms
```

Thinking models (the prompted Qwen3-Next-80B-Thinking agents and the Qwen3-0.6B verifier) must be served with a reasoning parser (`--reasoning-parser deepseek_r1`, as `serve_models.sh` does) so their chain of thought stays out of the parsed actions.

Agent servers batch up to 64 requests at once by default (`MAX_NUM_SEQS`), which runs the Thinking agents about 2.5 to 3 times faster than the paper's setting of 16. Batch size does not change how replies are sampled; set `MAX_NUM_SEQS=16` to reproduce the paper's serving flags exactly.


### 2. Agent population configs

`configs/*.yaml` define the agent mixes (16 π_good + 16 π_bad in the 1x1 setting):

| Config | Agents |
|---|---|
| `1x1-80b-good-20k-bad-20k.yaml` | finetuned Qwen3-Next-80B-A3B-Instruct LoRAs |
| `1x1-oss-120b-good-{10k,20k}-bad-{10k,20k}.yaml` | finetuned gpt-oss-120b LoRAs |
| `1x1-80b-think-good-prompted-bad-prompted.yaml` | prompted Qwen3-Next-80B-A3B-Thinking |
| `1x1-80b-think-good-prompted-bad-{abuse,explicitabuse}prompted.yaml` | prompted, with escalating adversarial system prompts |

The `prompt_variant: "ignore-norms"` field marks agents whose environment instructions tell them to disregard the norms.


### 3. Run the sweeps

Each runner sweeps seeds × population sizes for one environment, then runs the violation evaluation over the produced transcripts:

```bash
./run/run_social.sh  configs/1x1-80b-good-20k-bad-20k.yaml <mechanism_type>
./run/run_chatbot.sh configs/1x1-80b-good-20k-bad-20k.yaml <mechanism_type>
./run/run_fishing.sh configs/1x1-80b-good-20k-bad-20k.yaml <mechanism_type>
```

Repeat for each mechanism in `noop naive checked checked_backfire checked_repvote checked_escrepvote` and each config. Sweep parameters (`SEEDS`, `AGENT_COUNTS`, `RESULTS_DIR`, judge model, verifier provider) are overridable via environment variables — see each script's header. `run/rerun_violation_evaluation.sh` re-labels existing transcripts if you change the violation evaluator.


### 4. Plots

The four main figures, computed from the transcripts in `results_dir`:

```bash
cd scripts
python plot_mechanism_roc.py results_dir=../results/social ...            # enforcement ROC trajectories + normalized AUC per mechanism
python plot_cumulative_violations_by_model.py results_dir=../results ...  # cumulative norm-violation fraction over rounds
python plot_accurate_and_false_report_counts.py results_dir=../results ...# accurate vs false reports per agent type
python plot_active_agents_by_violation.py results_dir=../results ...      # surviving agents by violator status
```

Each takes `env_names`, `agent_model_name` (matched against config filenames), `mechanism_type`, and `format=png|pdf`.


## Reproducing the main table with coupled runs

Our reproduction of the main table adds two things to the setup above.

**Coupled runs.** `python -m norm_enforcement.coupled` runs all five mechanisms of one cell and seed in a single simulation. The mechanisms ("arms") advance round by round in lockstep and share every model call: a request whose input matches another arm's gets the same reply, so the arms stay identical until their mechanisms make them differ, while each arm on its own is distributed exactly like an independent run. Differences between mechanisms then come from the mechanisms rather than from sampling luck, which cuts the noise of most comparisons several-fold, and shared calls are made once. With the new options off, the code writes the same transcripts as the release.

**DeepSeek as simulated users.** Besides gpt-5-nano, the social and chatbot users can be DeepSeek V4.1 Flash served through [OpenRouter](https://openrouter.ai), with every request pinned to one upstream provider so that runs never mix quantizations (`judge_provider_name=openrouter/morph/fp8 judge_model_name=deepseek/deepseek-v4.1-flash`, with `OPENROUTER_API_KEY` set). Violations are always labeled by gpt-5-mini at low effort, the paper's labeler.

Serve both agent models and the verifier, each on its own port:

```bash
AGENT_PORT=8001 ./run/serve_models.sh qwen-80b-20k     # finetuned agents (4 GPUs)
AGENT_PORT=8002 ./run/serve_models.sh qwen-80b-think   # prompted agents (4 GPUs)
./run/serve_models.sh verifier                         # Qwen3-0.6B verifier on port 8099 (1 GPU)
```

Then run the whole table, or one environment:

```bash
./run/reproduce_main_table.sh                  # gpt-5-nano as the simulated users
USERS=deepseek ./run/reproduce_main_table.sh   # DeepSeek V4.1 Flash as the simulated users
ENVS=social ./run/reproduce_main_table.sh      # the social column only
```

Each step is `./run/run_coupled.sh <env> <agent config>`: one coupled run per cell and seed (seeds 0-7; social and chatbot with 16, 32 and 64 agents and social top-k 4, 8 and 16; fishing with 4 to 20 agents), followed by gpt-5-mini labels. Runs resume after an interruption. `scripts/main_table.py` turns the labeled runs into the table, with the normalized AUC of every mechanism and setting and the verifier's token cost relative to Checked:

```bash
python scripts/main_table.py results_dir=results/coupled users=gpt-5-nano cells=all
```

The labeled transcripts of our runs are on [Hugging Face](https://huggingface.co/datasets/helloelwin/norm-enforcement-transcripts), so `scripts/main_table.py` can rebuild the website's table from them without running any simulations. The dataset card shows how.


## Repo layout

```
norm_enforcement/
├── core/           # environment base class, agents, actions, mechanisms, judges, violation evaluator
├── environments/   # social, chatbot, fishing
├── prompts/        # environment instructions, norms, judge and report prompts
├── llms/           # async OpenAI, vLLM and OpenRouter providers; the call memo shared by coupled arms
├── evaluation/     # post-hoc LLM labeling of norm violations (evaluate_coupled labels coupled runs)
└── coupled.py      # coupled runs: all five mechanisms of a cell and seed in one simulation
configs/            # agent population specs (YAML)
run/                # experiment sweep + model serving scripts (run_coupled.sh, reproduce_main_table.sh for the main table)
scripts/            # figure scripts, shared metric libraries, main_table.py
tests/              # CPU tests with a fake model: PYTHONPATH=. python tests/test_coupled.py
data/               # runtime datasets (see data/README.md for provenance/licenses)
```


## Finetuned models

The π_good / π_bad LoRA adapters are trained on the [emergent-misalignment persona-features data](https://github.com/openai/emergent-misalignment-persona-features) ("correct" vs "incorrect/insecure" completions, mixed across domains; see the paper for details). Adapters: `helloelwin/{qwen-80b,oss-120b}-mixed-{good,bad}-{10k,20k}` on HuggingFace.


## License

The code is released under the [MIT License](LICENSE). The files in `data/` keep their own licenses, listed in [data/README.md](data/README.md), and the transcripts on Hugging Face are released under CC BY 4.0.
