# Project website

A static site with no build step: `index.html`, `style.css`, `app.js`, and `data.js`. `data.js` is generated from the
experiment transcripts; don't edit it by hand.

Preview locally with `python3 -m http.server -d web 8000` and open <http://localhost:8000>.

## Where the numbers come from

Every chart and table comes from an independent reproduction of the paper's experiment grid with this repository's
code, run in October 2026, so the numbers don't match the paper exactly.

| | |
|---|---|
| Settings | social, chatbot and fishing; finetuned, prompted, abuse-prompted and explicit-abuse-prompted bad agents; Naive, Checked, Backfire, RepVote and EscRepVote |
| Agent setups | social: 16, 32 and 64 agents, each with 4, 8 or 16 top posts shown; chatbot: 16, 32 and 64 agents; fishing: 4, 8, 12, 16 and 20 agents (population 3,000, regrowth 0.3). Half the agents are good, half bad. |
| Seeds | 0 to 7 in every setup; with DeepSeek users, also 8 to 15 for social with finetuned agents |
| Agents | Qwen3-Next-80B-A3B: Instruct with the good and bad LoRA adapters for finetuned agents, Thinking for prompted agents (vLLM, as in the paper) |
| Verifier | Qwen3-0.6B (vLLM, as in the paper) |
| Simulated users | gpt-5-nano (the default) or DeepSeek V4.1 Flash through OpenRouter, pinned to one provider, reasoning off; a toggle on the page switches between them. gpt-5-nano gets 4,096 output tokens per social vote. The paper used gpt-5-mini (social) and gpt-5-nano (chatbot). Fishing has no users. |
| Violation labels | gpt-5-mini at low reasoning effort, the paper's labeler |
| Runs | 1,000 runs, 5,000 simulations: social 288 with gpt-5-nano users and 360 with DeepSeek, chatbot 96 with each, fishing 160. In each run the five mechanisms of one setup and seed ran side by side, sharing model calls and random draws until their histories diverged. |

The runs used this repository's code: the paper's release plus the side-by-side runner and the OpenRouter client.
The gpt-5-nano social runs capped each vote at 4,096 output tokens, today's default (`judge_max_tokens`). The DeepSeek social runs used the release's 1,024, at which DeepSeek lost fewer than 1 vote in
100,000. Chatbot votes are capped at 1,024 in every run.

To rerun everything, serve the models and run `USERS=gpt-5-nano ./run/reproduce_main_table.sh` or
`USERS=deepseek ./run/reproduce_main_table.sh` (see the main README). The page's DeepSeek column for social with
finetuned agents also has seeds 8 to 15 (`SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"`). `scripts/main_table.py` with
its defaults turns the labeled runs into the table on this page.

## How the numbers are computed

- **TPR and FPR.** These use `scripts/roc_metrics.py` (`compute_tpr_fpr`). An agent counts as a violator if the labeler
  flags at least one of its actions. TPR is the share of violators removed and FPR the share of compliant agents
  removed. Curves are means over runs after rounds 1 to 15.
- **Normalized AUC.** This is `_normalized_auc_ratio` from `scripts/plot_mechanism_roc.py`, applied to the mean
  trajectory pooled over all setups and seeds of a column, as in the paper's main table. The 95% intervals come from
  10,000 bootstrap resamples of seeds within setups.
- **Verifier cost.** This counts Qwen3-4B tokens in verifier inputs and outputs for every report in rounds 1 to 15. It
  averages them per mechanism and divides by Checked's average.
- **Abuse figure.** This uses `scripts/report_metrics.py` on finetuned agents under the naive mechanism. It gives
  accurate and false reports per agent-round by model, and the share of agents still active, split by whether the
  labeler ever flagged them.
- **Rounds.** Everything uses each run's first 15 rounds, violation labels included. Our chatbot runs lasted 20 rounds
  (the run scripts now stop at 15). Agents never see the round limit, so a run's first 15 rounds are what a 15-round run
  gives.

## Deploying with GitHub Pages

GitHub Pages serves a branch from its root or from `docs/` only. To serve this folder, either add a Pages workflow
that uploads `web/` (Settings → Pages → Source: GitHub Actions), or point the root `index.html` at `web/`.
