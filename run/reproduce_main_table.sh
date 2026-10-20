#!/usr/bin/env bash
# Reproduce the paper's main table with coupled runs: social, chatbot and fishing, each with the four agent populations,
# every cell on seeds 0-7, all five mechanisms in each run, labeled by gpt-5-mini; then the table in the paper's format
# (scripts/main_table.py).
#
# Serve the models first, each on its own port:
#   AGENT_PORT=8001 ./run/serve_models.sh qwen-80b-20k     # finetuned agents (F), 4 GPUs
#   AGENT_PORT=8002 ./run/serve_models.sh qwen-80b-think   # prompted agents (P, AP, EAP), 4 GPUs
#   ./run/serve_models.sh verifier                         # Qwen3-0.6B verifier on port 8099, 1 GPU
#
# Usage:
#   ./run/reproduce_main_table.sh                  # simulated users: gpt-5-nano, as in the paper
#   USERS=deepseek ./run/reproduce_main_table.sh   # simulated users: DeepSeek V4.1 Flash through OpenRouter
#   ENVS=social ./run/reproduce_main_table.sh      # the social column only
#
# Environment overrides: ENVS (default social chatbot fishing), CONDITIONS (F P AP EAP), FINETUNED_PORT (8001),
# PROMPTED_PORT (8002), and run_coupled.sh's RESULTS_DIR, SEEDS, PARALLEL, VERIFIER_VLLM_PORT, OPENROUTER_PROVIDER.
set -euo pipefail
cd "$(dirname "$0")/.."

ENVS=(${ENVS:-social chatbot fishing})
CONDITIONS=(${CONDITIONS:-F P AP EAP})
export RESULTS_DIR="${RESULTS_DIR:-results/coupled}"
export USERS="${USERS:-gpt-5-nano}"
declare -A CONFIG=(
  [F]=configs/1x1-80b-good-20k-bad-20k.yaml
  [P]=configs/1x1-80b-think-good-prompted-bad-prompted.yaml
  [AP]=configs/1x1-80b-think-good-prompted-bad-abuseprompted.yaml
  [EAP]=configs/1x1-80b-think-good-prompted-bad-explicitabuseprompted.yaml
)

for env in "${ENVS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    if [ "$cond" = F ]; then port="${FINETUNED_PORT:-8001}"; else port="${PROMPTED_PORT:-8002}"; fi
    echo "== $env, $cond ($USERS users)"
    AGENT_PORT="$port" ./run/run_coupled.sh "$env" "${CONFIG[$cond]}"
  done
done

# The table pools every cell; cells=paper leaves out the 64-agent prompted cells, which the paper's main table did not run.
python scripts/main_table.py results_dir="$RESULTS_DIR" users="$USERS" cells=all
