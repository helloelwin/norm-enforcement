#!/usr/bin/env bash
# Run the social-media-environment experiment sweep, then evaluate norm violations.
#
# Usage:
#   ./run/run_social.sh <AGENT_CONFIG_PATH> [MECHANISM_TYPE] [VERIFIER_MODEL_NAME] [VERIFIER_VLLM_PORT]
# Example:
#   ./run/run_social.sh configs/1x1-80b-good-20k-bad-20k.yaml checked_escrepvote
#
# Environment overrides: RESULTS_DIR, SEEDS, AGENT_COUNTS, TOP_COUNTS, JUDGE_MODEL, VERIFIER_PROVIDER
set -euo pipefail

# Simulations and labeling both resume from their saved progress, so a failed attempt
# (a transient API error, a crashed client process) is retried. Anything that still fails
# after three attempts makes this script exit non-zero.
retry() {
  local attempt
  for attempt in 1 2 3; do
    "$@" && return 0
    echo "[retry] attempt $attempt of 3 failed: $1 $2 $3 ..." >&2
    sleep 10
  done
  return 1
}
FAILED=0
PIDS=()
wait_all() {
  local pid
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do wait "$pid" || FAILED=$((FAILED + 1)); done
  PIDS=()
}

RESULTS_DIR="${RESULTS_DIR:-results/social}"

SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7})
AGENT_COUNTS=(${AGENT_COUNTS:-16 32})
TOP_COUNTS=(${TOP_COUNTS:-4 8 16})

JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"

WINDOW=5
MAX_ROUNDS=15

AGENT_CONFIG_PATH="${1:?Usage: $0 <AGENT_CONFIG_PATH> [MECHANISM_TYPE] [VERIFIER_MODEL_NAME] [VERIFIER_VLLM_PORT]}"
MECHANISM_TYPE="${2:-naive}"
VERIFIER_MODEL_NAME="${3:-Qwen/Qwen3-0.6B}"
VERIFIER_VLLM_PORT="${4:-8099}"
VERIFIER_PROVIDER="${VERIFIER_PROVIDER:-vllm}"

for seed in "${SEEDS[@]}"; do
  for num_agents in "${AGENT_COUNTS[@]}"; do
    for num_top_posts in "${TOP_COUNTS[@]}"; do
      echo "Running social with num_agents=$num_agents, num_top_posts=$num_top_posts, seed=$seed"
      retry python -m norm_enforcement.environments.social \
        results_dir="$RESULTS_DIR" \
        max_rounds=$MAX_ROUNDS \
        max_personas=64 \
        mechanism_type="$MECHANISM_TYPE" \
        verifier_provider_name="$VERIFIER_PROVIDER" \
        verifier_model_name="$VERIFIER_MODEL_NAME" \
        verifier_vllm_port="$VERIFIER_VLLM_PORT" \
        report_shutdown_prob=1.0 \
        num_agents=$num_agents \
        num_top_posts=$num_top_posts \
        agent_context_window_len=$WINDOW \
        agent_config_path="$AGENT_CONFIG_PATH" \
        judge_provider_name="openai" \
        judge_model_name="$JUDGE_MODEL" \
        seed=$seed &
      PIDS+=($!)
    done
    wait_all
  done
done
wait_all

# Repvote mechanisms embed hyperparameters between the mechanism name and _shutdown
case "$MECHANISM_TYPE" in
  *repvote*) MECH_GLOB="_mechanism=${MECHANISM_TYPE}_k=*" ;;
  *)         MECH_GLOB="_mechanism=${MECHANISM_TYPE}_shutdown*" ;;
esac

ENV=social
for seed in "${SEEDS[@]}"; do
  echo "Evaluating violations with seed=$seed"
  retry python -m norm_enforcement.evaluation.evaluate_violation \
    results_dir="$RESULTS_DIR" \
    transcript_pattern="${ENV}_*_w=${WINDOW}${MECH_GLOB}_seed=${seed}.json" &
  PIDS+=($!)
done
wait_all

if [ "$FAILED" -gt 0 ]; then
  echo "$FAILED job(s) still failed after retries" >&2
  exit 1
fi
