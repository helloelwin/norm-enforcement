#!/usr/bin/env bash
# Coupled runs for one environment and one agent population: each cell and seed is one simulation in which all five
# mechanisms run side by side on shared model calls (python -m norm_enforcement.coupled), so their differences come from
# the mechanisms rather than from sampling luck. The runs are then labeled by gpt-5-mini at low effort, the paper's
# violation labeler.
#
# Usage:
#   ./run/run_coupled.sh <social|chatbot|fishing> <AGENT_CONFIG_PATH>
# Examples:
#   ./run/run_coupled.sh social configs/1x1-80b-good-20k-bad-20k.yaml
#   USERS=deepseek ./run/run_coupled.sh chatbot configs/1x1-80b-think-good-prompted-bad-prompted.yaml
#
# Environment overrides:
#   USERS               simulated users of social and chatbot: gpt-5-nano (default, needs OPENAI_API_KEY) or deepseek
#                       (DeepSeek V4.1 Flash through OpenRouter, pinned to OPENROUTER_PROVIDER, default morph/fp8; needs
#                       OPENROUTER_API_KEY). Labels always come from gpt-5-mini (OPENAI_API_KEY).
#   RESULTS_DIR         default results/coupled
#   SEEDS               default 0 1 2 3 4 5 6 7
#   AGENT_COUNTS        default 16 32 64 (social, chatbot) or 4 8 12 16 20 (fishing)
#   TOP_COUNTS          social only, default 4 8 16
#   AGENT_PORT          the agent server's port, if not the config's (a copy of the config with this port is used)
#   VERIFIER_VLLM_PORT  default 8099
#   PARALLEL            simulations at a time, default 4
set -euo pipefail

# Simulations resume from their memo and labeling skips finished runs, so a failed attempt (a transient API error, a
# server restart) is retried. Anything that still fails after three attempts makes this script exit non-zero.
retry() {
  local attempt
  for attempt in 1 2 3; do
    "$@" && return 0
    echo "[retry] attempt $attempt of 3 failed: $1 $2 $3 ..." >&2
    sleep 10
  done
  return 1
}

ENV_NAME="${1:?Usage: $0 <social|chatbot|fishing> <AGENT_CONFIG_PATH>}"
AGENT_CONFIG_PATH="${2:?Usage: $0 <social|chatbot|fishing> <AGENT_CONFIG_PATH>}"
RESULTS_DIR="${RESULTS_DIR:-results/coupled}"
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7})
case "$ENV_NAME" in
  social|chatbot) AGENT_COUNTS=(${AGENT_COUNTS:-16 32 64}) ;;
  fishing)        AGENT_COUNTS=(${AGENT_COUNTS:-4 8 12 16 20}) ;;
  *) echo "unknown environment: $ENV_NAME" >&2; exit 1 ;;
esac
if [ "$ENV_NAME" = social ]; then TOP_COUNTS=(${TOP_COUNTS:-4 8 16}); else TOP_COUNTS=(-); fi
USERS="${USERS:-gpt-5-nano}"
PARALLEL="${PARALLEL:-4}"
VERIFIER_VLLM_PORT="${VERIFIER_VLLM_PORT:-8099}"
MECHANISMS="naive,checked,checked_backfire,checked_repvote,checked_escrepvote"

case "$USERS" in
  gpt-5-nano) JUDGE=(judge_provider_name=openai judge_model_name=gpt-5-nano) ;;
  deepseek)   JUDGE=(judge_provider_name="openrouter/${OPENROUTER_PROVIDER:-morph/fp8}" judge_model_name=deepseek/deepseek-v4.1-flash) ;;
  *) echo "USERS must be gpt-5-nano or deepseek" >&2; exit 1 ;;
esac

# The agent config, moved to AGENT_PORT if given. The copy keeps the file name, which names the runs.
CONFIG="$AGENT_CONFIG_PATH"
if [ -n "${AGENT_PORT:-}" ]; then
  mkdir -p "$RESULTS_DIR/configs/port$AGENT_PORT"
  CONFIG="$RESULTS_DIR/configs/port$AGENT_PORT/$(basename "$AGENT_CONFIG_PATH")"
  sed "s/vllm_port: *[0-9][0-9]*/vllm_port: $AGENT_PORT/" "$AGENT_CONFIG_PATH" > "$CONFIG"
fi

ARGS=(env="$ENV_NAME" results_dir="$RESULTS_DIR" mechanisms="$MECHANISMS" seeded_sampling=false
      verifier_provider_name=vllm verifier_model_name=Qwen/Qwen3-0.6B verifier_vllm_port="$VERIFIER_VLLM_PORT"
      report_shutdown_prob=1.0 agent_context_window_len=5 agent_config_path="$CONFIG")
case "$ENV_NAME" in
  social)
    # each persona vote may use 4,096 output tokens (judge_max_tokens, the default): at the release's 1,024, gpt-5-nano
    # reasons past the limit on long social prompts and loses the vote
    ARGS+=(max_rounds=15 max_personas=64 dataset_path=data/sales-train.json persona_path=data/judge-personas.json "${JUDGE[@]}") ;;
  chatbot)
    ARGS+=(max_rounds=15 max_personas=5 requests_path=data/woh5k.json persona_path=data/judge-personas.json "${JUDGE[@]}") ;;
  fishing)
    ARGS+=(max_rounds=15) ;;
esac

FAILED=0
running=0
for seed in "${SEEDS[@]}"; do
  for num_agents in "${AGENT_COUNTS[@]}"; do
    for num_top_posts in "${TOP_COUNTS[@]}"; do
      CELL=(num_agents="$num_agents" seed="$seed")
      if [ "$num_top_posts" != - ]; then CELL+=(num_top_posts="$num_top_posts"); fi
      echo "coupled run: $ENV_NAME $(basename "$CONFIG" .yaml) ${CELL[*]} users=$USERS"
      retry python -m norm_enforcement.coupled "${ARGS[@]}" "${CELL[@]}" &
      running=$((running + 1))
      if [ "$running" -ge "$PARALLEL" ]; then
        wait -n || FAILED=$((FAILED + 1))
        running=$((running - 1))
      fi
    done
  done
done
while [ "$running" -gt 0 ]; do
  wait -n || FAILED=$((FAILED + 1))
  running=$((running - 1))
done

# Labels from gpt-5-mini at low effort. The arms of a run are labeled together, so identical action windows get the same
# label, and strict=true retries a failed call instead of recording it as no violation.
echo "labeling with gpt-5-mini"
retry python -m norm_enforcement.evaluation.evaluate_coupled results_dir="$RESULTS_DIR" \
  summary_pattern="${ENV_NAME}_agent=$(basename "$CONFIG" .yaml)_*" model_name=gpt-5-mini reasoning_level=low strict=true \
  || FAILED=$((FAILED + 1))

if [ "$FAILED" -gt 0 ]; then
  echo "$FAILED step(s) failed after three attempts each" >&2
  exit 1
fi
