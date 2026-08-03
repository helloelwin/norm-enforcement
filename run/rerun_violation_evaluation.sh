#!/usr/bin/env bash
# Re-run the LLM violation evaluation over existing transcripts (overwrites old labels).
#
# Usage:
#   ./run/rerun_violation_evaluation.sh <ENV> [AGENT_NAME] [MECHANISM_TYPE]
# Example:
#   ./run/rerun_violation_evaluation.sh social 1x1-80b-good-20k-bad-20k naive
#
# Environment overrides: RESULTS_DIR, SEEDS, VIOLATION_EVALUATOR
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-results/social}"

SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7})

WINDOW=5

ENV="${1:?Usage: $0 <ENV> [AGENT_NAME] [MECHANISM_TYPE]}"
AGENT_NAME="${2:-1x1-80b-good-20k-bad-20k}"
MECHANISM_TYPE="${3:-naive}"
VIOLATION_EVALUATOR="${VIOLATION_EVALUATOR:-gpt-5-mini}"

# Repvote mechanisms embed hyperparameters between the mechanism name and _shutdown
case "$MECHANISM_TYPE" in
  *repvote*) MECH_GLOB="_mechanism=${MECHANISM_TYPE}_k=*" ;;
  *)         MECH_GLOB="_mechanism=${MECHANISM_TYPE}_shutdown*" ;;
esac

for seed in "${SEEDS[@]}"; do
  echo "Evaluating violations with seed=$seed"
  python -m norm_enforcement.evaluation.evaluate_violation \
    results_dir="$RESULTS_DIR" \
    force_overwrite=True \
    model_name="$VIOLATION_EVALUATOR" \
    transcript_pattern="${ENV}_agent=${AGENT_NAME}_*_w=${WINDOW}${MECH_GLOB}_seed=${seed}.json" &
done
wait
