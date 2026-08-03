#!/usr/bin/env bash
# Serve the agent models (and the verifier) with vLLM for the main experiments.
#
# The YAML configs in configs/ reference agents by their *served model alias*
# (e.g. "qwen-80b-good-20k"); this script maps each alias to the corresponding
# public LoRA adapter on the HuggingFace Hub via --lora-modules.
#
# Usage:
#   ./run/serve_models.sh qwen-80b-20k     # finetuned Qwen3-Next-80B agents (port 8001)
#   ./run/serve_models.sh qwen-80b-think   # prompted Qwen3-Next-80B-Thinking agents (port 8001)
#   ./run/serve_models.sh oss-120b-20k     # finetuned gpt-oss-120b agents (port 8001)
#   ./run/serve_models.sh oss-120b-10k     # finetuned gpt-oss-120b agents, 10k data (port 8001)
#   ./run/serve_models.sh verifier         # Qwen3-0.6B verifier, as in the paper (port 8099)
#
# Thinking models (Qwen3-Next-80B-Thinking, the Qwen3 verifier) are served with
# a reasoning parser so their chain of thought is returned separately and never
# enters the parsed actions, agent context, or verifier input.
#
# GPU requirements: the 80B/120B agent servers use tensor-parallel-size 4
# (4x80GB); the verifier fits on a single GPU. Adjust CUDA_VISIBLE_DEVICES,
# VERIFIER_MODEL, and ports for your setup.
#
# MAX_NUM_SEQS sets how many requests an agent server batches at once. It only
# affects speed, not how replies are sampled. The paper's runs used 16; the
# default of 64 runs the Thinking agents about 2.5 to 3 times faster. Set
# MAX_NUM_SEQS=16 to reproduce the paper's serving flags exactly.
set -euo pipefail

TARGET="${1:?Usage: $0 <qwen-80b-20k|qwen-80b-think|oss-120b-20k|oss-120b-10k|verifier>}"

MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
AGENT_PORT="${AGENT_PORT:-8001}"
VERIFIER_PORT="${VERIFIER_PORT:-8099}"
VERIFIER_MODEL="${VERIFIER_MODEL:-Qwen/Qwen3-0.6B}"

case "$TARGET" in
  qwen-80b-20k)
    vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
      --enable-lora \
      --max-lora-rank 32 \
      --tensor-parallel-size 4 \
      --lora-modules \
        qwen-80b-good-20k="helloelwin/qwen-80b-mixed-good-20k" \
        qwen-80b-bad-20k="helloelwin/qwen-80b-mixed-bad-20k" \
      --max-model-len 65536 \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --port "$AGENT_PORT"
    ;;
  qwen-80b-think)
    # Prompted (non-finetuned) agents: the YAML configs reference the base model directly.
    vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking \
      --tensor-parallel-size 4 \
      --reasoning-parser deepseek_r1 \
      --max-model-len 65536 \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --port "$AGENT_PORT"
    ;;
  oss-120b-20k)
    vllm serve openai/gpt-oss-120b \
      --enable-lora \
      --max-lora-rank 32 \
      --tensor-parallel-size 4 \
      --lora-modules \
        oss-120b-good-20k="helloelwin/oss-120b-mixed-good-20k" \
        oss-120b-bad-20k="helloelwin/oss-120b-mixed-bad-20k" \
      --max-model-len 65536 \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --port "$AGENT_PORT"
    ;;
  oss-120b-10k)
    vllm serve openai/gpt-oss-120b \
      --enable-lora \
      --max-lora-rank 32 \
      --tensor-parallel-size 4 \
      --lora-modules \
        oss-120b-good-10k="helloelwin/oss-120b-mixed-good-10k" \
        oss-120b-bad-10k="helloelwin/oss-120b-mixed-bad-10k" \
      --max-model-len 65536 \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --port "$AGENT_PORT"
    ;;
  verifier)
    vllm serve "$VERIFIER_MODEL" --reasoning-parser deepseek_r1 --port "$VERIFIER_PORT"
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Options: qwen-80b-20k, qwen-80b-think, oss-120b-20k, oss-120b-10k, verifier"
    exit 1
    ;;
esac
