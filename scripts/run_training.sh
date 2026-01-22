#!/bin/bash
# Run adversarial prompt injection training
# This script starts vLLM and runs training in one go
# Usage: bash run_training.sh [--model MODEL] [--difficulty DIFFICULTY]

set -e

# Defaults
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DIFFICULTY="${DIFFICULTY:-easy}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.85}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
MOCK_SANDBOX="${MOCK_SANDBOX:-false}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --difficulty) DIFFICULTY="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu-util) GPU_UTIL="$2"; shift 2 ;;
        --train-steps) TRAIN_STEPS="$2"; shift 2 ;;
        --mock-sandbox) MOCK_SANDBOX="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Prompt Injection Adversarial Training"
echo "=========================================="
echo "Model: $MODEL"
echo "Difficulty: $DIFFICULTY"
echo "Port: $PORT"
echo "GPU Utilization: $GPU_UTIL"
echo "Training Steps: $TRAIN_STEPS"
echo "Mock Sandbox: $MOCK_SANDBOX"
echo "=========================================="

# Ensure we're in the ludic directory
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || {
    echo "Error: Virtual environment not found. Run setup_pod.sh first."
    exit 1
}

# Export env vars
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# Start vLLM server in background
echo ""
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --enable-lora \
    --max-lora-rank 16 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --disable-log-requests \
    > /tmp/vllm.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to start (this may take a few minutes for first run)..."
MAX_WAIT=300  # 5 minutes
WAITED=0
while ! curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "Error: vLLM server died. Check /tmp/vllm.log"
        cat /tmp/vllm.log
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Error: vLLM server didn't start in time. Check /tmp/vllm.log"
        cat /tmp/vllm.log
        exit 1
    fi
    echo "  Still waiting... ($WAITED s)"
done
echo "vLLM server is ready!"

# Run training
echo ""
echo "Starting training..."
MOCK_ARG=""
if [ "$MOCK_SANDBOX" = "true" ]; then
    MOCK_ARG="--mock-sandbox"
fi

python examples/prompt_injection/train_adversarial.py \
    --model "$MODEL" \
    --difficulty "$DIFFICULTY" \
    --train-steps "$TRAIN_STEPS" \
    --agent-port "$PORT" \
    --logger rich \
    $MOCK_ARG

echo ""
echo "Training complete!"
