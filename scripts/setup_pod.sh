#!/bin/bash
# Setup script for prompt injection adversarial training on a fresh Ubuntu pod
# Usage: curl -sSL <url> | bash
#    or: bash setup_pod.sh

set -e  # Exit on error

echo "=========================================="
echo "Ludic Prompt Injection Training Setup"
echo "=========================================="

# --- 1. System dependencies ---
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update
apt-get install -y git curl wget build-essential

# --- 2. Install uv (fast Python package manager) ---
echo ""
echo "[2/6] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
source $HOME/.local/bin/env 2>/dev/null || true

# --- 3. Clone ludic repository ---
echo ""
echo "[3/6] Cloning ludic repository..."
cd /root
if [ -d "ludic" ]; then
    echo "ludic directory exists, pulling latest..."
    cd ludic
    git pull
else
    git clone https://github.com/hallerite/ludic.git
    cd ludic
fi

# Checkout the prompt injection branch if it exists
git checkout env/prompt-injection-minimax 2>/dev/null || echo "Using current branch"

# --- 4. Create venv and install dependencies ---
echo ""
echo "[4/6] Creating Python environment and installing dependencies..."
uv venv --python 3.12
source .venv/bin/activate

# Install ludic with prompt-injection extras
uv pip install -e ".[prompt-injection]"

# Install additional dependencies for training
uv pip install transformers accelerate

# --- 5. Set up HuggingFace cache (optional, for faster model loading) ---
echo ""
echo "[5/6] Setting up environment variables..."
export HF_HOME=/root/.cache/huggingface
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p $HF_HOME

# --- 6. Download model (optional, can skip if you want to do it later) ---
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. Activate the environment:"
echo "   cd /root/ludic && source .venv/bin/activate"
echo ""
echo "2. Start vLLM server (in a tmux/screen session):"
echo "   python -m vllm.entrypoints.openai.api_server \\"
echo "       --model Qwen/Qwen2.5-7B-Instruct \\"
echo "       --port 8000 \\"
echo "       --enable-lora \\"
echo "       --max-lora-rank 16 \\"
echo "       --gpu-memory-utilization 0.85"
echo ""
echo "3. In another terminal, run training:"
echo "   cd /root/ludic && source .venv/bin/activate"
echo "   python examples/prompt_injection/train_adversarial.py \\"
echo "       --difficulty easy \\"
echo "       --train-steps 50"
echo ""
echo "For a quick test without sandbox:"
echo "   python -c 'from environments.prompt_injection import PromptInjectionEnv; print(\"Import OK!\")'"
echo ""
