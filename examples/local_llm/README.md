# Local LLM Deployment with vLLM

**HiPO** communicates with any OpenAI-compatible chat completions endpoint via [`HiPO/model/llm.py`](../../HiPO/model/llm.py) (`get_model`). You can point `--base_url` and `--model_config` to a hosted API or run open-weights models locally using **vLLM**.

Pre-configured client JSONs for all benchmarked models live in [`HiPO/config/models/`](../../HiPO/config/models), and the default production template is [`HiPO/config/llm.json`](../../HiPO/config/llm.json).

> [!NOTE]
> Serving models locally with vLLM requires **Linux + an NVIDIA GPU with CUDA**. On GPU-limited machines you can use a smaller open-weights model (such as `openai/gpt-oss-20b` or `Qwen/Qwen3-8B`), or point `--model_config` directly to a hosted API provider (`openai`, `anthropic`, etc.).

---

## Step 1: Download / Install vLLM

Create an isolated virtual environment and install `vllm` together with `huggingface_hub` using [uv](https://docs.astral.sh/uv/):

```bash
cd examples/local_llm
chmod +x install_vllm.sh serve_vllm.sh run_hipo_example.sh
./install_vllm.sh
```

Or run the commands manually:

```bash
uv venv env-vllm --python=3.12
source env-vllm/bin/activate
uv pip install --no-cache -U vllm huggingface_hub
```

On HPC clusters with limited `$HOME` disk quotas, redirect caches before installing or downloading models:

```bash
export CACHE_DIR=/path/to/scratch/hf_cache
export HF_HOME="$CACHE_DIR/hf"
export UV_CACHE_DIR="$CACHE_DIR/uv"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
```

---

## Step 2: Download Model Weights (Optional)

`vllm serve` automatically downloads models from Hugging Face on first startup. If compute nodes lack internet access or you want shared local weights, pre-download a model into a local folder first:

```bash
source env-vllm/bin/activate
export HF_TOKEN="<your_hf_token>"   # Only required for gated models

hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ./models/Qwen3.6-35B-A3B-FP8
```

For smaller single-GPU setups (~16–24 GB VRAM), you can test with a compact model such as `Qwen/Qwen3-8B` or `openai/gpt-oss-20b`.

---

## Step 3: Start the vLLM OpenAI-Compatible Server

Launch the server pointing to either a Hugging Face model ID or your downloaded local directory:

```bash
./serve_vllm.sh Qwen/Qwen3.6-35B-A3B-FP8
```

Common overrides via environment variables:

```bash
PORT=8000 TP_SIZE=1 MAX_MODEL_LEN=8192 ./serve_vllm.sh ./models/Qwen3.6-35B-A3B-FP8
```

<details>
<summary>Optional: Example portable SLURM job wrapper</summary>

```bash
#!/bin/bash
#SBATCH --job-name=hipo_vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail
export HOST=0.0.0.0
export PORT=8000
export TP_SIZE=1
./serve_vllm.sh /path/to/models/Qwen3.6-35B-A3B-FP8
```

</details>

Once the server prints `Application startup complete`, verify the endpoint from the client machine (or another terminal):

```bash
curl -sf http://localhost:8000/v1/models
```

---

## Step 4: Connect and Run HiPO

### 4.1 Verify Client Connectivity

Ensure your HiPO environment (`toxvaldb/01.hipo/env-text-to-onto`) is active, then verify both the HTTP endpoint and LangChain model initialization without needing an ontology DB yet:

```bash
./run_hipo_example.sh --check-only --base-url http://localhost:8000/v1
```

### 4.2 Run HiPO on the Toy Input

Once you have built an ontology embedding database with [`01.process_ontology.py`](../../toxvaldb/00.data_preparation/modules/02.prepare_ontologies/scripts/01.process_ontology.py), run the 5-row example in [`sample_effects.tsv`](sample_effects.tsv):

```bash
./run_hipo_example.sh \
    --base-url http://localhost:8000/v1 \
    --model-config ../../HiPO/config/models/Qwen3.6-35B-A3B-FP8.json \
    --ontology mp \
    --ontology-db /path/to/mp.db
```

> [!TIP]
> The `"model"` field in your `--model_config` JSON (or the `--model` CLI flag) must match the server's `--served-model-name`. When testing custom models via `./serve_vllm.sh Qwen/Qwen3-8B`, pass `--model Qwen3-8B` or create a custom JSON in `HiPO/config/models/`.

