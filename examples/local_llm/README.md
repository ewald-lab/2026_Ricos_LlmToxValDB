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

Common overrides via environment variables (see [Recommendations: Normal vs. Thinking Models](#recommendations-normal-vs-thinking-models) before deploying with thinking mode, which needs larger `MAX_MODEL_LEN` and benefits from multi-GPU `TP_SIZE`):

```bash
PORT=8000 TP_SIZE=1 MAX_MODEL_LEN=16384 ./serve_vllm.sh ./models/Qwen3.6-35B-A3B-FP8
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

---

## Recommendations: Normal vs. Thinking Models

HiPO supports both **standard (non-thinking)** and **reasoning (`thinking`)** configurations against the same served model checkpoint (e.g., `Qwen/Qwen3.6-35B-A3B-FP8` or `Qwen/Qwen3.6-27B-FP8`), toggled client-side via `extra_body.chat_template_kwargs.enable_thinking` in [`HiPO/config/models/`](../../HiPO/config/models/).

| Setting | Normal (Non-Thinking) | Thinking Mode |
| :--- | :--- | :--- |
| **Client preset example** | [`Qwen3.6-35B-A3B-FP8.json`](../../HiPO/config/models/Qwen3.6-35B-A3B-FP8.json) | [`Qwen3.6-35B-A3B-FP8-thinking.json`](../../HiPO/config/models/Qwen3.6-35B-A3B-FP8-thinking.json) / [`Qwen3.6-27B-FP8-thinking.json`](../../HiPO/config/models/Qwen3.6-27B-FP8-thinking.json) |
| **vLLM `MAX_MODEL_LEN`** | `8192` min / `16384` recommended (with `--ontology_definitions`) | `32768` min / `65536` recommended |
| **vLLM `--reasoning-parser`** | `qwen3` (default in [`serve_vllm.sh`](serve_vllm.sh)) | `qwen3` (required to separate `reasoning` / `reasoning_content` cleanly) |
| **Client `max_tokens`** | `4096` | `8192`–`16384` (shared between reasoning trace and final JSON) |
| **Sampling (`temp` / `top_p` / `presence_penalty`)** | `0.1` / `0.8` / `1.5` | `0.6` / `0.95` / `1.0` |
| **Client `timeout` & `max_concurrency`** | `300`s / `100` | `600`s / `64` (lower on small single-GPU setups) |
| **Overall runtime** | Faster baseline | **Significantly slower** (more generated tokens per step) |

### 1. Context Window (`MAX_MODEL_LEN`) vs. `max_tokens`

In vLLM, `--max-model-len` is the **combined ceiling** for input + output (`prompt_tokens + max_tokens <= MAX_MODEL_LEN`):

- **Why `8192` is not enough for thinking mode**: `serve_vllm.sh` defaults `MAX_MODEL_LEN` to `8192` for quick smoke tests, whereas thinking presets request `max_tokens: 8192` (or `16384` for `27B-thinking`) because reasoning tokens and the final JSON answer share the same `max_tokens` budget. With typical HiPO prompts (system instructions, Pydantic schema, RAG candidates, hierarchy paths, and `--ontology_definitions`), `prompt_tokens + max_tokens` exceeds `8192` and triggers context-length errors (forcing HiPO's retry logic to clamp `max_tokens` tightly).
- **Recommended deployment lengths**:
    - **Normal models**: `MAX_MODEL_LEN=8192` works for short prompts, while `16384` gives comfortable headroom when `--ontology_definitions` is enabled.
    - **Thinking models**: deploy with at least `MAX_MODEL_LEN=32768`, and prefer **`MAX_MODEL_LEN=65536`** when GPU VRAM allows so long reasoning traces (often tens of thousands of characters) and rich ontology-definition prompts do not hit context limits or truncate outputs (`finish_reason="length"`).
- **Reasoning parser & traces**: keep `--reasoning-parser qwen3` enabled in vLLM for Qwen3/3.6 models. HiPO extracts both separated `reasoning`/`reasoning_content` fields and inline `<think>...</think>` blocks, storing them under `"thinking"` in `--reasoning_file` and recovering automatically if occasional calls truncate or loop.

### 2. Speed and Multi-GPU Scaling

- **Expect slower execution in thinking mode**: generating internal reasoning before every JSON verification increases generated tokens per request substantially, reduces effective batch concurrency, and requires higher per-request timeouts (`timeout: 600` vs `300`). Reserve thinking mode for harder disambiguation or higher-accuracy runs rather than quick exploratory passes.
- **Deploying across multiple GPUs speeds up vLLM significantly**:
    - Running vLLM with **`TP_SIZE=2` (or higher)** via `--tensor-parallel-size` spreads model weights and increases aggregate memory bandwidth and available VRAM for the KV cache.
    - More KV cache headroom lets vLLM keep `MAX_MODEL_LEN=65536` active across high concurrent batches (`--descriptions_concurrency 64` / `max_concurrency: 64`) without request preemption or queueing bottlenecks.
    - On a **single GPU** with thinking enabled and a large context window (`32768`–`65536`), reduce `max_concurrency` in your model JSON (and `--descriptions_concurrency` in `hipo.py`, e.g., to `16`–`32`) if KV cache usage nears 100%.

### 3. Example Launch Commands

**Normal mode (single GPU, faster throughput):**

```bash
MAX_MODEL_LEN=16384 TP_SIZE=1 ./serve_vllm.sh Qwen/Qwen3.6-35B-A3B-FP8

./run_hipo_example.sh \
    --base-url http://localhost:8000/v1 \
    --model-config ../../HiPO/config/models/Qwen3.6-35B-A3B-FP8.json \
    --ontology mp \
    --ontology-db /path/to/mp.db
```

**Thinking mode (multi-GPU recommended for speed and `65k` KV cache):**

```bash
MAX_MODEL_LEN=65536 TP_SIZE=2 ./serve_vllm.sh Qwen/Qwen3.6-35B-A3B-FP8

./run_hipo_example.sh \
    --base-url http://localhost:8000/v1 \
    --model-config ../../HiPO/config/models/Qwen3.6-35B-A3B-FP8-thinking.json \
    --ontology mp \
    --ontology-db /path/to/mp.db
```

> [!NOTE]
> The `-thinking.json` presets keep `"model"` set to the served checkpoint name (e.g., `"Qwen3.6-35B-A3B-FP8"`), so you can switch between normal and thinking runs without redeploying the server—provided the server was started with enough `MAX_MODEL_LEN` (and `--reasoning-parser qwen3`).

