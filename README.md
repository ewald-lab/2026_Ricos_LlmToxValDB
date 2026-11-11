<p align="center">
  <img src="assets/HiPO-logo.png" alt="HiPO Logo" width="360" />
</p>

# Hierarchy-aware Phenotype-to-Ontology mapping

This repository provides **HiPO** (Hierarchy-aware Phenotype-to-Ontology mapping), a framework designed to translate free-text phenotypic descriptions into standardized biomedical ontology terms.

By integrating dense retrieval-augmented generation (RAG) with large language model (LLM) hierarchy validation, HiPO identifies appropriate ontology concepts while preserving parent-child relationships and semantic depth.

This repository presents the reference application of HiPO to the curation of the EPA Toxicity Value Database ([ToxValDB](https://www.epa.gov/chemical-research/toxicity-value-database-toxvaldb)), standardizing thousands of unstructured chemical toxicity study outcomes into machine-readable ontological annotations to support downstream computational toxicology, chemical grouping, and hazard assessment.

## Repository Overview

- **`HiPO/`**: Core hierarchy-aware phenotype-to-ontology mapping framework and reusable Python module, including default and per-model client configurations under [`HiPO/config/`](HiPO/config/).
- **`examples/local_llm/`**: Portable guide and helper scripts to install vLLM, download model weights, launch a local OpenAI-compatible inference server, and run HiPO ([Local LLM Deployment Guide](examples/local_llm/README.md)).
- **`toxvaldb/`**: Reference implementation, data preparation, expert annotation benchmark, and evaluation pipelines applied to ToxValDB. Detailed documentation and submodule setup instructions can be found in the [ToxValDB Pipeline README](toxvaldb/README.md).

## Prerequisites

- **Python & Package Manager**: Python 3.12 and [uv](https://docs.astral.sh/uv/) (see [ToxValDB Environment Setup](toxvaldb/README.md#environment-setup) for stage-specific virtual environments).
- **Two Separate Compute Roles**: HiPO queries an **OpenAI-compatible LLM endpoint** while running **dense embeddings and reranking locally in PyTorch** (`bfloat16` on CUDA, falling back to `float32` on CPU if no GPU is available):

| Component | Default Model(s) | Execution Role | Notes & Alternatives |
| :--- | :--- | :--- | :--- |
| **LLM** | `Qwen/Qwen3.6-35B-A3B-FP8` ([`llm.json`](HiPO/config/llm.json)) | External or local OpenAI-compatible endpoint (e.g., vLLM) | Served on `1x` 80 GB GPU (`A100`/`H200`) for FP8; swap in smaller models (e.g., `Qwen/Qwen3-8B`, `openai/gpt-oss-20b`) or cloud APIs (`model_provider` in `llm.json`). Other benchmark model configs live in [`HiPO/config/models/`](HiPO/config/models/). |
| **Embeddings** | `Qwen/Qwen3-Embedding-8B` ([`choice_rag.json`](HiPO/config/choice_rag.json)) / `tencent/KaLM-Embedding-Gemma3-12B-2511` | Local PyTorch inside `HiPO` | Runs on the client node GPU (`1x A100` in full ToxValDB production/evaluation jobs). |
| **Reranker** | `Qwen/Qwen3-Reranker-0.6B` ([`choice_rag.json`](HiPO/config/choice_rag.json)) / `Qwen/Qwen3-Reranker-8B` ([`rag.json`](HiPO/config/rag.json)) | Local PyTorch inside `HiPO` | `0.6B` keeps local VRAM modest; `8B` is available for heavier retrieval configurations. |

To set up vLLM locally and test connectivity with HiPO, follow the [Local LLM Deployment Guide](examples/local_llm/README.md).

## Citation

```bibtex
@article{saurina2026automated,
  title={Automated Ontology Annotation of Phenotypic Descriptions using Large Language Models (LLMs)},
  author={Saurina-i-Ricos, Joan and Puig-Barbe, Aleix and Carli, Francesco and Johannesson, Bjarki and Chouhan, Bhavik and McLaughlin, James and Ewald, Jessica D.},
  year={2026},
  note={In preparation}
}
```
