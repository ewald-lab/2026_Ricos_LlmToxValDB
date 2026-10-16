# ToxValDB Phenotype-to-Ontology Pipeline

This directory contains the end-to-end reference implementation of the **HiPO** framework applied to the curation, standardization, and evaluation of toxicological effect descriptions in **ToxValDB**.

The pipeline is organized into four sequential stages (`00` to `03`), each designed to be modular and reproducible.

---

## Pipeline Architecture

### `00.data_preparation/`
Handles raw data acquisition, curation, guideline standardization, and the creation of biomedical ontology knowledge bases.
- **`00.download_toxvaldb`**: Automated retrieval and extraction of the raw ToxValDB database dumps.
- **`01.preprocess_toxvaldb`**: Cleans, parses, and extracts unique free-text toxicological effect phrases from study records.
- **`02.prepare_ontologies`**: Downloads biomedical ontologies (Mammalian Phenotype Ontology, Human Phenotype Ontology, HPATH, AOP-Wiki, TxPO), formats them as OWL/TTL graphs, and builds dense vector embedding databases (e.g. KaLM) and definition lookup stores.
- **`03.prepare_guidelines`**: Curates and standardizes OECD test guidelines and study endpoints.

---

### `01.hipo/`
Production mapping pipeline matching toxicological effect descriptions to ontology terms via dense retrieval and LLM hierarchy-aware reranking.
- **`00.hipo_toxvaldb`**: Production-scale mapping of ToxValDB effect phrases against candidate ontologies using RAG and LLM validation.
- **`01.hipo_guidelines`**: Applies the mapping pipeline to OECD guideline terms and endpoints to support regulatory read-across.

---

### `02.expert_annotation/`
Curation of human expert benchmark datasets and interactive validation interfaces.
- **`00.create_test`**: Constructs stratified test sets and candidate term pools across ontologies (e.g. MP and HPATH) for domain expert evaluation.
- **`01.create_form`**: Generates standalone, interactive HTML annotation forms with bundled styling and typography for toxicologists and domain experts.

---

### `03.evaluation/`
Comprehensive computational benchmarking, agreement analysis, and downstream visualization.
- **`00.eval_rag`**: Systematic hyperparameter grid search (top-$k$, rerankers, embeddings, similarity thresholds) optimizing the retrieval component.
- **`01.eval_experts`**: Analyses human expert annotations, evaluates inter-annotator agreement metrics, and builds the consensus ground truth (`Expanded_average_human.csv`).
- **`02.eval_llms`**: Multi-model benchmarking across open-weights LLMs (Qwen, NVIDIA-Nemotron, gpt-oss, GLM) under various prompt formats, consensus runs, and quantization regimes.
- **`03.eval_results`**: Visualizes phenotypic landscapes, ontology tree depth distributions, coverage treemaps, and overall mapping success.

---

## Environment Setup

Each of the four main stages maintains an independent, isolated Python environment managed by [uv](https://docs.astral.sh/uv/) to ensure reproducible dependencies and clean separation of concerns.

To set up and activate the virtual environment for any stage:

```bash
cd toxvaldb/<stage_folder>
chmod ugo+x install_env.sh
./install_env.sh
source env-<name>/bin/activate
```

### Module Environments Overview

| Stage Folder | Environment Script | Virtual Environment Name |
| :--- | :--- | :--- |
| `00.data_preparation/` | `install_env.sh` | `env-data-prep` |
| `01.hipo/` | `install_env.sh` | `env-hipo` |
| `02.expert_annotation/` | `install_env.sh` | `env-expert-annotation` |
| `03.evaluation/` | `install_env.sh` | `env-evaluation` |

All Slurm jobs in each module automatically activate their respective `env-<name>` environment upon execution.\n