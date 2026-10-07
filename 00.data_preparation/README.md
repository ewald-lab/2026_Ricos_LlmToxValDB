# Data Preparation Pipeline (`00.data_preparation`)

This directory houses the data acquisition, curation, normalization, LLM cleaning, and embedding generation pipelines for chemical-toxicological datasets and biomedical ontologies.

The primary reference implementation is **`toxvaldb/`** ([EPA ToxValDB](https://www.epa.gov/chemical-research/toxicity-value-database-toxvaldb)). The system is structured modularly so that any new database or experimental dataset (e.g., ChEMBL, PubChem BioAssay, eChemPortal, Tox21, internal screening assays) can be added as an independent folder under `00.data_preparation/<dataset_name>/` following the contracts and patterns detailed below.

---

## 1. Directory Architecture

```text
00.data_preparation/
├── README.md                          # <-- This adaptation guide for LLMs and developers
├── modules -> toxvaldb/modules         # <-- Backwards compatibility symlink for legacy paths
│
└── toxvaldb/                          # Reference dataset implementation
    ├── install_env.sh                 # Environment installer using uv
    ├── requirements.txt               # Pinned Python dependencies
    └── modules/
        ├── 00.download_toxvaldb/      # Module 00: Acquisition & raw dump unification
        │   ├── jobs/
        │   │   └── 00.download_and_unify.sh
        │   ├── outputs/
        │   └── scripts/
        │       └── 00.unify.py
        │
        ├── 01.preprocess_toxvaldb/    # Module 01: Profiling, LLM cleaning, merging & embeddings
        │   ├── analysis/              # EDA notebooks and summaries
        │   ├── config/
        │   │   ├── clean_prompts/     # Prompts for splitting & exception handling
        │   │   └── rag/               # Embeddings model configs (Harrier, Kalm, Nemotron, Qwen)
        │   ├── jobs/
        │   │   ├── 00.eda_toxvaldb.sh
        │   │   ├── 01.cleaner_llm.sh
        │   │   ├── 01.cleaner_llm_negatives.sh
        │   │   ├── 02.merge.sh
        │   │   └── 03.embeddings.sh
        │   └── scripts/
        │       ├── 01.llm_clean.py
        │       ├── 01.llm_clean_negatives.py
        │       ├── 02.merge_pos_neg.py
        │       ├── 03.embeddings.py
        │       ├── model/             # Embeddings & vLLM client wrappers
        │       └── src/               # Shared utilities
        │
        ├── 02.prepare_ontologies/     # Module 02: Biomedical ontologies & hierarchy embedding
        │   ├── config/
        │   ├── inputs/
        │   │   ├── owl_files/         # OWL / TTL ontology source files
        │   │   ├── ontology_context/  # Context text files for each ontology
        │   │   └── root_uris.json     # Map of ontology key -> Root class URI
        │   ├── jobs/
        │   │   ├── 00.process_ontologies.sh   # Array job: generates LLM defs & embeddings
        │   │   └── 01.all_ontologies.sh       # Processes all ontologies sequentially
        │   ├── outputs/                       # SQLite DBs per model (kalm/, harrier/, etc.)
        │   └── scripts/
        │       ├── 00.aopwiki.py
        │       └── 01.process_ontology.py
        │
        └── 03.prepare_guidelines/     # Module 03: OECD Guidelines extraction & normalization
            ├── config/
            ├── inputs/                # Guideline PDFs and reference lists
            ├── jobs/
            │   ├── 00.extract_endpoints_llm.sh
            │   └── 01.normalize_guidelines.sh
            ├── outputs/               # Extracted endpoints & normalization reports
            └── scripts/
                ├── 00.extract_endpoints_llm.py
                └── 01.normalize_guidelines.py
```

---

## 2. Walkthrough of Reference Implementation (`toxvaldb/`)

The data preparation workflow consists of 4 distinct modules:

### Module 00: `00.download_toxvaldb`
- **Goal:** Ingest raw dumps from external archives (e.g. Zenodo, FTP, MySQL dumps) and produce a single unified tabular file.
- **Job:** `jobs/00.download_and_unify.sh` downloads the raw archive, unzips chunks, and runs `scripts/00.unify.py`.
- **Output:** `outputs/ToxValDB_v9.7.0_Unified.tsv`.

### Module 01: `01.preprocess_toxvaldb`
- **Goal:** Filter out irrelevant assays, clean free-text descriptions using LLMs into atomic phenotypic effects, separate negative/control effects from positive findings, merge them, and pre-compute dense embeddings.
- **Workflow:**
  1. **EDA & Filtering (`00.eda_toxvaldb.sh`):** Filters rows missing effect descriptions or chemical identifiers.
  2. **LLM Cleaning for Positives (`01.cleaner_llm.sh` -> `01.llm_clean.py`):**
     - Queries a served LLM (via OpenAI-compatible vLLM endpoint) using prompt templates in `config/clean_prompts/`.
     - Decomposes multi-effect descriptions (e.g., *"increased liver weight; hepatocellular hypertrophy"*) into discrete atomic phenotypic terms.
     - Produces `ToxValDB_v9.7.0_Cleaned_02_positives.tsv` with columns `TOXICOLOGICAL_EFFECT_LLM_CLEANED` and `TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED`.
  3. **LLM Negatives Classification (`01.cleaner_llm_negatives.sh` -> `01.llm_clean_negatives.py`):**
     - Detects descriptions indicating no adverse effect (e.g., *"no treatment-related changes"*, *"normal histology"*).
     - Produces `ToxValDB_v9.7.0_Cleaned_02_negatives.tsv`.
  4. **Merging (`02.merge.sh` -> `02.merge_pos_neg.py`):**
     - Adds boolean flag `NEGATIVE = False` for positives and `NEGATIVE = True` for negatives.
     - Concatenates into final clean dataset: `outputs/ToxValDB_v9.7.0_Cleaned_03.tsv`.
  5. **Embedding Precomputation (`03.embeddings.sh` -> `03.embeddings.py`):**
     - Extracts all unique strings from `TOXICOLOGICAL_EFFECT` and `TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED`.
     - Generates dense vector embeddings into a persistent SQLite database (`outputs/embeddings_db/tox_effects_<model>.db`).
     - **Checkpointing Guarantee:** Checks if text is already present in `input_embeddings` table before calling the model, preventing redundant recomputations.

### Module 02: `02.prepare_ontologies`
- **Goal:** Build search spaces and term representations for biomedical ontologies:
  - **MP** (Mammalian Phenotype Ontology)
  - **HP** (Human Phenotype Ontology)
  - **TXPO** (Toxicology Pathophysiology Ontology)
  - **AOP** (Adverse Outcome Pathway Wiki)
  - **HPATH** (Histopathology Ontology)
- **Root URIs Configuration (`inputs/root_uris.json`):**
  Defines the root IRI for pruning and hierarchy traversal:
  ```json
  {
    "mp": "http://purl.obolibrary.org/obo/MP_0000001",
    "hp": "http://purl.obolibrary.org/obo/HP_0000001",
    "txpo": "http://purl.obolibrary.org/obo/TXPO_0000009",
    "aop": null,
    "hpath": null
  }
  ```
- **LLM Definitions & Evaluation:** If an ontology term lacks a definition, `01.process_ontology.py` uses an LLM to generate a synthetic definition from labels, synonyms, and ancestor nodes, and evaluates its accuracy against candidate hallucination rubrics.
- **Ontology DB Output:** SQLite database per ontology and embedding model containing terms, ancestor hierarchies, definitions, and embedding BLOBs.

### Module 03: `03.prepare_guidelines`
- **Goal:** Ingest regulatory testing guidelines (OECD Test Guidelines 401–453), extract standardized endpoints using LLMs, and deterministically normalize them.
- **Outputs:** Guideline endpoint catalog and category normalization tables.

---

## 3. Data Contracts & Standard Schemas

Any dataset module prepared under `00.data_preparation/` must produce outputs adhering to the following strict contracts to remain 100% compatible with downstream pipelines (`01.text_to_onto`, `02.expert_annotation`, `03.evaluation`).

### A. Preprocessed TSV Contract (`*_Cleaned_03.tsv`)

The final clean file must be a tab-delimited file (`.tsv`) containing at minimum:

| Column Name | Type | Description |
|---|---|---|
| `dtxsid` (or primary substance ID) | String | Unique chemical or substance identifier (e.g. DSSTox substance ID, ChEMBL ID, PubChem CID). |
| `name` / `chemical_name` | String | Common substance name. |
| `casrn` | String | CAS Registry Number (if available, else empty string). |
| `TOXICOLOGICAL_EFFECT` | String | Original raw effect text from the database. |
| `TOXICOLOGICAL_EFFECT_LLM_CLEANED` | String | Atomic phenotypic effect phrase extracted by LLM (dosage, duration, routes stripped). |
| `TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED` | String | Canonical standardized term phrase. |
| `NEGATIVE` | Boolean | `False` for observed toxic/adverse effects; `True` for control, normal, or no-effect findings. |

### B. SQLite Target Embeddings Contract (`*_effects_<model>.db`)

Precomputed text embeddings must be stored in a SQLite database at:
`outputs/embeddings_db/<source>_effects_<model>.db`

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS input_embeddings (
    text TEXT PRIMARY KEY,
    vector BLOB
);
```

**Storage Details:**
- `text`: Exact string being embedded (case-sensitive or lowercase per tokenizer config).
- `vector`: Raw bytes of the float32 embedding array (e.g., `tensor.cpu().numpy().astype(np.float32).tobytes()`).

### C. Embedding Models Supported

The pipeline natively supports 4 embedding architectures specified in `config/rag/embeddings.json`:
1. **`harrier`**: `ibm-granite/granite-embedding-278m-multilingual`
2. **`kalm`**: `BAAI/bge-m3`
3. **`nemotron`**: `nvidia/NV-Embed-v2`
4. **`qwen`**: `Alibaba-NLP/gte-Qwen2-7B-instruct`

---

## 4. How an LLM Should Adapt This for a New Dataset

When an LLM is requested to support a new database (e.g., `chembl`, `pubchem`, `tox21`, `echemportal`, `my_internal_db`), it should execute the following 6-step workflow:

### Step 1: Initialize New Dataset Directory
Create a new sibling directory under `00.data_preparation/`:
```bash
mkdir -p 00.data_preparation/<dataset_name>/modules
```
Copy `install_env.sh` and `requirements.txt` from `toxvaldb/` or create a symlink to the shared virtual environment:
```bash
cp 00.data_preparation/toxvaldb/install_env.sh 00.data_preparation/<dataset_name>/
cp 00.data_preparation/toxvaldb/requirements.txt 00.data_preparation/<dataset_name>/
```

### Step 2: Implement Module `00.download_<dataset_name>`
1. Create `00.data_preparation/<dataset_name>/modules/00.download_<dataset_name>/` with subfolders `scripts/`, `jobs/`, `outputs/`.
2. Write `scripts/00.unify.py` to:
   - Read the raw dumps / API extracts.
   - Standardize column names (e.g. mapping `assay_description` or `activity_comment` to `raw_effect`).
   - Deduplicate records.
   - Save unified TSV to `outputs/<Dataset>_Unified.tsv`.
3. Create `jobs/00.download_and_unify.sh` following Slurm conventions.

### Step 3: Implement Module `01.preprocess_<dataset_name>`
1. Create `00.data_preparation/<dataset_name>/modules/01.preprocess_<dataset_name>/` with subfolders:
   `analysis/`, `config/clean_prompts/`, `jobs/`, `outputs/`, `scripts/model/`, `scripts/src/`.
2. Reuse or adapt helper modules:
   - `scripts/model/llm.py`: vLLM client supporting async chat completions.
   - `scripts/model/embeddings.py`: SentenceTransformer / HuggingFace embedder.
   - `scripts/src/utils.py`: SQLite caching, logging, prompt loader.
3. Configure prompts in `config/clean_prompts/`:
   - System prompt defining domain boundaries (e.g., distinguishing in vivo toxicological findings from in vitro target binding).
   - Decomposing compound phrases into atomic effect terms.
4. Implement scripts:
   - `01.llm_clean.py`: Processes positive toxicological findings.
   - `01.llm_clean_negatives.py`: Identifies negative / control findings.
   - `02.merge_pos_neg.py`: Merges and sets `NEGATIVE = False / True`.
   - `03.embeddings.py`: Embeds unique strings into SQLite DB `input_embeddings`.
5. Create corresponding Slurm bash jobs in `jobs/` with standard argument parsing (`ENDPOINT`, `MODEL_CONFIG`, `PORT`).

### Step 4: Ontologies (`02.prepare_ontologies`)
- **If using existing biomedical ontologies (MP, HP, TXPO, AOP, HPATH):**
  Simply symlink to the existing processed ontology databases:
  ```bash
  ln -s ../../toxvaldb/modules/02.prepare_ontologies 00.data_preparation/<dataset_name>/modules/02.prepare_ontologies
  ```
- **If adding a new ontology (e.g. GO, ChEBI, Mondo):**
  1. Add source `.ttl` or `.owl` into `02.prepare_ontologies/inputs/owl_files/<new_onto>.ttl`.
  2. Add root URI entry to `02.prepare_ontologies/inputs/root_uris.json`.
  3. Add domain summary text in `02.prepare_ontologies/inputs/ontology_context/<new_onto>.txt`.
  4. Run `00.process_ontologies.sh` to build SQLite embeddings database.

### Step 5: Guidelines (`03.prepare_guidelines`)
- If the new dataset references OECD guidelines or similar regulatory protocols, link or copy `03.prepare_guidelines`.

### Step 6: Verification Checklist
Before declaring the new dataset ready, ensure:
- [ ] `outputs/<Dataset>_Cleaned_03.tsv` exists and has columns `TOXICOLOGICAL_EFFECT`, `TOXICOLOGICAL_EFFECT_LLM_CLEANED`, `TOXICOLOGICAL_EFFECT_LLM_STANDARDIZED`, and `NEGATIVE`.
- [ ] SQLite database `outputs/embeddings_db/<dataset>_effects_<model>.db` exists, contains table `input_embeddings`, and row count matches unique effect strings.
- [ ] Slurm jobs dynamically parse host endpoint and model config without hardcoded hostnames.
- [ ] Virtual environment activation checks both `${PREP_ROOT}/env-data-prep/bin/activate` and `${PREP_ROOT}/../env-data-prep/bin/activate`.

---

## 5. Slurm Job Conventions

All `.sh` batch jobs must follow this robust pattern:

```bash
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -J "prep_job"
#SBATCH -o "logs/%x_%j.out"
#SBATCH -e "logs/%x_%j.err"

# Resolve directory locations reliably
if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)
    SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

# Activate virtualenv with fallback
if [ -f "${PREP_ROOT}/env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/env-data-prep/bin/activate"
elif [ -f "${PREP_ROOT}/../env-data-prep/bin/activate" ]; then
    source "${PREP_ROOT}/../env-data-prep/bin/activate"
fi

# Parse dynamic LLM endpoint arguments
ENDPOINT="${1:-${CODON_NODE:-${BASE_URL:-}}}"
MODEL_CONFIG="${2:-${MODEL_CONFIG:-}}"
PORT="${3:-${CODON_PORT:-8003}}"

if [[ "$ENDPOINT" =~ ^https?:// ]]; then
    FINAL_BASE_URL="$ENDPOINT"
else
    FINAL_BASE_URL="http://${ENDPOINT}.ebi.ac.uk:${PORT}/v1"
fi
```

---

## 6. Prompt Template for Instructing an LLM

Copy and paste the following prompt when prompting an LLM to add a new dataset:

> **Prompt for LLM:**
> "Please adapt `00.data_preparation` for the `[DATASET_NAME]` dataset following the architecture in `00.data_preparation/README.md`.
> 1. Create `00.data_preparation/[dataset_name]/modules/`.
> 2. Create `00.download_[dataset_name]` to ingest the raw data from `[SOURCE_PATH_OR_URL]` and output `[Dataset]_Unified.tsv`.
> 3. Create `01.preprocess_[dataset_name]` to run LLM cleaning, negatives classification, merging into `[Dataset]_Cleaned_03.tsv`, and embedding generation into SQLite `[dataset]_effects_[model].db`.
> 4. Ensure all contracts specified in `00.data_preparation/README.md` (columns, SQLite schema `input_embeddings`, and Slurm path resolution) are strictly satisfied."
