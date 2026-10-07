# Data Preparation Pipeline (`00.data_preparation`)

This directory houses the data acquisition, curation, normalization, LLM cleaning, and embedding generation pipelines for chemical-toxicological datasets and biomedical ontologies.

The primary reference implementation is **`toxvaldb/`** ([EPA ToxValDB](https://www.epa.gov/chemical-research/toxicity-value-database-toxvaldb)). The system is structured modularly so that any new database or experimental dataset can be added as an independent folder under `00.data_preparation/<dataset_name>/` following the contracts and patterns detailed below.

---

## Reference Implementation (`toxvaldb/`)

The reference preparation workflow is organized into 4 focused modules:

### 1. `00.download_toxvaldb`
- **Goal:** Ingest raw archive dumps (Zenodo/FTP/MySQL) and produce a single unified dataset.
- **Workflow:** `00.download_and_unify.sh` downloads and extracts raw files, then runs `00.unify.py` to output `outputs/ToxValDB_v9.7.0_Unified.tsv`.

### 2. `01.preprocess_toxvaldb`
- **Goal:** Filter, clean with LLM into atomic phenotypic effects, separate negative/control findings, and generate dense embeddings.
- **Workflow:**
  1. **EDA (`00.eda_toxvaldb.sh`):** Filters out records missing effect descriptions or identifiers.
  2. **LLM Cleaning (`01.cleaner_llm.sh` -> `01.llm_clean.py`):** Uses an LLM to split multi-effect descriptions into atomic phenotypic phrases (`TOXICOLOGICAL_EFFECT_LLM_CLEANED`).
  3. **LLM Negatives (`01.cleaner_llm_negatives.sh` -> `01.llm_clean_negatives.py`):** Detects control or no-effect findings.
  4. **Merge (`02.merge.sh` -> `02.merge_pos_neg.py`):** Assigns `NEGATIVE=False` to positives and `NEGATIVE=True` to negatives, producing `outputs/ToxValDB_v9.7.0_Cleaned_03.tsv`.
  5. **Embeddings (`03.embeddings.sh` -> `03.embeddings.py`):** Computes dense vector embeddings for unique strings into SQLite `outputs/embeddings_db/tox_effects_<model>.db` with incremental caching.

### 3. `02.prepare_ontologies`
- **Goal:** Build search spaces and term representations for biomedical ontologies (MP, HP, TXPO, AOP, HPATH).
- **Workflow:** Parses OWL/TTL files using root IRIs (`inputs/root_uris.json`), generates and evaluates LLM definitions for terms without them, and builds SQLite ontology embedding databases for each supported embedding model (`00.process_ontologies.sh` / `01.all_ontologies.sh`).

### 4. `03.prepare_guidelines`
- **Goal:** Ingest regulatory testing guidelines (OECD PDFs), extract standardized endpoints with LLMs, and deterministically normalize them (`00.extract_endpoints_llm.sh`, `01.normalize_guidelines.sh`).

---

## Data Contracts & Standard Schemas

Any dataset module prepared under `00.data_preparation/` must produce outputs adhering to the following strict contracts to remain compatible with downstream pipelines (`01.text_to_onto`, `02.expert_annotation`, `03.evaluation`).

### A. Preprocessed TSV Contract (`*_Cleaned_03.tsv`)

The final clean file must be a tab-delimited file (`.tsv`) containing at minimum:

| Column Name | Type | Description |
|---|---|---|
| `dtxsid` (or primary ID) | String | Unique chemical or substance identifier (e.g. DSSTox substance ID, ChEMBL ID, PubChem CID). |
| `name` / `chemical_name` | String | Common substance name. |
| `casrn` | String | CAS Registry Number (if available, else empty string). |
| `TOXICOLOGICAL_EFFECT` | String | Original raw effect text from the database. |
| `TOXICOLOGICAL_EFFECT_LLM_CLEANED` | String | Atomic phenotypic effect phrase extracted by LLM (dosage, duration, routes stripped). |
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
- `text`: Exact string being embedded.
- `vector`: Raw bytes of the float32 embedding array (`vector.tobytes()`).

### C. Supported Embedding Models

Configured in `config/rag/embeddings.json`:
1. **`harrier`**: `ibm-granite/granite-embedding-278m-multilingual`
2. **`kalm`**: `BAAI/bge-m3`
3. **`nemotron`**: `nvidia/NV-Embed-v2`
4. **`qwen`**: `Alibaba-NLP/gte-Qwen2-7B-instruct`

---

## How an LLM Should Adapt This for a New Dataset

When an LLM is requested to support a new database, execute the following 6-step workflow:

### Step 1: Initialize Directory & Environment
Create `00.data_preparation/<dataset_name>/modules` and copy `install_env.sh` and `requirements.txt` from `toxvaldb/` (or symlink the shared virtual environment):
```bash
mkdir -p 00.data_preparation/<dataset_name>/modules
cp 00.data_preparation/toxvaldb/install_env.sh 00.data_preparation/<dataset_name>/
cp 00.data_preparation/toxvaldb/requirements.txt 00.data_preparation/<dataset_name>/
```

### Step 2: Implement Module `00.download_<dataset_name>`
- Create `modules/00.download_<dataset_name>/` with `scripts/`, `jobs/`, `outputs/`.
- Write `scripts/00.unify.py` to ingest the raw source, harmonize columns, deduplicate records, and output `outputs/<Dataset>_Unified.tsv`.
- Create `jobs/00.download_and_unify.sh`.

### Step 3: Implement Module `01.preprocess_<dataset_name>`
- Create `modules/01.preprocess_<dataset_name>/` copying `scripts/model/` and `scripts/src/` from `toxvaldb/`.
- Adapt prompts in `config/clean_prompts/` to define the domain boundaries and splitting instructions.
- Implement `01.llm_clean.py` (positives), `01.llm_clean_negatives.py` (negatives), `02.merge_pos_neg.py` (merging), and `03.embeddings.py` (caching embeddings into SQLite).
- Add Slurm batch scripts in `jobs/` parsing `ENDPOINT`, `MODEL_CONFIG`, and `PORT`.

### Step 4: Ontologies (`02.prepare_ontologies`)
- **If reusing existing ontologies (MP, HP, TXPO, AOP, HPATH):** Symlink to `toxvaldb/modules/02.prepare_ontologies`:
  ```bash
  ln -s ../../toxvaldb/modules/02.prepare_ontologies 00.data_preparation/<dataset_name>/modules/02.prepare_ontologies
  ```
- **If adding a new ontology:** Add the source `.ttl` to `inputs/owl_files/`, register its root IRI in `inputs/root_uris.json`, and run `00.process_ontologies.sh`.

### Step 5: Guidelines (`03.prepare_guidelines`)
- If the dataset uses regulatory testing protocols (e.g. OECD guidelines), link or adapt `03.prepare_guidelines`.

### Step 6: Verification Checklist
Ensure:
- [ ] `outputs/<Dataset>_Cleaned_03.tsv` has `TOXICOLOGICAL_EFFECT`, `TOXICOLOGICAL_EFFECT_LLM_CLEANED`, and `NEGATIVE`.
- [ ] SQLite database `outputs/embeddings_db/<dataset>_effects_<model>.db` exists with table `input_embeddings`.
- [ ] Slurm jobs dynamically parse host endpoint and model config without hardcoded hostnames.
- [ ] Virtual environment activation checks both `${PREP_ROOT}/env-data-prep/bin/activate` and `${PREP_ROOT}/../env-data-prep/bin/activate`.

---

## Slurm Job Conventions

All `.sh` batch jobs must follow this pattern:

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

## Prompt Template for Instructing an LLM

Copy and paste the following prompt when instructing an LLM to add a new dataset:

> **Prompt for LLM:**
> "Please adapt `00.data_preparation` for the `[DATASET_NAME]` dataset following the architecture in `00.data_preparation/README.md`.
> 1. Create `00.data_preparation/[dataset_name]/modules/`.
> 2. Create `00.download_[dataset_name]` to ingest raw data from `[SOURCE_PATH_OR_URL]` and output `[Dataset]_Unified.tsv`.
> 3. Create `01.preprocess_[dataset_name]` to run LLM cleaning, negatives classification, merging into `[Dataset]_Cleaned_03.tsv`, and embedding generation into SQLite `[dataset]_effects_[model].db`.
> 4. Ensure all contracts specified in `00.data_preparation/README.md` (columns, SQLite schema `input_embeddings`, and Slurm path resolution) are strictly satisfied."
