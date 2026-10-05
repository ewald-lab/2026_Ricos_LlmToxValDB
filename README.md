# LLM-based Curation of ToxValDB Using Biomedical Ontologies

This project investigates the use of large language models (LLMs) to map free-text toxicological effect descriptions from [ToxValDB](https://www.epa.gov/chemical-research/toxicity-value-database-toxvaldb) to structured biomedical ontologies (e.g. Mammalian Phenotype Ontology, Human Phenotype Ontology, HPATH, AOP-Wiki, TxPO). The goal is to produce high-quality, machine-readable annotations that can support downstream computational toxicology applications such as hazard classification and chemical grouping.

The pipeline combines retrieval-augmented generation (RAG) with LLM-based reranking to identify the most appropriate ontology terms for each effect description. Performance is evaluated both computationally (via retrieval metrics) and through expert annotation.

## Repository Structure

Each analysis module has its own Python environment managed by [uv](https://docs.astral.sh/uv/). To set up a module:

```bash
cd ./XX.module_name
chmod ugo+x install_env.sh
./install_env.sh
source env-<name>/bin/activate
```

### `00.data_preparation/`

Contains data acquisition, curation, and embedding preparation pipelines divided into specialized modules:

- **`00.download_toxvaldb/`**: Downloads the raw ToxValDB database release dumps and unifies partitioned data chunks into a consolidated dataset.
- **`01.preprocess_toxvaldb/`**: Filters and curates toxicological effect descriptions, performing exploratory data analysis, LLM-assisted cleaning/splitting of multi-effect entries, identification of negative/no-effect descriptors, and generation of dense vector embedding databases.
- **`02.prepare_ontologies/`**: Parses biomedical ontologies (MP, HP, HPATH, TXPO, AOP-Wiki) from OWL/TTL sources, reconstructs class hierarchies, generates and evaluates missing definitions using LLMs, and creates embedding databases for vector retrieval.
- **`03.prepare_guidelines/`**: Ingests and processes OECD toxicity testing guidelines, extracting standardized endpoints and generating structured guideline reports using LLMs.

### `01.ontology_pipeline/`

Core mapping pipeline that matches toxicological effect descriptions to ontology terms. Implements a RAG approach using dense retrieval followed by LLM-based reranking. Includes a NextFlow workflow for scalable execution on the EBI cluster.

### `02.rag_evaluation/`

Systematic evaluation of retrieval-augmented generation parameters. Includes hyperparameter tuning for embedding models, chunk sizes, retrieval depth, and reranking strategies. Generates configuration sweeps and aggregates performance metrics.

### `03.expert_annotation/`

Generation of expert annotation test sets and interactive HTML interfaces for domain expert evaluation. Supports sampling strategies (PCA/K-Means) for representative phrase selection and produces structured outputs for computing inter-annotator agreement.

## Citation

```bibtex
@article{saurina2026automated,
  title={Automated Ontology Annotation of Phenotypic Descriptions using Large Language Models (LLMs)},
  author={Saurina-i-Ricos, Joan and Puig-Barbe, Aleix and Carli, Francesco and Johannesson, Bjarki and Chouhan, Bhavik and McLaughlin, James and Ewald, Jessica D.},
  year={2026},
  note={In preparation}
}
```
