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

Preprocessing and cleaning of the ToxValDB dataset and ontology sources. Includes LLM-assisted standardisation of free-text effect descriptions and preparation of ontology databases (parsing OWL files, generating definitions, building embedding databases).

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
