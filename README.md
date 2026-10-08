# LLM-based Curation of ToxValDB Using Biomedical Ontologies

This project investigates the use of large language models (LLMs) to map free-text toxicological effect descriptions from [ToxValDB](https://www.epa.gov/chemical-research/toxicity-value-database-toxvaldb) to structured biomedical ontologies (e.g. Mammalian Phenotype Ontology, Human Phenotype Ontology, HPATH, AOP-Wiki, TxPO). The goal is to produce high-quality, machine-readable annotations that can support downstream computational toxicology applications such as hazard classification and chemical grouping.

The pipeline combines retrieval-augmented generation (RAG) with LLM-based reranking to identify the most appropriate ontology terms for each effect description. Performance is evaluated both computationally (via retrieval metrics) and through expert annotation.

## Repository Structure

- **`HiPO/`**: Core hierarchy-aware phenotype ontology mapping framework.
- **`toxvaldb/`**: Reference implementation and end-to-end pipeline applied to ToxValDB:
  - **`00.data_preparation/`**: Data acquisition, curation, guidelines, and vector embedding databases.
  - **`01.hipo/`**: Core mapping pipeline matching toxicological effect descriptions to ontology terms via RAG + LLM hierarchy validation.
  - **`02.expert_annotation/`**: Expert annotation test set generation and interactive annotation interfaces.
  - **`03.evaluation/`**: Systematic evaluation of RAG parameters, LLM backbones, expert agreement metrics, and figure generation.

Each analysis module in `toxvaldb/` has its own Python environment managed by [uv](https://docs.astral.sh/uv/). To set up a module:

```bash
cd ./toxvaldb/XX.module_name
chmod ugo+x install_env.sh
./install_env.sh
source env-<name>/bin/activate
```

## Citation

```bibtex
@article{saurina2026automated,
  title={Automated Ontology Annotation of Phenotypic Descriptions using Large Language Models (LLMs)},
  author={Saurina-i-Ricos, Joan and Puig-Barbe, Aleix and Carli, Francesco and Johannesson, Bjarki and Chouhan, Bhavik and McLaughlin, James and Ewald, Jessica D.},
  year={2026},
  note={In preparation}
}
```
