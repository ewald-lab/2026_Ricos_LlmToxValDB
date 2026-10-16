# Hierarchy-aware Phenotype-to-Ontology mapping

This repository provides **HiPO** (Hierarchy-aware Phenotype-to-Ontology mapping), a general framework designed to translate free-text phenotypic and toxicological descriptions into standardized biomedical ontology terms (e.g., Mammalian Phenotype Ontology, Human Phenotype Ontology, HPATH, AOP-Wiki, TxPO).

By integrating dense retrieval-augmented generation (RAG) with large language model (LLM) hierarchy validation, HiPO identifies appropriate ontology concepts while preserving parent-child relationships and semantic depth.

While HiPO is domain-agnostic and applicable to diverse biomedical and clinical annotation tasks, this repository presents its reference application to the curation of the EPA Toxicity Value Database ([ToxValDB](https://www.epa.gov/chemical-research/toxicity-value-database-toxvaldb)). The end-to-end pipeline standardizes thousands of unstructured chemical toxicity study outcomes into machine-readable ontological annotations, supporting downstream computational toxicology, chemical grouping, and hazard assessment.

## Repository Overview

- **`HiPO/`**: Core hierarchy-aware phenotype-to-ontology mapping framework and standalone Python package.
- **`toxvaldb/`**: Complete reference implementation, data preparation, expert annotation benchmark, and evaluation pipelines applied to ToxValDB. Detailed documentation and submodule setup instructions can be found in the [ToxValDB Pipeline README](toxvaldb/README.md).

## Citation

```bibtex
@article{saurina2026automated,
  title={Automated Ontology Annotation of Phenotypic Descriptions using Large Language Models (LLMs)},
  author={Saurina-i-Ricos, Joan and Puig-Barbe, Aleix and Carli, Francesco and Johannesson, Bjarki and Chouhan, Bhavik and McLaughlin, James and Ewald, Jessica D.},
  year={2026},
  note={In preparation}
}
```\n