# Hierarchy-aware Phenotype-to-Ontology mapping

This repository provides **HiPO** (Hierarchy-aware Phenotype-to-Ontology mapping), a framework designed to translate free-text phenotypic descriptions into standardized biomedical ontology terms.

By integrating dense retrieval-augmented generation (RAG) with large language model (LLM) hierarchy validation, HiPO identifies appropriate ontology concepts while preserving parent-child relationships and semantic depth.

This repository presents the reference application of HiPO to the curation of the EPA Toxicity Value Database ([ToxValDB](https://www.epa.gov/chemical-research/toxicity-value-database-toxvaldb)), standardizing thousands of unstructured chemical toxicity study outcomes into machine-readable ontological annotations to support downstream computational toxicology, chemical grouping, and hazard assessment.

## Repository Overview

- **`HiPO/`**: Core hierarchy-aware phenotype-to-ontology mapping framework and standalone Python package.
- **`toxvaldb/`**: Reference implementation, data preparation, expert annotation benchmark, and evaluation pipelines applied to ToxValDB. Detailed documentation and submodule setup instructions can be found in the [ToxValDB Pipeline README](toxvaldb/README.md).

## Citation

```bibtex
@article{saurina2026automated,
  title={Automated Ontology Annotation of Phenotypic Descriptions using Large Language Models (LLMs)},
  author={Saurina-i-Ricos, Joan and Puig-Barbe, Aleix and Carli, Francesco and Johannesson, Bjarki and Chouhan, Bhavik and McLaughlin, James and Ewald, Jessica D.},
  year={2026},
  note={In preparation}
}
```
