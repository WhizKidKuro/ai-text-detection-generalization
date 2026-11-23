
# Documentation

This directory contains the written documentation associated with 
the study *Do AI-Text Detectors Transfer?*, including the project 
report, methodological details, experimental design, findings, 
and interpretation of results.

The documentation is intended to provide a clearer understanding 
of the motivation behind the study, the questions it investigates, 
the approaches used in the experiments, and the conclusions drawn 
from the analysis.

## Contents

- **`project_report.md`**: A detailed report describing the research 
  problem, research questions, experimental methodology, detector 
  configurations, datasets, results, analysis, limitations, and 
  conclusions.

## About the Study

The project investigates the generalization of classical AI-text 
detectors across different text domains and examines whether 
English Language Learner (ELL) essays are disproportionately 
classified as AI-generated.

The study is organized around two research questions:

1. **Cross-domain generalization:** How does the performance of 
   AI-text detectors change when they are evaluated on text from 
   domains different from those used during training?

2. **Group fairness:** Are human-written essays produced by 
   English Language Learners flagged as AI-generated more often 
   than essays written by other students?

The experiments use two classical machine-learning approaches 
to AI-text detection and a length-only baseline to investigate 
performance, generalization, and potential confounding factors.

## Documentation Scope

The report documents:

- The motivation and background of the study.
- Research questions and hypotheses.
- Dataset sources and preprocessing.
- Detector architectures and feature representations.
- Experimental design and evaluation metrics.
- Cross-domain performance analysis.
- ELL-related false-positive analysis.
- Statistical analysis and interpretation.
- Limitations and reproducibility considerations.

## Reproducibility

The report is intended to be read alongside the implementation 
and experiment scripts in the repository. Experimental procedures, 
data preparation, and analysis are documented to make the study 
easier to inspect and reproduce in compatible environments.

Dataset availability, computational requirements, and any 
environment-specific restrictions are described where relevant.

---

This documentation accompanies the implementation and experimental 
materials available in this repository.

