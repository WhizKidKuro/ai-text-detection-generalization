# Datasets

This project uses the following datasets for model development, cross-domain evaluation, and fairness analysis.

## HC3 (Human ChatGPT Comparison Corpus)

HC3 contains human-written responses collected from sources such as Reddit alongside ChatGPT-generated responses to corresponding questions. It provides paired human and AI-generated text used in AI-text detection research. In this project, HC3 is used through the existing data-loading pipeline for the detector experiments.

**Source:** [HC3 on Hugging Face](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

## PERSUADE 2.0

PERSUADE 2.0 is a corpus of student-written argumentative essays, including annotated writing data and associated writer information. It is used in this project to examine potential differences in false-positive rates between essays written by English Language Learners (ELL) and non-ELL writers. Due to the size of the dataset, the original files are not hosted in this repository.

**Dataset:** [PERSUADE 2.0 on Kaggle](https://www.kaggle.com/datasets/mpware/persuade-2-0)

**Original repository:** [PERSUADE Corpus 2.0](https://github.com/scrosseye/persuade_corpus_2.0)
