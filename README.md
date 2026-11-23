
# Do AI-Text Detectors Transfer?

**Cross-domain generalization and group fairness of two classical AI-text detectors**

AI-text detectors label a passage as human-written or AI-written. Schools use them, sometimes for misconduct decisions, but they are usually tested on text that looks like their training data. This project asks two questions with simple, explainable models:

- **RQ1 (generalization):** If a detector is trained on one domain, how much does it lose on others?
- **RQ2 (fairness):** Are human essays by English language learners (ELL) flagged as AI more often than essays by other students?

Everything runs on CPU in minutes. No GPU and no deep learning are needed.

## Key Findings

| Question | Result |
|---|---|
| RQ1: Transfer across domains | Mean AUROC falls from 0.989 to 0.886 (Detector A) and from 0.953 to 0.798 (Detector B) on unseen domains. The hypothesis is supported. |
| RQ2: ELL vs non-ELL essays | No detectable gap (A: 10.3% vs 8.2%; B: 41.5% vs 41.7%; every 95% interval includes zero), though a gap of up to about 6 points cannot be ruled out. |
| Overall reliability | Both detectors wrongly flag many human essays: about 8–10% (A) and about 42% (B). They should never be the only evidence against a student. |
| Methodological lesson | A first comparison looked like strong bias but came from essay length and formatting artifacts. A length-only control and a data audit exposed it. |

## The Detectors

| Detector | Reads | Model |
|---|---|---|
| **A** | Which words and word pairs appear | TF-IDF (up to 50,000 terms) + logistic regression |
| **B** | Six writing-style numbers (sentence length, vocabulary variety, punctuation and others) | Random forest, 300 trees |
| **L** (control) | Text length only | Logistic regression on log word count |

## Repository Structure

```text
ai-text-detection-generalization/
│
├── documentation/
│   └── project_report.md
│
├── implementation/
│   └── Python implementation files
│
├── demo/
│   ├── detector.html
│   ├── model.js
│   ├── export_model.py
│   └── README.md
│
├── data/
│   └── README.md
│
├── app.py
├── extras.py
├── run_experiments.py
├── train.csv
├── requirements.txt
├── .gitignore
└── README.md
```

The repository contains the Python implementation, project documentation, experimental scripts, dataset documentation, and an offline browser demo.

## Quick Start (Python)

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the experiments:

```bash
python run_experiments.py --source hc3 --essays essays.csv --seeds 3 --cap 800
```

Run the additional analysis:

```bash
python extras.py --essays essays.csv
```

The project also includes a Streamlit application (`app.py`). However, on my Windows laptop, local execution of the Streamlit-based application was restricted by system security policies affecting Python scientific packages and executable files.

The application and experimental scripts are retained in the repository for use in compatible environments.

- `essays.csv` needs two columns, `text` and `group` (`native` or `non-native`), containing human-written essays only. It is not included in this repository.
- `--source demo` runs the pipeline on a small synthetic dataset. It only checks that the code works; its results mean nothing.
- Colab users can run the commands in notebook cells using `!`.

## Offline Browser Demo

An offline HTML and JavaScript demo is included as an alternative to the locally restricted Streamlit environment.

The demo (`demo/detector.html`) scores pasted text with all three detectors, highlights the words that push Detector A's score, and shows Detector B's six style numbers against the average human and AI training texts.

The demo runs directly in a browser without requiring Python or an internet connection, once the model file is available.

Open `demo/detector.html` in Chrome or Edge to use the demo.

**How it works.** `model.js` stores the trained parameters: Detector A's vocabulary, inverse document frequencies and weights; all 300 trees of Detector B; and Detector L's coefficients. `detector.html` re-implements the cleaning, features and scoring in JavaScript. On test texts it matched scikit-learn to within about 1e-7 for Detectors A and L, and exactly for Detector B, using a small test model.

**Please note:**

- The model file included in the demo may be a synthetic demo model. A yellow banner is shown for the demo model, and its scores mean nothing. Use a model exported from the actual training pipeline for meaningful scores.
- The demo models are trained on all HC3 data, so a pasted paragraph will not reproduce the exact numbers of the paper's tables, which use 80% training splits.

## Running on Locked-Down Windows Laptops

On some college or managed Windows laptops, installing or running Python scientific packages fails. Typical messages include:

```text
pip.exe failed to run: An Application Control policy has blocked this file
```

```text
ImportError: DLL load failed while importing _multiarray_umath: An Application Control policy has blocked this file
```

These restrictions are imposed by the machine's security policy and can only be changed by its administrator. Do not attempt to bypass them.

Alternatives include:

1. **Offline browser demo:** Use `demo/detector.html`. It requires only a web browser.
2. **Streamlit Community Cloud:** Deploy the repository through [Streamlit Community Cloud](https://share.streamlit.io), selecting `app.py` as the main file, subject to compatible dependencies and configuration.
3. **Google Colab:** Run the Python experiments in a notebook environment.
4. **Another computer:** Run the project on a machine without the same restrictions.

Streamlit through a Colab tunnel often fails. Serving Streamlit from Colab with a temporary tunnel such as localtunnel can show red `TypeError: Failed to fetch dynamically imported module` boxes in place of the widgets, because the tunnel cannot deliver Streamlit's interface files. Deploy on Streamlit Community Cloud instead, or use the offline browser demo.

## Data

| Dataset | Used for | How it is obtained |
|---|---|---|
| **HC3** (Human ChatGPT Comparison Corpus), five domains: reddit_eli5, finance, medicine, open_qa, wiki_csai | Training and domain-transfer evaluation | Downloaded automatically from the Hugging Face Hub (`Hello-SimpleAI/HC3`) |
| **Feedback Prize – English Language Learning** | Human essay data used in the fairness analysis | The `train.csv` file is retained under its original filename for compatibility with the existing implementation. |
| **PERSUADE 2.0** student essays with ELL status | Fairness test (500 ELL and 500 non-ELL essays, 150–600 words) | Obtain the dataset separately and check its license before use. The larger dataset file used locally is not included in this repository because of its size. |

All text is cleaned with one function (`lab/data.py`) that removes formatting artifacts found by a data audit: literal backslash-n sequences (in 8.4% of ChatGPT answers and no human answers), `url_N` placeholders, and spaces before punctuation.

## Limitations and Ethics

- One AI source (ChatGPT) and simple models; results may not hold for newer generators or fine-tuned transformers.
- Three seeds vary the splits but not the sampled data.
- The ELL label is administrative and not identical to being a non-native speaker; the groups may also differ in prompts or grade level, so RQ2 shows an association, not a cause.
- Results use a 0.5 cutoff; other cutoffs may behave differently.
- A detector score is a statistical guess, not proof. Detectors like these should never be the only evidence in a misconduct decision.

## Reproducing the Results

Every number in the report and paper comes from `run_experiments.py` and `extras.py`. Random seeds are fixed, and the commands are listed above.
