# Do AI-Text Detectors Transfer?

## Cross-domain Generalization and Group Fairness of Two Classical AI-Text Detectors

## Abstract

AI-text detectors are classifiers that label a passage as human-written or machine-written. Schools and publishers use them, sometimes to inform misconduct decisions, yet they are usually evaluated on text that resembles their training data. This project investigates two questions with simple, interpretable models: (RQ1) how much does detector performance fall on domains not seen in training, and (RQ2) are human essays written by English language learners (ELL) flagged as AI more often than essays by other students?

Two classical detectors were built: Detector A, a TF-IDF logistic-regression model that reads word patterns, and Detector B, a random forest over six stylometric features, together with a length-only control. They were trained on human and ChatGPT answers from five domains of the HC3 corpus and tested across domains and on 1,000 human essays from the PERSUADE 2.0 corpus. On unseen domains, mean AUROC fell from 0.989 to 0.886 for Detector A and from 0.953 to 0.798 for Detector B. After controlling for text length, no detectable difference in false-flag rates was found between ELL and non-ELL essays (A: 10.3% against 8.2%; B: 41.5% against 41.7%), but both detectors wrongly flagged many human essays (about 8 to 10% and 42%).

A first comparison that appeared to show strong bias against ELL writers was traced to essay length and to formatting artifacts in the data; a length-only control and a data audit exposed and removed both. The work is packaged as reproducible code, an offline interactive demonstration that scores a pasted paragraph, and this report. The findings caution against using such detectors as the sole evidence of misconduct.

**Keywords:** AI-generated text detection, domain shift, algorithmic fairness, English language learners, stylometry, TF-IDF, random forest.

---

# 1. Introduction and Problem Statement

## 1.1 Overview of the Topic

Large language models (LLMs) such as ChatGPT can write fluent essays, answers and summaries in seconds [2]. Since the end of 2022 these tools have become part of everyday study and work, and educators, publishers and employers now face a practical question: was this text written by a person or by a machine? Answering that question reliably matters because the answer is increasingly used to judge people.

AI-text detection is the task of labelling a passage as human-written or machine-written. It is a binary classification problem. A detector reads a text, produces a score between 0 and 1 that expresses how likely the text is to be AI-written, and compares that score with a cutoff, commonly 0.5. Detectors range from simple statistical tools to deep neural networks, and several commercial products are marketed directly to schools and universities.

Vendors usually report accuracy figures measured on benchmark data that looks very much like the data the detector was built on. Independent testing of detection tools for academic use has found them neither accurate nor reliable enough to carry high-stakes decisions on their own [24]. This gap between laboratory accuracy and real-world behaviour is the starting point of this project.

## 1.2 Problem Statement

This project addresses two linked reliability problems of AI-text detectors.

- **Generalization.** A detector trained on one kind of text, for example medical answers, may fail on another kind, for example casual forum posts. Such a change in the type of text is called a domain shift [18]. If detectors are only tested on text similar to their training data, their real-world accuracy is unknown.
- **Fairness.** If simple or less varied wording looks machine-like to a detector, writers who are still learning English may be wrongly accused of using AI. Earlier work reported exactly this pattern for several well-known detectors [13].

**Problem statement.** Do AI-text detectors keep working outside the data they were trained on, and do they wrongly flag essays by English language learners (ELL) as AI-written more often than essays by other students? The project answers this question with two simple, explainable detectors, a controlled experimental design, and honest statistical reporting.

## 1.3 Real-World Need

A false positive, a human text labelled as AI, is not a harmless error. A student whose own essay is flagged may face a failed assignment, an academic-integrity hearing, a note on their record, or loss of trust from teachers. Because most students never see how a detector reached its verdict, they have little means of defending themselves. A false negative, an AI text labelled as human, is a smaller harm by comparison. This asymmetry makes the false positive rate the most important error to measure.

Fairness adds a second need. Students who are learning English often write with simpler sentences and a narrower vocabulary. A detector that treats such writing as machine-like would place an extra burden on exactly those students. Institutions therefore need evidence about three things before relying on a detector: how it behaves on unfamiliar text, whether its errors fall equally on different groups of writers, and how large the error rates are at the cutoff they plan to use.

The main stakeholders are:

- Students, who bear the cost of a wrong accusation;
- Teachers and examiners, who need to know how much weight a detector score deserves;
- Institutions, who write misconduct policies and purchase tools;
- Tool developers and researchers, who need honest evaluation protocols.

## 1.4 Scope and Approach

The study is deliberately small and controlled. It trains two classical machine-learning detectors on public human-versus-ChatGPT text and stress-tests them across domains and on essays by two groups of students. Everything runs on a laptop or a free cloud notebook in minutes, with no GPU and no deep learning. The contribution is not a better detector; it is a careful comparison that shows when detectors can and cannot be trusted.

**In scope:** binary classification (human = 0, AI = 1); two detectors, a TF-IDF logistic-regression model and a stylometric random-forest model, plus a length-only control; a cross-domain test (RQ1); an ELL versus non-ELL false-positive test (RQ2); one ablation study; an error analysis; an interactive demonstration.

**Out of scope:** fine-tuning transformer models, generating new AI text, testing many different AI models, and building a state-of-the-art detector.

## 1.5 Significance of the Work

The study matters for three reasons.

First, it addresses a decision that affects real people. Institutions are making choices about detector use now, often on the basis of vendor claims, and evidence about behaviour outside the vendor's test data is scarce. A small, transparent study can show which questions institutions should ask before relying on a tool.

Second, the project demonstrates a style of evaluation that is easy to adopt: test on unseen domains, include a trivial control, report uncertainty, and audit the data. Each of these steps changed a conclusion in this study. Without the length-only control, the first comparison would have reported strong bias against ELL students; without the data audit, formatting artifacts would have inflated the apparent skill of the detectors.

Third, the project is educational. It applies the main stages of a machine-learning workflow, from data collection and cleaning through modelling, evaluation, statistics and deployment, to a problem where the quality of the evidence matters more than the size of the model.

## 1.6 Organisation of the Report

Section 2 states the learning objectives and expected outcomes. Section 3 reviews related work and the background concepts needed to follow the method. Section 4 describes the proposed methodology and the architecture of the detectors. Section 5 describes the datasets and the pre-processing and audit steps. Section 6 documents the implementation and experimental setup. Section 7 presents and analyses the results. Section 8 concludes and outlines future scope. Section 9 lists the references.

---

# 2. Learning Objectives & Expected Outcomes

The project has four objectives. Each objective is linked to a measurable outcome so that success can be judged objectively at the end of the study (Table 1).

## 2.1 Learning Objectives

**Objective 1: Build and evaluate supervised AI-text detectors.** Design, train and evaluate two interpretable detectors, one that reads word patterns and one that reads writing style, together with a length-only control, using standard classification metrics.

**Objective 2: Measure cross-domain generalization (RQ1).** Quantify how much detector performance drops when a detector is tested on a domain it was not trained on, and test the hypothesis (H1) that performance on unseen domains is clearly lower than on the training domain.

**Objective 3: Evaluate fairness for English language learners (RQ2).** Compare the rate at which human essays by ELL and non-ELL students are wrongly flagged as AI, with confidence intervals and a length-matched comparison, and test the hypothesis (H2) that the false positive rate is higher for ELL essays.

**Objective 4: Audit the data and deliver a reproducible package.** Check the data for shortcuts and confounds that could distort conclusions, and deliver reproducible code, an interactive demonstration that scores a pasted paragraph, and a written report.

## 2.2 Expected Outcomes

**Table 1: Objectives, expected outcomes and success criteria**

| Objective | Expected outcome | Success criterion |
|---|---|---|
| 1. Build detectors | Two working detectors (A and B) and a length-only control, trained on HC3 | All three models trained and evaluated with AUROC, accuracy, F1 and false positive rate |
| 2. Generalization (RQ1) | A 5 × 5 transfer matrix for each detector; average drop between in-domain and unseen domains | Drop reported with standard deviation over three random seeds; H1 accepted or rejected |
| 3. Fairness (RQ2) | False positive rates for ELL and non-ELL essays with 95% confidence intervals, plus a length-matched comparison | Gap and its interval reported; H2 accepted or rejected using a pre-stated rule |
| 4. Audit and package | Data audit findings, a code repository, an interactive demonstration and this report | Every number in the report regenerated by the repository code |

A hypothesis can be wrong. If the transfer matrix shows little drop, or the false positive rates turn out to be similar, that is still a valid result and is reported as a finding. The success criteria therefore concern the quality of the evidence, not the direction of the result.

---

# 3. Literature Review / Background Study

## 3.1 Large Language Models and Machine-Generated Text

Brown et al. showed that very large language models can perform new tasks from a few examples in the prompt [2], and chat-oriented descendants made such models available to the general public. Their output is fluent and often factually plausible, which is why distinguishing it from human writing by reading alone is difficult. The same fluency motivates technical detectors: tools that analyse a text and estimate its origin.

## 3.2 Approaches to Detecting Machine-Generated Text

Existing detectors fall into four broad families (Table 2).

**Statistical and zero-shot methods** use the probabilities a language model assigns to the words of a text. GLTR visualises how likely each word is under a language model so that humans can spot suspiciously predictable text [7]. DetectGPT scores a passage by how its log-probability changes under small rewrites, without any labelled training data [16]. These methods need access to a scoring model, and their behaviour depends on how closely it matches the model that wrote the text.

**Trained classifiers** learn from labelled examples of human and machine text. An early example is the RoBERTa-based output detector released alongside GPT-2 [21]. Commercial detectors are generally of this kind. They can be very accurate on text like their training data, but they are opaque and their behaviour on different text is rarely documented.

**Watermarking** changes the generator so that its output carries a hidden statistical signal that a detector can later check [12]. It is attractive but depends on the cooperation of the model provider and cannot identify text from models that do not embed a watermark.

**Stylometry and classical machine learning** measure writing style, such as sentence length and vocabulary variety, and feed it to simple classifiers. Stylometry has a long history in authorship attribution [22]. These methods are cheap and explainable, which is why this project uses them as transparent baselines.

**Table 2: Comparison of AI-text detection approaches**

| Approach | Main idea | Strengths | Weaknesses |
|---|---|---|---|
| Statistical / zero-shot (GLTR, DetectGPT) | Use language-model probabilities of the words | No labelled data needed; some theory | Needs a scoring model; sensitive to model mismatch and to low-perplexity human text |
| Trained classifiers (RoBERTa detector, commercial tools) | Learn from labelled human and AI text | High accuracy on similar data | Opaque; can fail on new domains and generators |
| Watermarking | Generator embeds a hidden signal | Strong detection when present | Needs generator cooperation; no help for other models |
| Stylometry and classical ML (this project) | Style or word statistics with simple classifiers | Cheap, fast, explainable | Limited capacity; style cues can be confounded |

## 3.3 The Human ChatGPT Comparison Corpus (HC3)

Guo et al. built HC3, a corpus in which each question has both human answers and ChatGPT answers, collected across several domains including open-domain questions, finance, medicine, computer science and an explain-like-I'm-five forum [8]. They also analysed linguistic differences between the two kinds of answer and trained detectors on the data. Because both answer types respond to the same question, the corpus is well suited to controlled comparisons, and it is the main training and test resource of this project.

## 3.4 Reliability of Detectors

Sadasivan et al. argued that as language models improve, their output distribution approaches that of human text, which limits the reliability of any detector [19]. Weber-Wulff et al. tested a set of available detection tools on texts of known origin and concluded that they were neither accurate nor reliable enough to be used as the only basis of academic-misconduct decisions [24]. These results motivate asking not only how accurate a detector is, but where and for whom it fails.

## 3.5 Fairness and Bias

Liang et al. evaluated several widely used detectors on essays written by non-native English speakers and by native speakers, and found that the non-native essays were misclassified as machine-written much more often; they linked this to the low perplexity of simple wording [13]. Their work is the main motivation for RQ2.

Fairness can be defined in several ways. One family of definitions, equality of opportunity and equalised odds, requires that error rates are similar across groups [9]. In this project the relevant error is the false positive rate on human essays, so fairness is operationalised as similar false positive rates for ELL and non-ELL students. This is only one of several possible notions, and the report states clearly when conclusions depend on it.

## 3.6 Dataset Shift and Shortcut Learning

Dataset shift occurs when the data a model meets in use differs from the data it was trained on [18]. A related problem is shortcut learning, where a model succeeds by exploiting an irrelevant feature that happens to correlate with the label in the training data, and fails when the correlation disappears [6]. In AI-text detection, candidate shortcuts include text length, formatting characters and topic. Section 5 and Section 7 show that both length and formatting artifacts played a role in this study.

## 3.7 Foundational Concepts

### 3.7.1 TF-IDF

Term Frequency-Inverse Document Frequency converts text into numbers by weighting each word by how often it occurs in the document and how rare it is across all documents [20]. Words that appear in nearly every text receive a low weight, and words that are informative for a subset of texts receive a high weight. TF-IDF vectors are sparse and high-dimensional, which suits linear classifiers.

### 3.7.2 Logistic Regression

Logistic regression is a linear classifier that multiplies each input feature by a learned weight, adds the products and a bias, and passes the sum through the logistic function to obtain a probability between 0 and 1. Its weights can be inspected directly, so the words that push a score towards AI or towards human can be listed.

### 3.7.3 Random Forests

A random forest is an ensemble of many decision trees, each trained on a random resample of the data, whose outputs are averaged [1]. Averaging reduces the variance of individual trees and makes the model robust. Forests can capture non-linear relationships between features, but they are harder to interpret than linear models.

### 3.7.4 ROC Curves and AUROC

The receiver operating characteristic (ROC) curve plots the true positive rate against the false positive rate as the cutoff varies, and the area under it (AUROC) summarises ranking quality independently of any single cutoff [5]. An AUROC of 1.0 means every AI text scores higher than every human text; 0.5 corresponds to random guessing.

### 3.7.5 The Bootstrap

The bootstrap estimates the uncertainty of a statistic by repeatedly resampling the observed data with replacement and recomputing the statistic [4]. The middle 95% of the resampled values gives a 95% confidence interval without assuming a particular distribution.

## 3.8 Evaluation Practice and Its Pitfalls

Evaluation practice in detector research has several common weaknesses. Many studies split a single dataset at random into training and test parts, so that both parts share topics, writing conditions and formatting; a detector can then succeed by exploiting those shared properties. Accuracy at one cutoff is often the only reported figure, which hides how scores are distributed and how error rates change when the cutoff moves. Few studies test on text from a different domain or a different generator, and fewer still compare groups of writers. Finally, results are often reported as single numbers without any indication of uncertainty, so small differences are interpreted as real.

Good practice therefore includes splitting so that related items stay together, reporting a cutoff-free measure such as AUROC alongside error rates at the operating cutoff, including trivial baselines, repeating experiments over several random seeds, and attaching confidence intervals to group comparisons. This project applies all of these practices.

## 3.9 Research Gap

Most published detector evaluations either report in-distribution accuracy or examine a single question, such as bias against non-native writers, for a fixed set of commercial tools. Few studies combine a cross-domain evaluation, a length-only control, a controlled comparison of writer groups, and an audit of the data for shortcuts, using transparent models whose behaviour can be inspected. This project fills that gap on a small, reproducible scale.

In summary, the literature shows that detectors are widely used, that accuracy claims are rarely tested outside the training distribution, and that fairness towards non-native writers is a documented concern. What is missing is a compact, controlled study that measures both problems together with transparent models and honest statistics. The next sections describe how this project provides one.
---

# 4. Proposed Methodology / AI Model Architecture

## 4.1 Overview of the Approach

The methodology is one straight pipeline that is run once for each of three random seeds (Figure 1). Public data are loaded, audited and cleaned; each domain is split by question into training and test parts; three detectors are trained; four experiments are run; and the results are turned into tables, figures and an interactive demonstration.

**Figure 1: End-to-end methodology pipeline**

## 4.2 Detector A: TF-IDF with Logistic Regression

Detector A reads which words and word pairs appear in a text (Figure 2). After cleaning, the text is lower-cased and split into word tokens. Both single words and pairs of consecutive words (bigrams) are counted, terms that appear in fewer than two training texts are dropped, and at most 50,000 terms are kept.

For a term \(t\) in a document \(d\) with count \(c(t,d)\), the term weight is:

\[
tf(t,d) = 1 + \ln c(t,d)
\tag{1}
\]

\[
idf(t) = \ln\left[\frac{1+N}{1+df(t)}\right] + 1
\tag{2}
\]

where \(N\) is the number of training documents and \(df(t)\) the number of documents containing \(t\). The weights \(tf \cdot idf\) form a vector \(v\), which is scaled to unit length, \(x = v / \|v\|_2\). Logistic regression then converts the vector into a probability that the text is AI-written:

\[
P(AI \mid x) = \sigma(w \cdot x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
\tag{3}
\]

The sub-linear term frequency in Equation 1 prevents a word repeated many times from dominating, and the length normalisation makes long and short texts comparable. Because the weights \(w\) are visible, the words that push the score towards AI or towards human can be listed.

**Figure 2: Architecture of Detector A (TF-IDF with logistic regression)**

## 4.3 Detector B: Stylometric Features with a Random Forest

Detector B ignores which words are used and reads only six numbers that describe writing style (Table 3 and Figure 3). The features are computed with plain Python and regular expressions on the cleaned text.

**Table 3: Stylometric features used by Detector B**

| No. | Feature | Definition | Intuition |
|---|---|---|---|
| 1 | Average sentence length | Mean number of words per sentence | Longer sentences suggest formal writing |
| 2 | Sentence length variation | Standard deviation of sentence lengths | Humans often mix short and long sentences |
| 3 | Average word length | Mean number of letters per word | Longer words suggest technical vocabulary |
| 4 | Vocabulary variety | Type-token ratio: unique words divided by all words | Low values mean repeated wording |
| 5 | Comma and punctuation rate | Commas, semicolons and colons per word | Reflects sentence structure |
| 6 | Long-word share | Share of words with 7 or more letters | Complements average word length |

The six features are passed to a random forest of 300 decision trees. Each tree votes with the share of AI texts at the leaf reached by the input, and the forest score is the average of the votes:

\[
P(AI \mid f) = \frac{1}{T} \cdot \sum_{k} p_k(f), \quad T = 300
\tag{4}
\]

**Figure 3: Architecture of Detector B (stylometric features with a random forest)**

## 4.4 Control: A Length-Only Detector

Detector L uses a single input, the logarithm of the number of words, with logistic regression. It is not meant to be a good detector. Its purpose is to reveal how much of the apparent signal can be explained by text length alone. A comparison that survives against this control is more credible, and a result that the control reproduces is suspect. As Section 7 shows, the control exposed a misleading first result.

## 4.5 Experimental Design

Four experiments change one thing at a time and hold everything else fixed, so that any difference can be traced to that change (Figure 4 and Table 4).

**Figure 4: Experimental design: transfer matrix (E1, E2) and the fairness and ablation experiments (E3, E4)**

**Table 4: Summary of experiments**

| ID | Question | What changes | What stays fixed | Output |
|---|---|---|---|---|
| E1 | How good is each detector on data like its training data? | Detector | Train and test domain | Matrix diagonal |
| E2 | How much does accuracy fall on unseen domains? (RQ1) | Test domain | Detector, training domain, seeds | Full 5 × 5 matrix and average drop |
| E3 | Are ELL essays flagged more often? (RQ2) | Writer group | Detector, training data, cutoff | False positive rate per group with intervals |
| E4 | Which kind of signal does Detector B rely on? | Feature group removed | Everything else | Change in AUROC per removed group |

For E1 and E2 each detector is trained on one domain at a time and tested on the held-out part of every domain, giving a 5 × 5 matrix whose diagonal is in-domain performance. For E3 each detector is trained on all HC3 domains and run on human-only essays from the two groups; every essay in this set is human, so every AI label is a false positive. For E4, Detector B is retrained with one feature group removed at a time (sentence length, vocabulary variety, punctuation, word length), trained on all domains and tested on the pooled held-out sets.

## 4.6 Evaluation Metrics

Let TP, FP, TN and FN denote true positives, false positives, true negatives and false negatives when AI is the positive class.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\tag{5}
\]

\[
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
\tag{6}
\]

\[
FPR = \frac{FP}{FP + TN}
\tag{7}
\]

The AUROC equals the probability that a randomly chosen AI text receives a higher score than a randomly chosen human text, with ties counted as one half. It does not depend on the cutoff, which makes it a fair way to compare domains and detectors. The false positive rate (Equation 7) is the share of human texts wrongly labelled AI at the default 0.5 cutoff; it is the error that harms people and the key metric for RQ2.

## 4.7 Statistical Methodology

Every number is the mean over three random seeds, and standard deviations over seeds are reported where relevant. For RQ2, each group's false positive rate is given with a 95% bootstrap confidence interval from 1,000 resamples of the essays.

The gap between groups, defined as:

\[
Gap = FPR(ELL) - FPR(non\text{-}ELL)
\tag{8}
\]

is given with a 95% bootstrap interval from 2,000 resamples of seed-averaged flag rates. The gap is computed on all essays and on a length-matched subset: essays are grouped into word-count bands and the same number of ELL and non-ELL essays is sampled from each band (868 essays in total).

The decision rule for H2 was fixed before the final runs: H2 is supported only if the confidence intervals of the two groups do not overlap; the interval of the gap is also examined directly.

## 4.8 Design Rationale

Classical models were chosen on purpose. They train in seconds, so every experiment can be repeated over several seeds on a free CPU runtime. Their behaviour can be explained to a non-specialist, which matters when the topic is the fairness of decisions about students. Finally, a deep model would add many uncontrolled factors, such as pre-training data, that would make the controlled comparison of this study harder to interpret.

## 4.9 Worked Example: How a Paragraph Is Scored

The following walk-through shows what happens when a paragraph is analysed, whether in the Python code or in the live demonstration. Consider a pasted paragraph of about fifty words.

- **Cleaning.** Literal backslash-n pairs and url placeholders are removed, spaces before punctuation are closed up, and whitespace is collapsed.
- **Detector A.** The text is lower-cased and split into words and word pairs. Only terms that exist in the 50,000-term vocabulary learned during training are kept. Each kept term receives a TF-IDF weight (Equations 1 and 2), the vector is scaled to unit length, and every term contributes its weight multiplied by its learned coefficient to the score. Positive contributions push towards AI and negative contributions towards human; the sum is passed through the logistic function (Equation 3) to give a probability.
- **Detector B.** The six style features are computed (for two sample paragraphs they are listed in Table 25 of Annexure E). Each of the 300 trees follows a path of yes/no questions about these numbers to a leaf, and the forest averages the leaf votes (Equation 4).
- **Detector L.** The number of words is converted to its logarithm and passed through a logistic regression.
- **Output.** Three probabilities are shown against the 0.5 cutoff. For Detector A the words with the largest contributions are highlighted, and for Detector B the six features are compared with the average human and AI training texts.

The three detectors can disagree, because they read different signals. That disagreement is informative: a text that Detector A calls human-like and Detector B calls AI-like is one on which the evidence is mixed, and the correct reaction is caution, not a verdict.

---

# 5. Dataset & Data Pre-processing

## 5.1 HC3 (Domain Generalization, RQ1)

The first dataset is the English part of HC3 [8], obtained from the Hugging Face Hub (dataset Hello-SimpleAI/HC3). Each record contains a question, one or more human answers and one or more ChatGPT answers. Five domains are used (Table 5).

**Table 5: HC3 domains used in this study**

| Domain | Content | Typical text |
|---|---|---|
| reddit_eli5 | Questions from an explain-like-I'm-five forum | Conversational, informal explanations |
| finance | Questions about personal finance and economics | Advice-style answers |
| medicine | Medical questions and answers | Health information and recommendations |
| open_qa | Open-domain factual questions | Encyclopedia-style definitions and facts |
| wiki_csai | Computer-science concepts | Definition-style explanations |

For every question the first human answer and the first ChatGPT answer of at least 30 words are kept, so each question contributes one human text (label 0) and one AI text (label 1) and every domain is perfectly balanced. Up to 800 questions are sampled per domain with a fixed random seed, which gives at most 8,000 texts in total (five domains × 800 questions × two answers); smaller domains contribute fewer.

## 5.2 PERSUADE 2.0 Essays (Fairness, RQ2)

The second dataset is PERSUADE 2.0, a corpus of argumentative essays written by U.S. students in grades 6 to 12 that includes demographic information such as English language learner status [3]. The distributed training file contains one row per annotated essay segment, not one row per essay. Whole essays were therefore reconstructed by concatenating the segments of each essay in order, using the essay identifier. This produced 8,426 essays with a median length of 355 words. Of these, 701 were marked as ELL, 6,928 as not ELL, and 75 had no ELL value (Table 6).

**Table 6: Essay dataset construction (PERSUADE 2.0)**

| Step | ELL | Non-ELL |
|---|---|---|
| Essays reconstructed from annotated segments (all, including 75 without ELL status) | 701 | 6,928 |
| Length filter: 150 to 600 words | kept | kept |
| Random sample for the study (seed 0) | 500 | 500 |
| Mean essay length in the sample (words) | 310 | 346 |
| Length-matched subset used for the gap test (both groups together) | 868 essays | |

All essays are human-written, so any AI label a detector gives them is a false positive. The ELL variable identifies students recorded as English language learners; it is an administrative label and is not identical to being a non-native speaker. This limitation is discussed in Section 7.

## 5.3 Data Audit and Cleaning

Before the final experiments the text was audited for shortcuts: properties a detector could use to separate the classes without reading the writing. Three artifacts were found (Table 7). The most striking was a formatting artifact: literal backslash-n character pairs, a trace of how the text was exported, occurred in 8.4% of ChatGPT answers and in none of the human answers (Figure 5). A detector could therefore identify some AI texts by this formatting alone.

**Table 7: Artifacts found by the data audit and the cleaning applied**

| Artifact | Where it occurred | Cleaning applied |
|---|---|---|
| Literal backslash-n sequences | 8.4% of ChatGPT answers; no human answers | Replaced by a space |
| Placeholder tokens such as url_0 | Human answers (links removed by the corpus authors) | Replaced by a space |
| Spaces before punctuation | Many human answers (tokenisation quirk) | Spaces removed before , . ; : ! ? ) and after ( [ |
| Irregular whitespace | Both classes | Collapsed to single spaces; text trimmed |

**Figure 5: Share of texts containing a literal backslash-n sequence, by class**

One cleaning function is applied to every text, the essays included, so that both classes and both essay groups receive identical treatment. All results in this report use the cleaned text. The cleaning function is listed in Annexure A.

## 5.4 Splits and Prevention of Leakage

Within each domain, 20% of the questions are held out for testing. Because the human and AI answers to one question share the same topic and wording, both answers always stay on the same side of the split. If they were separated, a detector could succeed by recognising the question rather than the author, which is a form of data leakage. The split is repeated for three random seeds (0, 1 and 2); the seeds change which questions are held out and the random initialisation of the random forest, while the sample of HC3 questions stays fixed.

## 5.5 Length Matching

Text length differs between groups in many corpora and also differs between HC3 answers and student essays. Because length alone separates the classes to a surprising degree (Section 7), group comparisons must control for it. Essays are grouped into word-count bands (150 to 200, 200 to 250, 250 to 300, 300 to 400, 400 to 500 and 500 to 600 words); in each band the same number of ELL and non-ELL essays is drawn at random, which equalises the length distributions of the two groups.

## 5.6 Ethical Handling of Data

Both corpora are public research resources. Essays are used only to compute aggregate statistics, individual essays are not published, and the essay file is not redistributed with the code. The study draws conclusions about detectors, never about individual students.

## 5.7 Illustrative Examples

Two short excerpts from the open_qa domain illustrate why transfer across domains is difficult. A human answer in this domain typically reads like an encyclopedia entry, for example: ‘Calcitonin (also known as thyrocalcitonin) is a 32-amino acid linear polypeptide hormone that is produced in humans primarily by the parafollicular cells …’. A ChatGPT answer to a similar kind of question is typically a plain, narrative summary, for example: ‘The Persian Wars were a series of conflicts fought between the Greek city-states and the Persian Empire in the 5th century BC.’

In the finance domain, by contrast, human answers are informal advice written by forum users and ChatGPT answers are formal and organised. A detector trained on finance can learn that formal and explanatory writing means AI. When it meets encyclopedia-style human text, which is also formal and explanatory, that rule misfires. This is the pattern seen in the weakest cell of the transfer matrix (Section 7.1) and in the error analysis (Section 7.7).

---

# 6. Implementation & Experimental Setup

## 6.1 Tools, Libraries and Frameworks

The project uses only free and open-source tools (Table 8). Python is the single implementation language for training and evaluation; the interactive demonstration adds JavaScript that runs in an ordinary web browser.

**Table 8: Software tools and libraries**

| Tool | Purpose | Use in this project |
|---|---|---|
| Python 3 | Programming language | All data handling, training and evaluation (Python 3.13 in the Colab runtime used) |
| pandas [15] | Tabular data | Loading, cleaning, balancing and grouping the data |
| NumPy [10] | Numerical arrays | Splits, random sampling, bootstrap, metrics |
| scikit-learn [17] | Machine learning | TfidfVectorizer, LogisticRegression, RandomForestClassifier, pipelines, AUROC, F1 and accuracy |
| matplotlib [11], seaborn [23] | Plotting | Heatmaps, bar charts and forest plots |
| huggingface_hub, Datasets [14] | Data access | Downloading the HC3 files |
| re (regular expressions) | Text processing | Sentence splitting, style features and cleaning |
| Streamlit [25] | Web application | Python version of the interactive app |
| JavaScript (browser) | Client-side code | Offline live detector that scores a pasted paragraph |
| Git and GitHub | Version control | Code, results and documentation |

## 6.2 Computing Environment

**Table 9: Computing environment**

| Item | Details |
|---|---|
| Training and experiments | Google Colaboratory [28], free CPU runtime; no GPU used |
| Run time | A full experiment run (three seeds, five domains) takes a few minutes; each model trains in seconds to a few minutes |
| Data access | Internet connection to download HC3 and PERSUADE 2.0 |
| Documents and demonstration | Windows laptop with a web browser (Chrome or Edge) |
| Memory and storage | Standard laptop or free-tier notebook resources; no special hardware |

## 6.3 Code Organisation

The repository is organised as a small Python package with command-line entry points (Table 10). Every table and figure of this report is produced by these scripts, so that results can be reproduced.

**Table 10: Repository modules**

| File | Responsibility |
|---|---|
| lab/data.py | Loads HC3 and the essays; contains the text-cleaning function |
| lab/models.py | Defines Detectors A, B and L, including the six stylometric features |
| lab/experiments.py | Question-level split, metrics, bootstrap, ablation and error analysis |
| lab/plots.py | Heatmap and fairness charts |
| run_experiments.py | Runs experiments E1 to E4 and saves tables and figures |
| extras.py | Reproduces the headline numbers: drops, length-matched gaps, strongest words |
| app.py | Streamlit application with five tabs |
| export_model.py | Exports trained detectors to a file used by the browser demonstration |

## 6.4 Implementation Process

The implementation followed the steps of the pipeline in Figure 1.

- **Data loading.** HC3 files are downloaded for the five domains; for each question one human and one ChatGPT answer of at least 30 words are kept; up to 800 questions per domain are sampled.
- **Cleaning.** The cleaning function is applied to every text. The essay corpus is rebuilt from segments, filtered to 150 to 600 words, and sampled (500 per group).
- **Splitting.** For each seed, 20% of questions per domain are held out.
- **Training.** For each detector and each training domain a model is fitted on the training part of that domain. For E3 and E4 the models are fitted on the training parts of all domains.
- **Evaluation.** Each model is evaluated on the held-out part of every domain (E1, E2). Essay flag rates are computed per group (E3) and the ablation is run (E4).
- **Statistics.** Bootstrap intervals and the length-matched comparison are computed (Section 4.7).
- **Analysis.** Tables and figures are generated, the strongest words of Detector A are extracted from its weights, and 24 misclassified texts are read manually.

## 6.5 Model Configuration

**Table 11: Detector and experiment settings**

| Component | Setting |
|---|---|
| Detector A: vectoriser | Lower-case; word tokens of two or more characters; unigrams and bigrams; minimum document frequency 2; sub-linear term frequency; at most 50,000 features |
| Detector A: classifier | Logistic regression, L2 regularisation (default strength), up to 1,000 iterations |
| Detector B: features | Six stylometric features (Table 3) |
| Detector B: classifier | Random forest, 300 trees, default settings, random state set to the seed |
| Detector L | Logistic regression on log(1 + number of words) |
| Split and seeds | 20% of questions per domain held out; seeds 0, 1 and 2 |
| Cutoff | 0.5 for all label-based metrics |
| Bootstrap | 1,000 resamples for group rates; 2,000 for the gap |

## 6.6 Training Process and Convergence

All three models are classical and are not trained for epochs with a stochastic optimiser, so loss-versus-epoch curves, which are central to deep-learning projects, are not applicable here. Logistic regression is solved by an iterative optimiser that converged within the permitted 1,000 iterations, and the random forest has no iterative loss. The relevant evidence of training quality is therefore the held-out performance reported in Section 7 and its stability across seeds, whose standard deviation is below 0.01 for the average AUROC drop.

## 6.7 Interactive Demonstration

Two versions of an interactive demonstration were built. The first is a Streamlit application with five tabs: Try it (paste a paragraph to see each detector's score), Transfer (RQ1 heatmaps), Fairness (RQ2 chart), Ablation and errors, and About. It can open on saved results or run new experiments.

During development the Streamlit application could not be run on the presentation laptop, because the operating system's application-control policy blocked compiled Python packages, and a temporary tunnel used to serve it from Colab failed to deliver the interface files. A second version was therefore built that needs only a web browser (Figure 6). A script, export_model.py, trains the three detectors on the HC3 data and writes their parameters to a single file: the TF-IDF vocabulary, inverse document frequencies and logistic-regression weights of Detector A, the node arrays of the 300 trees of Detector B, and the coefficients of Detector L. A page, detector.html, re-implements the cleaning, feature computation and inference in JavaScript.

**Figure 6: Architecture of the browser-based live detector**

In a verification run on five test texts with a demonstration model, the browser implementation reproduced the scikit-learn probabilities: the largest difference was about 1.2 × 10⁻⁷ for Detector A and zero for Detectors B and L. For a pasted paragraph the page shows each detector's probability against the 0.5 cutoff; for Detector A it highlights the words that push the score towards AI or towards human; for Detector B it shows the six style numbers against the average human and AI training texts. Because the models run entirely in the browser, the demonstration works offline and the text never leaves the computer. The live models are trained on all HC3 data and therefore are not identical to the models trained on the training splits for the tables of Section 7.

## 6.8 Reproducibility

Random seeds are fixed for the splits and the random forest; the HC3 sample is fixed; the cleaning function is shared by training, evaluation and demonstration; and all numbers are written to result files by the scripts. The commands needed to reproduce the results are listed in Annexure C.

## 6.9 Challenges Encountered

- **Unsuitable essay data.** The first essay sets differed greatly in length, which produced a misleading result (Section 7.3). Both groups were rebuilt from a single corpus.
- **Hidden formatting shortcuts.** Found by the audit and removed (Section 5.3).
- **Essays split into pieces.** The essay file had to be reassembled.
- **Stale code in a notebook session.** Modules edited after import were not reloaded; the session was restarted and the final runs were repeated.
- **Restricted laptop.** Solved by the browser-based demonstration.

## 6.10 Use of AI Assistance

AI assistants were used during the project to help write code and draft documents. The datasets, experimental design, interpretation and decisions were reviewed by the author, every reported number comes from running the code, and the data audit, the length control and the repeated runs were included precisely so that outputs are verified instead of trusted.

## 6.11 Testing and Validation of the Code

Several checks were used to make sure that the code does what the report claims. A small synthetic dataset, generated by the repository, allows the whole pipeline to be run in seconds without downloading data; it was used to test the code and is clearly labelled in the application as meaningless for findings. The split function was checked to ensure that no question appears on both sides, that every domain is balanced, and that the same seed always gives the same split. All metrics were checked to lie in their valid ranges, and the length-only control was used as a sanity check because its behaviour is predictable.

The browser implementation was validated against scikit-learn, as described in Section 6.7. Finally, the experiments were rerun after each data correction (the change of essay corpus, the cleaning of artifacts and the removal of placeholder tokens), and the final numbers were taken from the last complete run only. Earlier results were kept solely to report the sensitivity analysis of Section 7.4.

---

# 7. Results & Performance Analysis

## 7.1 RQ1: Transfer Across Domains

Table 12 summarises the average in-domain AUROC (the diagonal of the transfer matrix) and the average AUROC on unseen domains (the off-diagonal cells), for the three detectors. The complete matrices are shown as heatmaps in Figures 7 and 8 and as tables in Tables 13 and 14.

**Table 12: Mean AUROC in-domain and on unseen domains (mean over 3 seeds)**

| Detector | In-domain | Unseen domains | Drop (mean ± SD) |
|---|---:|---:|---:|
| A (TF-IDF, logistic regression) | 0.989 | 0.886 | 0.103 ± 0.002 |
| B (stylometric, random forest) | 0.953 | 0.798 | 0.156 ± 0.008 |
| L (length only) | 0.761 | 0.761 | not applicable |

H1 is supported. Both detectors perform very well on data like their training data (0.989 and 0.953) and clearly worse on unseen domains, and the style-based detector loses more (0.156 against 0.103). The standard deviations over seeds are tiny (0.002 and 0.008), so the drops are stable. The length-only control has a drop of exactly zero by construction: a one-feature model ranks texts in the same order whichever domain it was trained on, so its AUROC depends only on the test domain. It is shown as a reference line, not as a transfer result. Both detectors score well above the control on unseen domains (0.886 and 0.798 against 0.761), so they use more than length, but length alone separates the classes with an AUROC of 0.76, a notable shortcut.

**Figure 7: AUROC transfer matrix of Detector A (rows: training domain; columns: test domain)**

**Figure 8: AUROC transfer matrix of Detector B (rows: training domain; columns: test domain)**

**Table 13: AUROC matrix of Detector A (diagonal in bold; values read from Figure 7, rounded to two decimals)**

| Train \ Test | finance | medicine | open_qa | reddit_eli5 | wiki_csai |
|---|---:|---:|---:|---:|---:|
| finance | **1.00** | 1.00 | 0.72 | 0.90 | 0.83 |
| medicine | 0.97 | **1.00** | 0.81 | 0.88 | 0.80 |
| open_qa | 0.77 | 0.98 | **0.97** | 0.82 | 0.84 |
| reddit_eli5 | 0.99 | 1.00 | 0.82 | **0.99** | 0.85 |
| wiki_csai | 0.92 | 0.99 | 0.90 | 0.91 | **0.99** |

**Table 14: AUROC matrix of Detector B (diagonal in bold; values read from Figure 8, rounded to two decimals)**

| Train \ Test | finance | medicine | open_qa | reddit_eli5 | wiki_csai |
|---|---:|---:|---:|---:|---:|
| finance | **0.96** | 0.83 | 0.42 | 0.82 | 0.58 |
| medicine | 0.84 | **1.00** | 0.93 | 0.90 | 0.75 |
| open_qa | 0.55 | 0.98 | **0.99** | 0.85 | 0.73 |
| reddit_eli5 | 0.92 | 0.99 | 0.85 | **0.93** | 0.80 |
| wiki_csai | 0.64 | 0.90 | 0.81 | 0.86 | **0.89** |

The drop is uneven across domain pairs. For Detector A the worst off-diagonal cell is finance to open_qa (0.72), and the other weak transfers (open_qa to finance, 0.77; medicine to wiki_csai, 0.80) are also moderate; most cells stay above 0.8. Detector B has weaker spots: trained on finance and tested on open_qa it scores 0.42, below the 0.5 expected from random guessing, which means that the style rule learned on finance advice text is reversed on encyclopedia-style text. Other weak cells are finance to wiki_csai (0.58), open_qa to finance (0.55) and wiki_csai to finance (0.64). Averaging each row and column gives a clearer picture (Table 15, Figure 10 and Annexure B): medicine is the easiest domain to transfer to for both detectors, and open_qa is the hardest target for Detector A, while wiki_csai and finance are the hardest targets for Detector B. As a source of training data, finance transfers worst for Detector B.

**Table 15: Mean AUROC on the other four domains, by training domain, and on each domain as a target (averages of the off-diagonal cells of Tables 13 and 14)**

| Domain | As training: A | As training: B | As target: A | As target: B |
|---|---:|---:|---:|---:|
| finance | 0.86 | 0.66 | 0.91 | 0.74 |
| medicine | 0.86 | 0.85 | 0.99 | 0.92 |
| open_qa | 0.85 | 0.78 | 0.81 | 0.75 |
| reddit_eli5 | 0.92 | 0.89 | 0.88 | 0.86 |
| wiki_csai | 0.93 | 0.80 | 0.83 | 0.72 |

**Figure 9: Mean AUROC in-domain and on unseen domains for Detectors A, B and the length-only control**

**Figure 10: Mean AUROC when each domain is the test domain (average over the four other training domains)**

## 7.2 RQ2: Human Essays Flagged as AI

Table 16 and Figure 11 show how often human essays from each group were labelled AI at the 0.5 cutoff. Every essay in this set is human-written, so these rates are false positive rates.

**Table 16: Human essays flagged as AI at the 0.5 cutoff (500 essays per group; 95% bootstrap intervals)**

| Detector | Non-ELL essays | ELL essays |
|---|---|---|
| A (TF-IDF) | 8.2% (5.9 to 10.5) | 10.3% (7.9 to 13.2) |
| B (stylometric) | 41.7% (37.8 to 45.7) | 41.5% (37.4 to 45.4) |

**Figure 11: Human essays wrongly flagged as AI, by writer group, with 95% bootstrap confidence intervals**

For Detector A the intervals of the two groups overlap, and for Detector B they overlap almost completely. Under the pre-stated rule, H2 is not supported for either detector. The gap was also examined directly (Table 17 and Figure 12). All four intervals include zero, in both the full and the length-matched sets. The intervals are about ±5 to 6 points wide, so a small gap, up to about 6 points, cannot be ruled out; the evidence supports "no detectable gap", not "no gap".

**Table 17: Gap in flag rate, ELL minus non-ELL (percentage points, 95% bootstrap interval)**

| Detector | Essay set | Gap | 95% interval |
|---|---|---:|---|
| A | All essays (n = 1,000) | +2.1 | −1.3 to +5.4 |
| A | Length-matched (n = 868) | +3.2 | −0.7 to +6.8 |
| B | All essays (n = 1,000) | −0.2 | −6.0 to +5.5 |
| B | Length-matched (n = 868) | +0.4 | −5.5 to +6.6 |

**Figure 12: Gap in false positive rate between ELL and non-ELL essays with 95% bootstrap intervals**

The more important finding is the overall level of error. Detector A wrongly flags about 8 to 10% of human essays and Detector B about 42%, in both groups. Detector B behaves almost like a coin flip on this kind of text. A likely contributor is genre shift: the detectors were trained on short question-answer texts, whereas essays are longer, so features such as sentence length and the type-token ratio lie outside the range seen in training. Whatever the cause, using either detector at the default cutoff would produce many false accusations for every group of students.

## 7.3 A Misleading First Result

The final fairness comparison is the product of correcting an earlier mistake. The first comparison used a different set of native-speaker essays (200 essays, median length about 60 words) against ELL essays (500 essays, median length about 396 words), with a single seed. On those data, Detector B flagged 56.2% of ELL essays but only 7.5% of native essays, which looks like strong bias (Table 18, Figure 13).

**Table 18: First, misleading comparison (single seed; native essays n = 200, ELL essays n = 500)**

| Detector | Native essays flagged | ELL essays flagged |
|---|---:|---:|
| A (TF-IDF) | 7.5% | 8.8% |
| B (stylometric) | 7.5% | 56.2% |
| L (length only) | 7.0% | 97.4% |

**Figure 13: First comparison with mismatched essay lengths: flag rates by group and detector**

The control exposed the problem. The length-only detector, which knows nothing about language background, flagged 97.4% of the long essays and 7.0% of the short ones, reproducing the gap. Detector B depends on sentence-length and vocabulary features that change with text length, so its apparent bias was largely a length effect. Rebuilding both groups from one corpus with similar lengths removed the effect. In the final data both groups are longer than typical HC3 human answers, so Detector L flags essentially every essay and is uninformative as a group comparison, which shows that all essays face the same length shift.

## 7.4 Sensitivity to Data Pre-processing

The data audit led to a second finding. On the uncleaned text, Detector B showed a length-matched gap of +7.8 points with a 95% interval of +1.5 to +13.9, which excludes zero. After cleaning, the gap was +0.4 points (−5.5 to +6.6). Figure 14 compares the intervals before and after cleaning. The project did not isolate which of the artifacts caused the earlier gap, so this result is exploratory. It does show that an apparent difference between writer groups can depend on formatting details of the data, and it is a reason for auditing data before reporting group differences.

**Figure 14: Gap between ELL and non-ELL flag rates before and after data cleaning (95% intervals)**

## 7.5 Ablation Study (E4)

To see which kind of signal Detector B relies on, one feature group at a time was removed and the model retrained on all domains. The full model reaches an AUROC of 0.929 on the pooled held-out sets (Table 19 and Figure 15).

**Table 19: Ablation of Detector B (mean AUROC over 3 seeds, pooled held-out sets)**

| Feature group removed | AUROC | Drop from full model |
|---|---:|---:|
| None (full model) | 0.929 | - |
| Sentence length (features 1 and 2) | 0.864 | 0.065 |
| Vocabulary variety (feature 4) | 0.884 | 0.044 |
| Word length (features 3 and 6) | 0.916 | 0.013 |
| Punctuation (feature 5) | 0.925 | 0.004 |

**Figure 15: Drop in AUROC when each feature group of Detector B is removed**

Detector B relies mainly on sentence-length features and on vocabulary variety; punctuation adds almost nothing. Because the features are correlated, removing a group does not isolate its causal contribution, since other features can partly compensate; the ablation shows what the model leans on, not what causes detection. It is consistent with the length confound of Section 7.3, because both sentence length and the type-token ratio depend on text length.

## 7.6 Words that Detector A Relies On

The coefficients of Detector A, trained on all domains, show what it leans on. Its strongest human-leaning terms are conversational: but, you, my, just, what, hi, probably. Its strongest AI-leaning terms are formal and hedged: such, such as, used, variety of, may, help, including, it is important. Human answers in these data are casual and personal, whereas ChatGPT answers are formal and explanatory, and Detector A learned that register difference. This also explains why formal human writing is at risk of being flagged.

## 7.7 Error Analysis

Twenty-four misclassified texts were read manually (Table 20). These observations are qualitative and were not tested as explanations.

**Table 20: Qualitative error analysis**

| Group of errors | Observation | Interpretation |
|---|---|---|
| Human essays flagged as AI (both writer groups) | Mostly source-based argumentative essays that summarise or quote provided articles in an explanatory tone; many contain spelling errors | Explanatory register resembles ChatGPT answers; surface polish is not the trigger |
| Human texts flagged in the worst transfer cell (finance to open_qa) | Encyclopedia-style definitions of the form ‘X is a …, also known as …’ | Style cues learned on advice-like finance text do not carry over |
| AI texts missed in the same cell | Short factual ChatGPT answers | Short plain answers lack the markers learned elsewhere |

## 7.8 Discussion

- **Generalization.** Both detectors transfer imperfectly but not catastrophically. The word-based detector transfers better than the style-based one, possibly because ChatGPT's word habits are more stable across domains than its sentence-level style; this was not tested directly.
- **Length is a shortcut.** A one-feature length model separates the classes with AUROC 0.76. Any comparison between writer groups must control for length.
- **Fairness.** No group gap was detectable, which differs from the bias reported for perplexity-based commercial detectors by Liang et al. [13]. The detectors here are not perplexity-based, and the groups (ELL and non-ELL U.S. students) differ from theirs, so the results are not in conflict. The detectors are not shown to be fair: the intervals are wide, and a small gap, or a gap under other conditions, remains possible.
- **Reliability.** Genre shift from question answers to essays produces high false positive rates for everyone, so reporting accuracy on in-distribution data would have hidden the main practical risk.

## 7.9 Limitations and Threats to Validity

**Table 21: Limitations and their effect on the conclusions**

| Limitation | Why it matters | Effect on conclusions |
|---|---|---|
| One AI source (ChatGPT) | Newer or different models write differently | Results may not hold for other generators |
| Simple models | Fine-tuned transformers may behave differently | Conclusions concern classical detectors |
| Three seeds | Seeds vary the splits, not the sampled data | Spread may be understated |
| ELL label is administrative | Not identical to being a non-native speaker | RQ2 concerns ELL status only |
| Groups differ in other ways | Prompts, grade level and school may differ | An association, not a cause |
| Essays rebuilt from segments | Unannotated text, if any, is missing | Essay lengths may be slightly understated |
| Single 0.5 cutoff | Other operating points change error rates | Rates apply to the default cutoff |
| Possible further artifacts | HC3 may contain shortcuts not found by the audit | Scores may be somewhat optimistic |

## 7.10 Answering the Research Questions

**RQ1: How much does accuracy drop on unseen domains?**

The average AUROC fell by 0.103 for Detector A and by 0.156 for Detector B, with standard deviations of 0.002 and 0.008 over seeds. The drop is real, stable and uneven; the worst cells fall to 0.72 for Detector A and 0.42 for Detector B. The caveat is that only five domains and one generator were tested.

**RQ2: Are ELL essays flagged more often?**

Not detectably. The estimated gaps are +2.1 and −0.2 percentage points for Detectors A and B, with intervals that include zero, and the length-matched estimates are similar. The caveat is that the intervals are about ±5 to 6 points wide, that ELL status is an administrative label, and that both groups differ from the training texts in genre.

## 7.11 Practical Implications

The results suggest several practical lessons for any institution that considers using AI-text detectors. These are implications of this study's evidence, not rules.

- **A score is not proof.** Both detectors flagged large shares of human essays, so a flag should start a conversation, not end one.
- **Test on local data.** Performance dropped when the domain changed, so a detector should be evaluated on text from the courses and levels where it will be used.
- **Check the error rate on human text.** The false positive rate at the planned cutoff should be measured on known human writing, and cutoffs that give a very low rate should be preferred even if some AI text is missed.
- **Monitor groups.** Error rates should be compared across groups of students, with confidence intervals and a control for text length.
- **Keep humans in charge.** Decisions should rest on several kinds of evidence, including the student's own explanation and drafting history.

---

# 8. Conclusion & Future Scope

## 8.1 Summary of Findings

This project asked whether AI-text detectors keep working outside the data they were trained on and whether they wrongly flag essays by English language learners. Two interpretable detectors and a length-only control were trained on human and ChatGPT answers from five HC3 domains and tested across domains and on 1,000 human essays from PERSUADE 2.0.

- **RQ1 (generalization):** supported. Mean AUROC fell from 0.989 to 0.886 for Detector A and from 0.953 to 0.798 for Detector B on unseen domains. The drops were stable across seeds and uneven across domain pairs; some cells fell to or below chance.
- **RQ2 (fairness):** not supported. No detectable difference in false-flag rates between ELL and non-ELL essays was found (A: 10.3% against 8.2%; B: 41.5% against 41.7%), but a gap of up to about 6 points cannot be excluded.
- **Practical risk:** both detectors wrongly flag many human essays (about 8 to 10% and 42%), so they should never be the only evidence against a student.
- **Methodological lesson:** a naive comparison suggested strong bias that was caused by essay length and by formatting artifacts. A length-only control and a data audit were what revealed it.

## 8.2 Contributions

- A cross-domain evaluation of two transparent detectors with a length-only control.
- A controlled ELL versus non-ELL comparison with bootstrap intervals and length matching.
- A documented data audit showing how artifacts and length can create or hide apparent bias.
- A reproducible code package and an offline interactive demonstration that scores any pasted paragraph.

## 8.3 Future Scope

Several extensions would strengthen and broaden the study (Table 22).

**Table 22: Future scope**

| Direction | Description | Expected benefit |
|---|---|---|
| Newer AI models | Add text from several recent language models, not only ChatGPT | Tests whether findings hold for current generators |
| Fine-tuned detectors | Compare with fine-tuned transformer classifiers on the same protocol | Shows whether stronger models transfer or discriminate differently |
| Operating points | Study cutoffs calibrated to a fixed false positive rate, such as 1% on held-out human text | Tells institutions what error rate a practical setting gives |
| More essay sources and groups | Use more corpora and additional group definitions | Reduces the risk that conclusions depend on one corpus |
| Artifact analysis | Isolate which artifacts change group gaps | Explains the sensitivity to pre-processing |
| Explanations for users | Show students and teachers why a text received its score | Supports transparent, contestable decisions |

In summary, detection scores are statistical guesses, not proof. Until detectors are evaluated across domains, controlled for length, and audited for shortcuts, they should not be used as the sole basis of decisions about people.

## 8.4 Learning Outcomes

Beyond the findings, the project produced practical learning outcomes. Technically, it exercised the full machine-learning workflow: collecting and cleaning data, building features, training linear and ensemble classifiers, choosing and interpreting metrics, and packaging the work as code and an interactive application. Methodologically, it showed the value of controls and audits: three of the most important results of the study, the length confound, the formatting shortcut and the sensitivity to pre-processing, were discovered only because a control or an audit was included.

The project also taught the discipline of reporting honestly. A hypothesis that was not supported was reported as such, a small gap that could not be ruled out was described as unresolved, and earlier mistakes were documented instead of hidden. Finally, it gave practice in communicating technical results to different audiences, through this report, a research paper, a presentation and a live demonstration.

---

# 9. References

1. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
2. Brown, T. B., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems, 33*, 1877–1901.
3. Crossley, S. A., Baffour, P., Tian, Y., et al. (2022). The persuasive essays for rating, selecting, and understanding argumentative and discourse elements (PERSUADE) corpus 1.0. *Assessing Writing, 54*, 100667.
4. Efron, B. (1979). Bootstrap methods: another look at the jackknife. *The Annals of Statistics, 7*(1), 1–26.
5. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
6. Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*, 665–673.
7. Gehrmann, S., Strobelt, H., and Rush, A. M. (2019). GLTR: Statistical detection and visualization of generated text. *Proceedings of ACL 2019: System Demonstrations*, 111–116.
8. Guo, B., Zhang, X., Wang, Z., et al. (2023). How close is ChatGPT to human experts? Comparison corpus, evaluation, and detection. arXiv:2301.07597.
9. Hardt, M., Price, E., and Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems, 29*.
10. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362.
11. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95.
12. Kirchenbauer, J., et al. (2023). A watermark for large language models. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202.
13. Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., and Zou, J. (2023). GPT detectors are biased against non-native English writers. *Patterns, 4*(7), 100779.
14. Lhoest, Q., et al. (2021). Datasets: A community library for natural language processing. *Proceedings of EMNLP 2021: System Demonstrations*, 175–184.
15. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61.
16. Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., and Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202.
17. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
18. Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., and Lawrence, N. D. (Eds.). (2009). *Dataset Shift in Machine Learning*. MIT Press.
19. Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang, W., and Feizi, S. (2023). Can AI-generated text be reliably detected? arXiv:2303.11156.
20. Salton, G., and Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523.
21. Solaiman, I., et al. (2019). Release strategies and the social impacts of language models. arXiv:1908.09203.
22. Stamatatos, E. (2009). A survey of modern authorship attribution methods. *Journal of the American Society for Information Science and Technology, 60*(3), 538–556.
23. Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software, 6*(60), 3021.
24. Weber-Wulff, D., et al. (2023). Testing of detection tools for AI-generated text. *International Journal for Educational Integrity, 19*, Article 26.
25. Streamlit Inc. Streamlit documentation. https://docs.streamlit.io (accessed November 2026).
26. Hugging Face. Dataset card: Hello-SimpleAI/HC3. https://huggingface.co/datasets/Hello-SimpleAI/HC3 (accessed November 2026).
27. Hugging Face. huggingface_hub documentation. https://huggingface.co/docs/huggingface_hub (accessed November 2026).
28. Google. Colaboratory documentation. https://colab.research.google.com (accessed November 2026).
