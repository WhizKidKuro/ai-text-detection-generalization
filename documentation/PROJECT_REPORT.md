# Cross-Domain Generalization and Fairness of AI-Text Detectors

## Abstract
We train two classic detectors (TF-IDF + logistic regression, and stylometric features + random forest) on human vs ChatGPT answers from HC3, test them on unseen domains, and run them on human essays by English language learners (ELL) and other students from PERSUADE 2.0. Performance drops moderately on unseen domains (AUROC 0.99→0.89 and 0.95→0.80). We find no reliable difference in false-flag rates between groups, but both detectors wrongly flag many human essays (~8–10% and ~42%), so neither is safe for misconduct decisions.

## 1. Questions and hypotheses
**RQ1:** how much does accuracy fall on unseen domains? **H1:** clearly lower. **RQ2:** are ELL essays flagged as AI more often? **H2:** higher false positive rate (FPR).

## 2. Method
- **Data:** HC3 (5 domains; one human and one ChatGPT answer per question, ≥30 words, up to 800 questions per domain). Essays: 500 ELL and 500 non-ELL essays (150–600 words) from PERSUADE 2.0, stitched from annotated pieces.
- **Cleaning:** removed shortcuts found by audit: literal `\n` only in ChatGPT answers, spaces before punctuation in human answers, `url_N` placeholders.
- **Detectors:** A (word and bigram TF-IDF + LogReg); B (six style features + random forest); L (text length only, as a control).
- **Splits:** per domain, 20% of questions held out; 3 seeds.
- **Experiments:** E1 in-domain, E2 cross-domain, E3 fairness (FPR per group at a 0.5 cutoff, bootstrap 95% CI, plus a length-matched check and a fixed-1%-FPR cutoff in `extras.py`), E4 ablation of Detector B.

## 3. Results
**RQ1 (AUROC, mean over 3 seeds):**
| Detector | In-domain | Unseen | Drop ± std |
|---|---|---|---|
| A | 0.989 | 0.886 | 0.103 ± 0.002 |
| B | 0.953 | 0.798 | 0.156 ± 0.008 |
| L (length) | 0.761 | 0.761 | n/a (a one-feature model ranks texts identically in every domain) |

H1 is supported. The weakest transfers involve open_qa and wiki_csai (see heatmaps).

**RQ2 (human essays flagged as AI, 0.5 cutoff):**
| Detector | Native | Non-native | Gap, all essays (95% CI) | Gap, length-matched (95% CI) |
|---|---|---|---|---|
| A | 8.2% | 10.3% | +2.1 pts (−1.3, +5.4) | +3.2 pts (−0.7, +6.8) |
| B | 41.7% | 41.5% | −0.2 pts (−6.0, +5.5) | +0.4 pts (−5.5, +6.6) |

H2 is not supported: all intervals include 0, though a small gap (up to ~6 points) cannot be ruled out. A naive first comparison showed B flagging 56% vs 7% of essays, but it compared essays of very different length; a length-only detector flagged 97% of the long essays, so that gap was mostly length and genre.

**E4 ablation (Detector B, AUROC drop when a feature group is removed; full model 0.929):** sentence length 0.065, vocabulary variety 0.044, word length 0.013, punctuation 0.004.

**Detector A's strongest words:** human-leaning: but, you, my, just, would, what, hi, probably; AI-leaning: such, such as, used, variety of, may, help, including, it is important.

## 4. Error analysis
- Human essays flagged as AI (both groups) are mostly source-based argumentative essays that summarize or quote the provided articles in an explanatory register (Venus, Face on Mars, car-free cities, facial coding). Many contain spelling errors, so polish is not the trigger.
- In the worst transfer cell (finance→open_qa), flagged human texts are Wikipedia-style definitions ("X is a …, also known as …"); missed AI texts are short factual ChatGPT answers. Style cues learned on advice-like finance text do not carry over.

## 5. Limitations
One AI source (ChatGPT) and simple models; 3 seeds only change splits, not the sampled data; ELL status means students identified as English learners, not all non-native writers; groups differ in other ways (prompts, grade); essays were stitched from annotated pieces; genre shift from HC3 answers to essays inflates false flags for everyone; HC3 may hold artifacts we did not find.

## 6. Ethics
A false accusation is a serious harm. At the default cutoff both detectors flag many human essays, so they must never be the only evidence against a student.

## 7. How AI helped build this
AI assistance wrote the code and drafts; every claim was checked against runs, which exposed three data artifacts and a length confound that changed the conclusions.

## 8. Demo script (3 minutes)
1) Paste a human essay and an AI answer into **Try it**. 2) Show the transfer heatmap. 3) Show the fairness chart and explain the length lesson. 4) State the ethics point.

## References (verify before citing)
Guo et al. (2023), HC3. Liang et al. (2023), GPT detectors are biased against non-native English writers. Crossley et al. (2022), PERSUADE corpus.
