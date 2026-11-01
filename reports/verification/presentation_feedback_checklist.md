# Professor feedback: verified checklist for the final presentation

This checklist reconciles Muhammet's message, the fuller `TeamUltron_Project2 (1).zip`, the consolidated repository results, and the TextEE paper. It does not change the presentation or published results.

## 1. Academic formulas

Use typeset equations rather than screenshots of code or literal LaTeX commands. Define a correct prediction as an exact span and role attached to the same event. Pool counts across events for micro scores.

```latex
P = \frac{TP}{TP+FP}, \qquad
R = \frac{TP}{TP+FN}, \qquad
F_1 = \frac{2PR}{P+R}
    = \frac{2TP}{2TP+FP+FN}.
```

For document bootstrap resamples, recompute the pooled micro F1 in each resample; do not average document F1 scores:

```latex
\widehat{F}_1^{*(b)} = F_1(D^{*(b)}), \qquad b=1,\ldots,B,
```

```latex
\mathrm{CI}_{95\%} =
\left[Q_{0.025}(\widehat{F}_1^*),\;
      Q_{0.975}(\widehat{F}_1^*)\right].
```

For a paired comparison, resample the same document IDs for both models and compute a difference in each replicate:

```latex
\Delta^{*(b)} = F_{1,B}^{*(b)} - F_{1,A}^{*(b)}.
```

Explain the bootstrap in one sentence: "We resample documents so events from the same article remain together." Report the number of resamples, seed, and interval method consistently.

## 2. Academic context and saturation

Replace "same setting" with:

> TextEE provides a related CASIE event-argument-extraction benchmark. Our experiment also supplies gold triggers, but uses a different split, preprocessing pipeline, and event-specific scoring protocol. We present the published results as context, not a directly comparable leaderboard.

TextEE Table 5 reports CASIE EAE results averaged over five splits. Its appendix gives 701/149/149 documents per split, compared with our corrected 799/100/100 split. It uses RoBERTa-large for classification models and BART-large for generation models.

Most importantly, distinguish its metric columns:

| TextEE model | CASIE AC F1 | CASIE AC+ F1 |
|---|---:|---:|
| TagPrime-CR | 69.2 | 66.1 |
| PAIE | 65.7 | 64.0 |
| X-Gear | 63.4 | 59.3 |
| BART-Gen | 60.0 | 58.3 |
| DEGREE | 59.0 | 54.7 |
| DyGIE++ | 56.0 | 51.5 |
| OneIE | 55.3 | 27.7 |
| EEQA | 54.0 | 49.0 |

TextEE AC does not require matching the argument's attached trigger. AC+ does. Our exact event-ID scorer is closer to the event-attachment requirement of AC+, although tokenization and other protocol differences still preclude direct ranking. Do not place our 61.96 alongside their AC values as if the metrics were identical.

"Published models score 54–69" is a selected-model range, not the full table range: TextEE also reports CASIE AC scores below 54. Identify the selected models or omit the range.

Suggested conclusion:

> The reported systems still make substantial exact-span and role errors, particularly on difficult roles and distant arguments. These results indicate considerable remaining room for improvement; our study does not establish a performance ceiling.

Our supplied five-model test range is 52.51–61.96 F1 points. The consolidated table, including the additional feature CRF, ranges from 51.88–61.96. Label which comparison is being presented.

Source: Huang et al. (2024), [TextEE paper](https://aclanthology.org/2024.findings-acl.760.pdf), Table 5 and the evaluation/split definitions. The figures above are selected values from that historical paper, not a claim about the current state of the art.

## 3. Algorithm classes

For the original supplied-model comparison:

| Class | Models |
|---|---|
| Classical structured prediction | CRF |
| General pretrained transformers | BERT cased, BERT uncased, RoBERTa |
| Domain-adapted pretrained transformer | SecureBERT |

For the full consolidated submission, also include:

- Neural sequence model with structured decoding: BiLSTM-CRF.
- General pretrained transformer: DeBERTa-v3-base.
- Role-conditioned formulation: the DeBERTa adaptation, shown as a methodological variant rather than an unrelated model family.
- Supplied and additional CRF are two runs of the same architecture. There are eight validation rows and nine test rows, not nine independent model families.

## 4. Confidence intervals and significance

The current `results/validation_summary.csv` and `results/test_summary.csv` already contain document-bootstrap 95% intervals for every listed model. The consolidated notebook uses 1,000 resamples. Muhammet's archived README separately mentions 2,000 resamples for its model intervals and 5,000 for paired comparisons. Avoid mixing these procedures without labeling them.

The archived README reports BERT uncased minus BERT cased as a paired-bootstrap difference of about +0.034, a 95% interval of [+0.013, +0.056], and p=0.002. However, the included `compare_models()` implementation returns only the observed F1 difference and `b_better_fraction`. It does not return that difference interval or a defined p-value. A generating script for the README's p-value was not identified in the inspected code.

The BERT claim has now been audited in [the BERT significance audit](bert_significance.md). With 5,000 paired document bootstrap samples, sorted document IDs and seed 42, uncased wins 4,995 times, loses 5 times and has no ties. Twice the smaller sign-tail fraction reproduces 0.002; the paired difference interval [0.0125825670, 0.0562368641] reproduces the archived rounded interval [+0.013, +0.056]. This is a reproducible calculation consistent with the archive, while the original generating script remains unidentified.

For the revised slides, prefer the separately defined two-sided document randomization test: 100,000 randomizations, seed 42, p=0.003599964. The 100,000-sample paired document bootstrap difference interval is [+0.0124943758, +0.0556281405]. Suggested slide wording: **uncased +3.43 F1 points, paired document bootstrap 95% CI [+1.25, +5.56], two-sided document randomization p=0.0036**. These are an unadjusted retrospective comparison of two fitted models. Do not relabel a bootstrap win fraction as a p-value. For several exploratory comparisons, identify them as exploratory or state a multiple-comparison procedure.

Document bootstrap describes uncertainty over evaluation documents conditional on these fitted models. It does not measure variability across training seeds.

## 5. Cased and uncased models

The reported supplied test scores support the observed difference:

```text
BERT cased:   53.5363 F1 points
BERT uncased: 56.9637 F1 points
Difference:   +3.4274 F1 points
```

Suggested wording:

> In these single-seed BERT runs, uncased BERT outperformed cased BERT by 3.43 F1 points. This does not establish that removing case is generally better: the pretrained models and vocabularies also differ. Our top two supplied systems, RoBERTa and SecureBERT, preserve case.

Do not describe this as a clean causal ablation of casing. The audited p-value and interval above support this fitted-model comparison; they do not establish repeatability across training seeds.

## 6. Latency and efficiency

The fuller archive includes `results/latency_gpu.json` and `results/latency_crf.json`. The accompanying README identifies the GPU as a T4; the JSON itself records `cuda:0`.

| Model | Reported hardware | Single-event median / p90 | Batch throughput |
|---|---|---:|---:|
| Supplied CRF | CPU | 12.3 / 16.4 ms | 106.0 events/s |
| BERT cased | T4 GPU | 20.6 / 23.9 ms | 143.6 events/s |
| BERT uncased | T4 GPU | 21.4 / 25.1 ms | 144.7 events/s |
| RoBERTa | T4 GPU | 20.7 / 25.2 ms | 145.4 events/s |
| SecureBERT | T4 GPU | 19.6 / 22.1 ms | 146.7 events/s |

The repository benchmark code includes tokenization in single-event latency and excludes it from batched throughput timing. Transformer throughput uses batch size 32, whereas single-event latency uses batch size 1. Thus ~20 ms per single event and ~145 events/s in batches are compatible measurements, not reciprocals from the same experiment.

These GPU and CPU figures are measurements on different hardware, not a controlled statement that one architecture is universally faster. They are also not the observed latency on this laptop.

RoBERTa improves over the supplied CRF by 9.44 F1 points (61.9557 minus 52.5140). Versus the additional CRF, the gain is 10.07 points. State which CRF baseline is used.

These latency artifacts cover the original five models; do not assign the same measured timings to BiLSTM or either DeBERTa run. A source document may contain multiple events, and lexicon scanning plus all event queries contribute to total demo response time.

## Recommended presentation sequence

1. Scope and data: gold-trigger argument extraction, split, role inventory, limitations.
2. Model families: include the additional experiments if the final consolidated results are shown.
3. Evaluation: typeset equations, exact event attachment, document bootstrap.
4. Validation selection and test results: error bars and clear supplied/additional provenance.
5. Focused comparisons: cased/uncased, RoBERTa/SecureBERT, role-conditioned DeBERTa; verified statistical claims only.
6. Academic context: TextEE with correct metric columns and explicit comparability limits.
7. Efficiency and error analysis: measured hardware/batch settings, distant/rare roles, and span boundaries.
8. Demo and conclusion: distinguish automatic trigger proposals from gold-trigger benchmark scores.

Keep historical published results separate from local CPU verification (RoBERTa 0.618787 and DeBERTa 0.592784). The returned T4 DeBERTa run now reproduces historical F1 0.5932521087; five events change versus CPU and yield one net additional correct argument. See [GPU verification](deberta_gpu_verification.md). The separate effects of backend, precision and batching remain unisolated. No retraining is needed merely to prepare these slides.
