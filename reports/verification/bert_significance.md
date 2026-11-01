# CASIE: BERT cased versus uncased significance audit

Verified on 1 November 2026. This is a retrospective analysis of the supplied test prediction artifacts. The audit performed no model loading or training and did not change historical benchmark scores. These notes integrate that audit into the repository.

## Finding

**The observed improvement is supported by both analyses.** BERT uncased improves exact event-specific argument span-and-role micro F1 by **3.4274 F1 points** over BERT cased. A paired document bootstrap 95% confidence interval for the improvement is **[+1.2494, +5.5628] points**. A separate two-sided document randomization test gives **p = 0.003599964** (100,000 randomizations, seed 42; unadjusted).

The archived **p = 0.002** can be reproduced numerically using 5,000 paired document bootstrap samples, seed 42, and twice the smaller sign-tail fraction. This establishes a reproducible calculation consistent with the archived claim, but the original generating script and p-value definition were not identified. It does not establish that this was the original calculation. Keep the bootstrap-tail calculation distinct from the randomization p-value.

## Scores and counts

All values below were recomputed with the repository's `team_code/casie_eval.py`, using all **736 events in 100 documents**, with **2,109 gold arguments**. Both prediction files contain exactly one row per expected event. There are no missing or unknown event IDs and **no scorer warnings**.

| Supplied prediction artifact | Precision | Recall | F1 | Correct | Predicted | Gold |
|---|---:|---:|---:|---:|---:|---:|
| BERT cased | 0.5289217955 | 0.5419630156 | 0.5353629977 | 1,143 | 2,161 | 2,109 |
| BERT uncased | 0.5396690709 | 0.6031294452 | 0.5696372593 | 1,272 | 2,357 | 2,109 |

Uncased finds **129 additional correct arguments** and predicts **196 additional arguments**. Recall improves by **6.1166 points**; precision improves by **1.0747 points**. The bootstrap interval for the precision difference includes zero, so a precision gain is not clearly established by that interval.

The original per-model 1,000-document-bootstrap intervals are also reproduced:

- Cased F1: [0.5100127823, 0.5609022556].
- Uncased F1: [0.5459839845, 0.5923612779].

The paired difference test is the relevant comparison; overlap between the two individual intervals is not itself a significance test.

## Paired bootstrap confidence interval

For each of 100,000 replicates, sample 100 document IDs with replacement and use the **same sampled IDs for both models**. All events of a selected document stay together. Recompute pooled corpus precision, recall, and micro F1 from argument counts; do not average document F1 scores. Compute each difference as uncased minus cased.

Seed: 42, using Python `random.Random`. Documents are sorted by document ID. The 95% percentile interval uses linear interpolation at the 2.5th and 97.5th percentiles.

| Difference | Observed, F1/percentage points | Paired bootstrap 95% interval, points |
|---|---:|---:|
| Precision | +1.0747 | [-1.6386, +3.8203] |
| Recall | +6.1166 | [+3.6995, +8.5687] |
| F1 | +3.4274 | [+1.2494, +5.5628] |

These are intervals conditional on the two fitted models, rather than uncertainty across retraining seeds.

## Two-sided randomization test

The null model treats the two systems' labels as exchangeable within each whole document. For each of 100,000 randomizations, independently swap the two systems' complete argument-count triples within each document with probability 0.5. Recompute the pooled corpus F1 difference. Using count triples is sufficient for this exact span-and-role micro F1 because the event-level matches have already been scored.

Count a randomization as extreme when its absolute F1 difference is at least the observed absolute difference. The implementation compares exact integer ratios, avoiding floating-point ambiguity at the threshold.

There were **359 extreme randomizations**. With the Monte Carlo correction:

```text
p = (359 + 1) / (100000 + 1)
  = 0.0035999640003599965
```

Seed: 42, with an independent random generator from the bootstrap. Approximate Monte Carlo standard error: 0.000189. This is one focused retrospective comparison; the p-value is unadjusted and should not be presented as a benchmark-wide multiple-comparison result. It rejects the exchangeability null at alpha 0.05.

The sampling unit is the document, preserving dependence between events from one article. Both analyses still assume that document clusters provide suitable independent sampling units.

## Reproducing the archived 0.002

Use the first 5,000 bootstrap replicates from the fixed seed-42 procedure. These reproduce the repository's sorted-document `compare_models()` win fraction exactly.

```text
Uncased F1 greater than cased: 4,995 / 5,000
Uncased F1 less than cased:        5 / 5,000
Ties:                             0 / 5,000

Twice smaller sign-tail fraction = 2 * (5 / 5000) = 0.002
Paired F1 difference 95% percentile CI = [0.0125825670, 0.0562368641]
Rounded to the archived precision = [+0.013, +0.056]
```

The repository's empirical-order-statistic interval convention gives [0.0125844867, 0.0562368254], which rounds to the same archived interval. With 100,000 bootstrap replicates, the corresponding sign-tail fraction is 0.00206. These are descriptive bootstrap-tail estimates; the primary hypothesis-test result in this audit is the separately defined randomization p-value above. Do not describe a bootstrap win fraction as a posterior probability that a model is superior.

## Wording to use on the presentation slide

> On 736 test events from 100 documents, BERT uncased achieved 56.96 F1 versus 53.54 for BERT cased: +3.43 F1 points (paired document bootstrap 95% CI: +1.25 to +5.56; two-sided document randomization p = 0.0036). These are single-seed fitted models, and the comparison does not isolate casing from other pretrained-model and vocabulary differences.

Suggested methods footnote: **100,000 paired document bootstrap samples; 100,000 document randomizations; seed 42; exact event-specific span-and-role micro F1; unadjusted retrospective comparison.**

If retaining the historical p = 0.002 instead, explicitly identify it as the **5,000-sample paired-bootstrap sign-tail estimate**, with the matching interval [+1.26, +5.62] F1 points. Do not pair that p-value with a claim that it came from the randomization test.

## Provenance and reproducibility

- Repository commit: `e563f31d273e7f03c6b566d211c6f67d340f095e`.
- Gold input: `dataset/CASIE_test.jsonl` inside `data/TeamUltron_Project2_Inputs.zip`.
- Predictions: `predictions/test_bert-base-cased.jsonl` and `predictions/test_bert-base-uncased.jsonl` inside that bundle.
- All three member SHA-256 hashes match the input manifest. Both prediction files also match the fuller `TeamUltron_Project2 (1).zip` byte for byte.
- Archived training configurations record seed 42, with best validation epoch 5 for cased and epoch 8 for uncased. The pretrained model and vocabulary differ; this is not a controlled casing-only intervention.
- Scorer, source, bundle, prediction and audit-script hashes are recorded in [bert_significance.json](../../results/verification/bert_significance.json).
- [bert_document_counts.json](../../results/verification/bert_document_counts.json) preserves the per-document count triples used in resampling without copying document text or weights.
- Statistical code checks include exhaustive comparison of all eight swap assignments in a toy problem, identical-model checks, interpolation checks, pooled count agreement with the scorer, and the repository's 5,000-sample bootstrap win fraction.

Run the accompanying script with the repository and a new output directory:

```bash
python scripts/verify_bert_significance.py --repo . --out runs/bert_significance_rerun
# Optional provenance check against the separately obtained original team archive:
# append --team-archive /path/to/TeamUltron_Project2_original.zip
```

The script uses the Python standard library and the repository scorer. `--team-archive` is optional. It refuses to overwrite an existing output directory. It performs no inference, training, commit, or push. Audit working files are under the repository's ignored `runs/` directory.

Methods references: [Dror et al. (2018)](https://aclanthology.org/P18-1128/) discuss significance-test selection for NLP and dependencies between observations. [Riezler and Maxwell (2005), Figure 1](https://aclanthology.org/W05-0908.pdf) describe the two-sided approximate-randomization calculation with the `(c + 1)/(R + 1)` correction. This audit adapts the permutation unit to whole CASIE documents.
