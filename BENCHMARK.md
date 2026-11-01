# Benchmark and evaluation protocol

## Task

Given a CASIE document, an annotated event subtype, and its gold trigger,
predict the spans and semantic roles of that event's arguments. The task has
26 roles; the standard sequence models use `O` plus beginning/inside labels
for each role (53 BIO tags). This is event-conditioned argument extraction,
not end-to-end trigger detection or ordinary named entity recognition.

## Shared data and selection rule

All rows below use the corrected 799 / 100 / 100 document train / validation /
test split. The 999 retained documents contain 8,482 event mentions. The
document split prevents the same article from appearing in two sets; a known
near-duplicate pair is assigned together. The input bundle and its checksums
are described in [DATASET_PREPARATION.md](DATASET_PREPARATION.md).

Select the model by **validation exact argument span-and-role micro F1**
(Arg-C F1), then report its test score. One prediction is correct only when
the event ID, start token, inclusive end token, and role match a gold item.
If TP, FP and FN are pooled across events, precision is TP/(TP+FP), recall
is TP/(TP+FN), and F1 is 2PR/(P+R). Token accuracy would be dominated by
`O` tokens. Macro role F1 and per-role support reveal weak rare roles.

The notebook also computes 95% intervals by resampling **documents** rather
than independent event mentions (1,000 bootstrap resamples in the final
consolidated evaluation). This retains dependence among events in one article.

## Final comparison

| Method | Validation Arg-C F1 | Test Arg-C F1 | Evidence in this repository |
|---|---:|---:|---|
| Feature CRF (additional run) | 0.5075 | 0.5188 | Executed notebook and tables; checkpoint separate |
| CRF (supplied run) | — | 0.5251 | Test prediction in input bundle |
| BERT cased | 0.5261 | 0.5354 | Supplied predictions in input bundle |
| BERT uncased | 0.5557 | 0.5696 | Supplied predictions in input bundle |
| BiLSTM-CRF | 0.5535 | 0.5415 | Executed notebook and tables; checkpoint separate |
| DeBERTa-v3-base | 0.5987 | 0.5933 | Executed notebook and tables; checkpoint separate |
| Role-conditioned DeBERTa | 0.5838 | 0.5966 | Executed notebook and tables; checkpoint separate |
| SecureBERT | 0.6183 | 0.6140 | Supplied predictions in input bundle |
| **RoBERTa** | **0.6189** | **0.6196** | Supplied predictions; selected on validation |

There are eight validation rows because the supplied CRF has no validation
prediction file. The nine test rows include **two runs of one CRF architecture**;
they are not nine distinct model families. Full precision, recall, macro F1,
and confidence intervals are in [validation_summary.csv](results/validation_summary.csv)
and [test_summary.csv](results/test_summary.csv). Figures are in `figures/`.

RoBERTa's test Arg-C F1 is 0.6196 with a document bootstrap 95% interval of
approximately 0.5968–0.6441. SecureBERT is close. The small observed gap
does not establish a general encoder advantage. The role-conditioned
adaptation increased validation recall over standard DeBERTa (0.722 versus
0.631) while decreasing precision (0.490 versus 0.569).

## Verification after the original evaluation

The historical table above is preserved. Subsequent checks are recorded
separately in [results/verification/](results/verification/) and the
[verification reports](reports/README.md).

**BERT uncased versus cased:** the supplied test artifacts reproduce F1
0.5696372593 versus 0.5353629977, an improvement of **3.4274 F1 points**.
A paired bootstrap over whole documents gives a 95% difference interval of
**[+1.2494, +5.5628] points** (100,000 resamples, seed 42). A separate
two-sided document randomization test gives **p = 0.003599964** (100,000
randomizations, seed 42; unadjusted retrospective comparison). A 5,000-sample
bootstrap sign-tail calculation reproduces the archived 0.002, but its
original generating script was not identified. See the
[complete audit](reports/verification/bert_significance.md) for both methods.
These fitted models have different pretrained weights and vocabularies;
this is not a causal intervention on casing alone or a multi-seed study.

**Checkpoint restoration:** RoBERTa and standard DeBERTa have been restored
locally and tested with the shared code. CPU float32, batch-size-1 inference
on all 736 events gives F1 **0.6187871581** and **0.5927835052**, respectively,
with no scorer warnings. RoBERTa differs from the supplied predictions on
2 events (734 identical). The DeBERTa aggregate has one fewer correct argument
than the historical metrics imply, with unchanged predicted/gold counts.
The separate causal effects of hardware, precision and batching are not
isolated. See the [CPU verification report](reports/verification/checkpoint_cpu_verification.md).
The returned [DeBERTa GPU comparison](reports/verification/deberta_gpu_verification.md)
has now been independently rescored: T4/float16/batch-32 predictions achieve
**0.5932521087 F1**, exactly matching the historical aggregate F1. Five events
differ from CPU, yielding one net additional correct argument. All 736 events
are present and neither scorer run warns. This reproduces the score under
the recorded GPU configuration; it does not prove identity to the unavailable
original historical per-event predictions. Historical tables remain intact.

**Latency evidence:** the original five-model archive includes measurements
for the supplied CRF on CPU and four transformers on a reported T4 GPU.
Single-event latency includes encoding; batched throughput uses a different
timing boundary and batch size. These are not laptop timings or a controlled
same-hardware comparison. See [latency notes](reports/verification/latency.md)
and the preserved source JSON under [results/latency/](results/latency/).

## Interpretation and limits

- The standard DeBERTa row is the **checkpoint-backed second run**. A separate
  earlier validation result of 0.6171 lost its checkpoint and has no test
  evaluation; it is excluded from the table.
- The supplied benchmark is backed by prediction artifacts. RoBERTa weights
  were subsequently obtained separately and restored locally; the other four
  supplied weight sets have not been restored here. Standard DeBERTa was also
  restored from its separate additional-run archive. No checkpoint weights
  are committed, and a fresh clone cannot load these fitted models without
  separately obtaining their checkpoints.
- The original team's test predictions were viewed before the extra runs.
  The final test set is therefore not a fully blind benchmark; additional
  models were selected using validation F1.
- The notebook's optional additional-run per-event export did not survive the
  Colab reset. The supplied predictions remain in the input bundle; the new
  CPU prediction files are retained locally under ignored `runs/`. The original
  additional DeBERTa GPU test predictions remain unavailable.
- The raw preprocessing appendix is retained for provenance but was not
  independently rerun end to end in the additional training session.

For published context, [TextEE (Huang et al., 2024)](https://aclanthology.org/2024.findings-acl.760/)
evaluates CASIE with a different reported document split (701 / 149 / 149).
Its AC and AC+ metrics also differ: AC+ requires matching the argument's
attached trigger, whereas AC does not. Our scorer requires the same event ID.
The scores are context, not a directly comparable leaderboard for this
corrected 799 / 100 / 100 split and event-specific protocol. The role-conditioned experiment is inspired
by [TAGPRIME](https://aclanthology.org/2023.acl-long.723/); it is not claimed
as a reproduction.
