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

## Interpretation and limits

- The standard DeBERTa row is the **checkpoint-backed second run**. A separate
  earlier validation result of 0.6171 lost its checkpoint and has no test
  evaluation; it is excluded from the table.
- Five initial model weight sets were not supplied. Their prediction files are
  present, but this repository cannot restore those fitted models. Additional
  checkpoint ZIPs were saved separately and are not committed here.
- The original team's test predictions were viewed before the extra runs.
  The final test set is therefore not a fully blind benchmark; additional
  models were selected using validation F1.
- The notebook's optional per-event export did not survive the Colab reset.
  Rebuilding it needs checkpoint restoration and inference.
- The raw preprocessing appendix is retained for provenance but was not
  independently rerun end to end in the additional training session.

For published context, [TextEE (Huang et al., 2024)](https://aclanthology.org/2024.findings-acl.760/)
evaluates CASIE with a different reported document split (701 / 149 / 149).
Its scores are context, not a directly comparable leaderboard for this
corrected 799 / 100 / 100 split. The role-conditioned experiment is inspired
by [TAGPRIME](https://aclanthology.org/2023.acl-long.723/); it is not claimed
as a reproduction.
