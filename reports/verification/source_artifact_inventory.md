# Source artifact inventory

This inventory records why the current submission uses the fuller team
package and keeps older drafts and local weights separate. Archive contents
were inspected before integration; no archive was extracted over the repository.

## Original five-model packages

| Item | `TeamUltron_Project2.zip` | `TeamUltron_Project2 (1).zip` |
|---|---:|---:|
| Compressed bytes | 852,284 | 2,500,429 |
| Files, excluding directory entries | 67 | 90 |
| Prediction files | 7 | 9 |
| Includes BERT uncased predictions | No | Yes |
| Weights | None | None |
| Dataset split files | None | None |
| RoBERTa saved test F1 | 0.6131889764 | 0.6195574590 |
| Prediction bytes match current input bundle | All seven differ | All nine match |

SHA-256 of the older archive:
`e2135bcfdc5543f6d6a0ac4a05f037bca81eaa55448210a8c53f057d30f51cc5`.

SHA-256 of the fuller archive:
`42cf8ab1a031435ab62527b32738d51d1384b83a3903914741fddd3d19909a5f`.
The latter also matches the audited team-prediction source hash recorded in
the input bundle's manifest.

The retained preprocessing PDF describes an earlier annotation-aware
token-splitting stage. It is historical provenance, rather than the description
of the final corrected text-only input representation; see
[current dataset preparation](../../DATASET_PREPARATION.md).

The fuller archive's `CASIE_Project_Report_EN.pdf` is byte-identical to the
existing `reports/CASIE_Initial_Five_Model_Report.pdf`; adding another copy
would duplicate the original report. Its separate team summary describes
the earlier five-model stage and contains literature and statistical wording
qualified by the current verification notes. It has not replaced the final
consolidated benchmark or the latest root contribution statement.

The archive's tiny training/evaluation template notebooks are earlier workflow
examples, not replacements for the executed consolidated submission. The
current shared scorer retains the corrected Arg-I denominators; the current
transformer training loader retains its float32 integration fix. No old source
module was copied over those versions during this audit.

## Checkpoints and lexicons

- Standard DeBERTa was restored separately from `deberta_v3_base_run.zip`.
- RoBERTa's weights were separately obtained and restored locally. Its training
  configuration matches the fuller package's best epoch 5 / validation F1
  0.6189304095 record.
- Neither five-model archive contains trained weights. CRF/BiLSTM/DeBERTa
  checkpoint ZIPs are separate artifacts and remain outside Git.
- The tracked 2,061-phrase lexicon matches the fuller archive. A separate
  2,062-phrase variant adds `were hacked` and is selectable through `--lexicon`.
  It does not replace the tracked version.

## Recent verification artifacts integrated

- BERT audit, per-document counts, numerical report and reproduction script.
- RoBERTa/DeBERTa CPU score summaries and RoBERTa event differences.
- Returned T4 DeBERTa verification summary and independently checked span/role
  differences, with article excerpts omitted from the public difference file.
- Original latency JSON and source hashes.
- Portable Colab verification source, prediction-format guide, professor-feedback
  checklist and current change/status documentation.

Raw CPU/GPU predictions, original return ZIP, local runtime logs, environments,
weights and temporary uploads remain under ignored local storage. Older duplicate
submission ZIPs and private chat-handoff notes are not part of the published
project. The cancelled startup prompt is unrelated and is not integrated.

See the [change log](../../CHANGELOG.md), [GPU verification](deberta_gpu_verification.md)
and [report index](../README.md) for the current evidence and remaining work.
