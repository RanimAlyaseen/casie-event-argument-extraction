# Project status

Updated 1 November 2026. Historical submission artifacts and subsequent
verification are separated so a new check does not silently replace a result.

## Completed

- Executed consolidated notebook preserved under `notebooks/`, including
  recorded outputs. Shared `team_code/` remains at the repository root.
- Corrected input bundle, eight-row validation comparison, nine-row test
  comparison, final figures and original PDFs retained. Newer GitHub edits
  moved the named contribution statement to root `CONTRIBUTIONS.md` and
  removed the old presentation directory; both decisions are preserved.
- RoBERTa selected by validation F1 0.6189; historical test F1 0.6196.
- RoBERTa and standard DeBERTa checkpoints restored locally; loading, sample
  inference and all-736-event CPU scoring succeed. Weights remain excluded
  from Git. Small discrepancies from historical metrics are documented in
  [checkpoint verification](../reports/verification/checkpoint_cpu_verification.md).
- The demo runs one model at a time. RoBERTa is the selected demonstration
  model; standard DeBERTa can be launched separately. Displayed model labels
  and external lexicon selection are explicit launcher options.
- BERT uncased-versus-cased significance audited using supplied predictions:
  +3.43 F1 points, paired-bootstrap 95% CI [+1.25, +5.56], two-sided document
  randomization p=0.0036. See the [audit](../reports/verification/bert_significance.md).
- Colab inference-and-comparison notebook prepared for the DeBERTa GPU check.
- Returned T4 DeBERTa predictions independently rescored: historical F1
  **0.5932521087 reproduced**, with 1,266 correct / 2,159 predicted / 2,109
  gold arguments, no scorer warnings, and five changed events versus CPU.
  See [GPU verification](../reports/verification/deberta_gpu_verification.md).
- Original project code/documentation licensed under MIT, with explicit
  [third-party exclusions](../THIRD_PARTY_NOTICES.md).
- [Prediction format](PREDICTION_FORMAT.md), [source artifact inventory](../reports/verification/source_artifact_inventory.md)
  and [recent change history](../CHANGELOG.md) document the current handoff.

## Remaining

1. **Presentation:** prepare the revised deck for the professor's requests:
   typeset formulas, algorithm groups, intervals, audited statistical claims,
   qualified TextEE context and hardware/batch-specific latency. Fix the
   CRF test bar and split-table layout, then inspect every rendered slide.
   Use the [verified feedback checklist](../reports/verification/presentation_feedback_checklist.md).
   The earlier repository PPTX was removed in commit `73cf8b1`; no revised
   presentation is currently published. The old version remains in Git history.
2. **Contribution review:** review the current named assignments in
   [CONTRIBUTIONS.md](../CONTRIBUTIONS.md) with the team before final submission.
   The latest individual entries were preserved from the newer GitHub commits.
3. **Submission review:** check the revised slides against the notebook,
   reports and tables, and rehearse the local demo on the presentation machine.

No retraining is required for these remaining tasks. The saved CPU/GPU
predictions now establish which five events changed, and the GPU run reproduces
the historical aggregate F1. Hardware, precision and batching changed together;
their individual effects remain unisolated. Matching the score does not recover
the missing original historical per-event GPU file. RoBERTa's CPU comparison
remains a close rather than exact reproduction of its supplied predictions.
