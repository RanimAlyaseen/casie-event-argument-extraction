# Project status

Updated 1 November 2026. Historical submission artifacts and subsequent
verification are separated so a new check does not silently replace a result.

## Completed

- Executed consolidated notebook preserved under `notebooks/`, including
  recorded outputs. Shared `team_code/` remains at the repository root.
- Corrected input bundle, eight-row validation comparison, nine-row test
  comparison, final figures, original PDFs and existing presentation retained.
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

## Remaining

1. **GPU comparison:** run the [verification notebook](../notebooks/CASIE_Colab_DeBERTa_Verification.ipynb)
   on an available T4, return its result ZIP, and review changed event spans.
   It requires the separately supplied checkpoint and saved CPU reference.
   The exact cause of the historical DeBERTa discrepancy remains unconfirmed.
2. **Presentation:** revise the existing PPTX for the professor's requests:
   typeset formulas, algorithm groups, intervals, audited statistical claims,
   qualified TextEE context and hardware/batch-specific latency. Fix the
   CRF test bar and split-table layout, then inspect every rendered slide.
3. **Contributions:** the team must confirm individual name-to-workstream
   assignments before making personal contribution claims. Current provenance
   records workstreams without inventing attribution.
4. **Submission review:** check the revised slides against the notebook,
   reports and tables, and rehearse the local demo on the presentation machine.

No retraining is required for these remaining tasks. The GPU comparison can
be completed independently of the slide and documentation work. Do not infer
that GPU precision caused a difference until the saved predictions establish
what changed; even a score match does not recover the missing original GPU file.
