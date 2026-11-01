# Change log

## 2026-11-01 — Returned GPU verification and artifact audit

- Independently rescored the returned T4 DeBERTa predictions on all 736 test
  events: historical F1 **0.5932521087 reproduced**, with no scorer warnings.
- Verified five changed event prediction sets versus CPU, with one net
  additional correct argument. Added report, runtime/hash provenance and
  span/role differences without publishing raw predictions or article excerpts.
- Updated the README, manifest, status and verification notes so the completed
  DeBERTa check is no longer described as pending. Historical scores remain intact.
- Added prediction-format guidance, the source-archive inventory and the verified
  professor-feedback checklist. Existing presentation revision remains pending.
- Audited repository inputs, notebooks, code, figures, reports and presentation
  package. Retained original artifacts and documented remaining review work.
- Clarified that the original preprocessing PDF describes an earlier
  annotation-aware step; the submitted corrected bundle uses text-only tokenization.
- Integrated on top of newer GitHub edits that moved named contributions to
  root `CONTRIBUTIONS.md` and removed the old presentation directory. Preserved
  those changes and repaired stale documentation links instead of restoring old files.

## 2026-11-01 — MIT license

Commit [426a295](https://github.com/RanimAlyaseen/casie-event-argument-extraction/commit/426a2957080c0e80c43ab166adcbfddca5c8da92).

- Added the MIT license for original project code and documentation.
- Linked it from the README and documented exclusions for CASIE inputs,
  supplied predictions, third-party material, dependencies and model checkpoints.

## 2026-11-01 — Verification and demo integration

Commit [938065b](https://github.com/RanimAlyaseen/casie-event-argument-extraction/commit/938065b5967d06e773d68b417b063b28c179c0f1).

- Replaced the empty README and documented the complete benchmark and demo scope.
- Added the BERT paired-document significance audit, results and reproduction script.
- Recorded full restored RoBERTa and standard DeBERTa CPU scoring separately
  from historical benchmark values; prepared the Colab comparison notebook.
- Added model-label and lexicon launcher options and a guard against incompatible
  role-conditioned checkpoints, with five lightweight launcher regression tests.
- Fixed the notebook path in the manifest and checkpoint-availability notes.
- Preserved original latency evidence with source hashes and timing qualifications.

## 2026-11-01 — Initial publication and organization

- [d04a65f](https://github.com/RanimAlyaseen/casie-event-argument-extraction/commit/d04a65fc69b3ab6c2bdba75f683cb5b065625d5f):
  published executed benchmark, inputs, source, tables and figures.
- [13bd8e6](https://github.com/RanimAlyaseen/casie-event-argument-extraction/commit/13bd8e6ec2e940a7af0d4fa4f98b237024fb4033):
  organized the notebook under `notebooks/` and added documentation, reports
  and the existing presentation while retaining root `team_code/`.
- [e563f31](https://github.com/RanimAlyaseen/casie-event-argument-extraction/commit/e563f31d273e7f03c6b566d211c6f67d340f095e):
  integrated the six-file portable demo package.
