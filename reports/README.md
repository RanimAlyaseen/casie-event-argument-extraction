# Reports

## Current verification

- [BERT significance audit](verification/bert_significance.md): paired document
  bootstrap intervals, a two-sided randomization test, and the reproducible
  calculation consistent with the archived bootstrap-tail estimate of 0.002.
  [Machine-readable result](../results/verification/bert_significance.json),
  [document counts](../results/verification/bert_document_counts.json), and
  [reproduction script](../scripts/verify_bert_significance.py) accompany it.
- [Restored-checkpoint CPU verification](verification/checkpoint_cpu_verification.md):
  RoBERTa and DeBERTa full-test scores, counts, warnings, and discrepancies.
  Historical benchmark scores remain unchanged.
- [DeBERTa Colab verification instructions](verification/colab_deberta_verification.md):
  required files, GPU settings and return artifacts for repeating the comparison.
- [Returned DeBERTa GPU verification](verification/deberta_gpu_verification.md):
  independent rescoring reproduces historical F1 0.5932521087; five events
  change relative to CPU, giving one additional correct argument overall.
- [Latency evidence](verification/latency.md): original measurements, timing
  boundaries, reported hardware and limits on comparison.
- [Source artifact inventory](verification/source_artifact_inventory.md):
  archive provenance, duplicates and why the fuller team package is the relevant source.
- [Presentation feedback checklist](verification/presentation_feedback_checklist.md):
  verified formulas, model groups, statistical wording, literature context and latency
  qualifications. The PPTX revision itself remains unfinished.

See [project status](../docs/PROJECT_STATUS.md) for the remaining presentation
and attribution work, and the [change log](../CHANGELOG.md) for recent updates.

## Original submission reports

- [Final results addendum](CASIE_Project2_Final_Results_Addendum.pdf): current
  nine-row comparison, evaluation design and limitations.
- [Dataset preparation and validation](CASIE_Dataset_Preprocessing_Validation.pdf):
  historical team preprocessing report, including an earlier annotation-aware
  token-splitting step. The final corrected input uses text-only tokenization;
  see [current dataset preparation](../DATASET_PREPARATION.md). The consolidated
  notebook checks that corrected bundle independently; its raw appendix was not
  rerun end to end during the additional experiments.
- [Initial five-model report](CASIE_Initial_Five_Model_Report.pdf): records the
  team's original baselines and analysis. Its literature comparison and model
  lineup predate the consolidated benchmark. Use the final addendum and the
  executed notebook for the submitted test table.

The PDFs are retained as original artifacts. Where older narrative describes
the casing comparison as causal, or treats literature metrics as directly
comparable, use the qualified interpretation in [BENCHMARK.md](../BENCHMARK.md)
and the newer verification notes. The earlier presentation directory was
removed by a subsequent GitHub update; no revised deck has been published here yet.
