# Team contributions and artifact provenance

Team Ultron prepared the CASIE Project 2 submission collaboratively. The
responsibilities below record each member's main work while acknowledging that
model training, evaluation, and final integration involved team support.

## Individual contributions

### Saleh Almasaadi and Tasnim Alibraheem — dataset preparation

- Inspected and cleaned the CASIE dataset.
- Checked for duplicate documents and records.
- Investigated annotation and character-offset issues.
- Documented the decisions made during data cleaning.
- Converted the source annotations into word-level labels while preserving
  their connection to each event mention.
- Prepared the document-level training, validation, and test splits.

### Ranim Alyasein and Muhammet Göksenin — modeling and presentation

- Researched suitable baseline and recent sequence-labeling models.
- Trained the baseline and newer model architectures.
- Recorded experimental settings and documented the modeling methods.
- Analyzed prediction errors and role-level performance.
- Compared traditional, neural, and transformer-based approaches.
- Prepared the final presentation slides.

### Azza Almousli — documentation and evaluation

- Managed the main project documentation, including the README.
- Evaluated model predictions using exact span-and-role metrics.

## Artifact provenance

| Workstream | Submitted evidence |
|---|---|
| Dataset preparation | Corrected splits, metadata, checksums, manifest, and constraints in the input bundle; raw preprocessing source retained in notebook Appendix A |
| Initial model and evaluation work | Supplied CRF, BERT cased/uncased, RoBERTa, and SecureBERT prediction files; shared scoring and model implementation |
| Additional experiments | Feature CRF, BiLSTM-CRF, DeBERTa-v3-base, and role-conditioned DeBERTa training and evaluation outputs in the executed notebook |
| Consolidation and presentation | Shared validation/test comparison, figures, error analysis, benchmark notes, addendum, README, and slides |

The five supplied model checkpoints were not provided with the prediction
files. Additional checkpoints were saved separately from this repository. The
provenance labels in `results/*.csv` and `results/experiment_manifest.json`
distinguish supplied predictions from additional runs. The two CRF rows are
separate runs of the same model family.
