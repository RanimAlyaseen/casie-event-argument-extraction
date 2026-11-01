# Contributions and artifact provenance

Team Ultron prepared the CASIE Project 2 submission together. The artifacts
are grouped here by workstream so the final notebook does not imply that one
person performed every experiment.

| Workstream | Submitted evidence |
|---|---|
| Dataset preparation | Corrected splits, metadata, checksums, manifest and constraints in the input bundle; raw preprocessing source retained in notebook Appendix A |
| Initial model and evaluation work | Supplied CRF, BERT cased/uncased, RoBERTa and SecureBERT prediction files; shared scoring and model implementation |
| Additional experiments | New feature CRF, BiLSTM-CRF, DeBERTa-v3-base and role-conditioned DeBERTa training and evaluation outputs in the executed notebook |
| Consolidation and presentation | One shared validation/test comparison, figures, error analysis, benchmark notes, addendum and slides |
| Subsequent verification and integration | Restored RoBERTa and standard DeBERTa loading/inference checks; full CPU test scoring; paired BERT significance audit; portable demo options and GPU verification notebook |

The initial five-model package supplied predictions rather than weights.
RoBERTa's checkpoint was later obtained separately, and standard DeBERTa was
restored from its additional-run checkpoint ZIP. Both remain in ignored local
storage. No model weights are distributed in this repository.
The provenance labels in `results/*.csv` and
`results/experiment_manifest.json` distinguish supplied predictions from
additional runs. The two CRF rows are separate runs of the same model family.

The team should confirm individual name-to-workstream attribution before
using this document as a personal contribution statement.

Verification notes credit source artifacts and distinguish historical results
from later CPU checks. They do not reassign the original model-training work.
The BERT audit rescores supplied prediction files without training or inference.
