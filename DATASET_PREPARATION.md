# Dataset preparation

## Source and task representation

The source is the [CASIE research corpus](https://github.com/Ebiquity/CASIE)
of annotated cybersecurity news. Please cite Satyapanich, Ferraro, and
Finin, *CASIE: Extracting Cybersecurity Event Information from Text*, AAAI
2020, DOI: [10.1609/aaai.v34i05.6401](https://doi.org/10.1609/aaai.v34i05.6401).

The submitted input bundle is
[`data/TeamUltron_Project2_Inputs.zip`](data/TeamUltron_Project2_Inputs.zip).
It contains corrected model-ready splits, metadata, checksums, split
constraints, and the supplied model prediction JSONL files. Each split line
represents one event mention with its document tokens, gold trigger, event
subtype, arguments, and event-conditioned labels. The notebook checks the
expected files and SHA-256 hashes before scoring.

## Cleaning and alignment

The dataset subgroup repaired annotation offsets and excluded one exact
duplicate document, leaving 999 documents. Tokenization is based only on
document text with the Unicode-aware expression `\w+|[^\w\s]`; annotation
boundaries do not determine token boundaries. Argument spans are aligned to
overlapping tokens and converted to event-conditioned BIO labels. This avoids
using gold span boundaries to create tokens at inference time.

The dataset metadata records 31,151 exact token-boundary alignments and 25
partial-token annotations. Partial boundaries cannot always be recovered
exactly from token labels. Some event-token positions have simultaneous roles;
the per-role layers retain them, while single-label sequence baselines make a
choice. These limitations matter to exact-span evaluation.

## Document-level split

| Split | Documents | Events | Arguments |
|---|---:|---:|---:|
| Train | 799 | 6,877 | 18,445 |
| Validation | 100 | 869 | 2,140 |
| Test | 100 | 736 | 2,109 |

The split uses seed 42 and keeps every event of an article in one partition.
The known near-duplicate documents `annotation/2663.json` and
`annotation/891.json` stay together. The included manifest and constraints
record the split assignment. The notebook checks for document overlap and
validates the known pair.

The bundle contains nine supplied prediction JSONL files: validation and test
for the four transformer baselines, plus test predictions for the supplied
CRF. The separate additional experiments are executed in the notebook.

## Reproduction boundary

The corrected split is verified as an **input** to the executed experiments.
Appendix A of the notebook retains the raw preprocessing source for context;
that appendix was not rerun end to end during the additional training session.
For the exact file inventory, hashes, and alignment caveats, inspect
`CASIE_README.md` and `CASIE_dataset_metadata.json` inside the input bundle.
