# DeBERTa GPU verification: historical F1 reproduced

The returned Colab T4 run achieves exact event-specific span-and-role F1
**0.5932521087160262**, matching the historical notebook's standard DeBERTa
test result. Both returned prediction files were independently rescored
locally using `team_code/casie_eval.py`; no model inference or training was
performed during this archive review.

## Scores and population

All 736 test events from 100 documents are present, with 2,109 gold arguments
and **no scorer warnings** for either file.

| Configuration | Precision | Recall | F1 | Correct | Predicted | Gold |
|---|---:|---:|---:|---:|---:|---:|
| Saved CPU float32, batch 1 | 0.5859194071 | 0.5998103367 | 0.5927835052 | 1,265 | 2,159 | 2,109 |
| Returned T4 float16, batch 32 | 0.5863825845 | 0.6002844950 | 0.5932521087 | 1,266 | 2,159 | 2,109 |

The recorded GPU run used Python 3.13.16, PyTorch 2.11.0+cu130,
Transformers 5.21.0, Tokenizers 0.23.3, SentencePiece 0.2.2 and CUDA 13.0.
The repository inference path uses maximum input length 512 and CUDA float16
autocast. Recorded `predict_file()` elapsed time was 15.00 seconds for the
full test population; this timing includes that entry point's encoding and
file-writing work and is not a pure model-kernel latency measurement.

## Event differences

**731 event prediction sets are identical; five change.** The changes give
one net additional correct argument, while predicted and gold totals stay
the same.

| Event UID | CPU correct | GPU correct | Change |
|---|---:|---:|---|
| `annotation/10218.json::0::0::E1` | 3 | 4 | Discoverer tokens 0–6 split into 0–1 and 3–6; 3–6 matches gold |
| `annotation/10329.json::1::0::E1` | 2 | 2 | Spurious Vulnerable_System_Version at token 32 removed |
| `annotation/1611.json::0::3::E5` | 2 | 2 | Non-gold Vulnerability span 81–83 split into 81 and 83 |
| `annotation/2093.json::2::2::E7` | 1 | 1 | Non-gold Attack-Pattern boundary changes from 266–268 to 267–268 |
| `annotation/552.json::0::1::E2` | 1 | 1 | Non-gold owner/discoverer predictions merge into Discoverer 13–16 |

The [machine-readable differences](../../results/verification/deberta_cpu_gpu_event_differences.json)
retain event IDs, inclusive token spans, roles and gold-match flags, without
copying article excerpts. The original full differences and raw prediction
files remain under ignored `runs/`.

## Provenance and independent checks

Source archive: `results_20261101T140646Z_CASIE_verification.zip`,
36,174 bytes. SHA-256:
`3f3c0bad943213accbeeb1c5bb881427784e367d99bab13ff76b0691800cd8d7`.

Its five members are GPU predictions, reserialized CPU predictions,
`comparison_report.json`, `event_differences.json` and `event_differences.csv`.
It contains no weights or dataset bundle. CRC, exact member inventory and
unique event populations were checked before accepting the result.

- Test data hash matches the repository input manifest.
- Recorded source hashes match the shared scorer and transformer code.
- Recorded checkpoint hash matches the locally restored weights.
- Every returned metric matches independent rescoring.
- Independently computed changed events match the returned JSON and CSV.
- The returned CPU prediction **sets** match the original CPU reference exactly.
  Its file bytes differ because the notebook reserializes predictions through
  `casie_eval.save_predictions`; this does not change the event/span/role sets.

The [verification JSON](../../results/verification/deberta_gpu_verification.json)
contains hashes, runtime information, counts and check results. Weights, raw
predictions and the duplicate return ZIP are not committed to Git.

## What this establishes

The historical aggregate score is reproduced under the recorded T4,
float16, batch-32 configuration. The CPU/GPU comparison now identifies the
five changed events and the span responsible for the extra correct argument.

Hardware/backend, precision and batch size differ together; this comparison
does **not isolate their individual causal effects**. The original historical
GPU per-event file remains unavailable, so matching aggregate F1 does not
prove that the new GPU predictions are identical to that missing file.
Historical benchmark tables, figures and executed notebook outputs remain
unchanged. The DeBERTa GPU verification task is complete at this stated scope.
