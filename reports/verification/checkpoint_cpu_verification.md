# Restored-checkpoint CPU verification

Both the separately obtained RoBERTa checkpoint and the additional standard
DeBERTa checkpoint load and run with `team_code/casie_transformer.py`.
The Gradio sample-extraction callback also succeeds for both.

## Full test evaluation

The runs use all 736 events from 100 test documents and 2,109 gold arguments
in the input bundle. The shared `casie_eval.py` requires exact event ID,
inclusive token span and role. Gold triggers and subtypes are supplied.

| Model / evidence | Precision | Recall | F1 | Correct | Predicted | Gold |
|---|---:|---:|---:|---:|---:|---:|
| RoBERTa supplied artifact | 0.6217765043 | 0.6173541963 | 0.6195574590 | 1,302 | 2,094 | 2,109 |
| Restored RoBERTa, CPU | 0.6207061069 | 0.6168800379 | 0.6187871581 | 1,301 | 2,096 | 2,109 |
| DeBERTa historical notebook | 0.5863825845 | 0.6002844950 | 0.5932521087 | 1,266* | 2,159* | 2,109 |
| Restored DeBERTa, CPU | 0.5859194071 | 0.5998103367 | 0.5927835052 | 1,265 | 2,159 | 2,109 |

\* DeBERTa historical correct/predicted counts are inferred from the recorded
precision, recall and F1; the original additional-run GPU test prediction
file is unavailable. RoBERTa's supplied predictions are present in the bundle
and were rescored directly. Both CPU runs have **no scorer warnings**.

CPU inference used float32, batch size 1 and two CPU threads. RoBERTa took
605.13 seconds and DeBERTa took 804.91 seconds on the local verification
machine. These are full-inference elapsed times, not the original T4 latency
benchmark or measurements transferable to another computer.

## Known differences

RoBERTa matches the supplied prediction sets on **734 of 736 events**:

- `annotation/2095.json::0::5::E6`: CPU adds a Purpose prediction at token 134;
  the event's two correct arguments are unchanged.
- `annotation/2288.json::1::2::E5`: supplied Attacker tokens 161–165 becomes
  Attacker 161–164 and Trusted-Entity 165; correct arguments fall from three to two.

For DeBERTa, the CPU run has one fewer correct argument in aggregate and the
same total predicted/gold counts. Without the original GPU predictions, that
does not identify a specific changed event or span.

Source code, input checksums and restored checkpoint files were checked.
Historical inference used CUDA float16 autocast and larger batches, whereas
the new runs use CPU float32 and batch 1. Those differences are plausible
explanations, not an established cause. **The historical benchmark tables,
figures and executed outputs have not been replaced.**

## Artifacts and next check

The [machine-readable summary](../../results/verification/checkpoint_cpu_verification.json)
records versions, hashes, counts and relative local run paths. CPU predictions
and checkpoint weights remain in ignored `runs/`; they are not distributed
in this repository.

The [DeBERTa Colab notebook](../../notebooks/CASIE_Colab_DeBERTa_Verification.ipynb)
has now produced a returned T4 result. Independent rescoring reproduces
historical F1 **0.5932521087**; five event prediction sets differ from CPU,
giving one additional correct argument overall. See the
[GPU verification report](deberta_gpu_verification.md). The notebook and
[instructions](colab_deberta_verification.md) remain available for repetition.
It is DeBERTa-specific and must not be used unchanged for RoBERTa.

The original tracked demo lexicon has 2,061 phrases. A separately received
2,062-phrase variant was used for the local live demonstration. Both gave
the same trigger evaluation on the test documents. This is separate from
gold-trigger argument evaluation and does not establish the demo's complete
raw-text trigger-to-argument F1.
