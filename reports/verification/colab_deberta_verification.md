# Run the DeBERTa GPU verification on Colab

This is an inference-only check of the restored checkpoint against the saved
CPU predictions. It does not train, commit, upload weights to GitHub, or change
the historical result.

## Files to obtain from the team

1. [CASIE_Colab_DeBERTa_Verification.ipynb](../../notebooks/CASIE_Colab_DeBERTa_Verification.ipynb).
2. `deberta_v3_base_run.zip` (743,581,511 bytes, approximately 744 MB), containing
   the fine-tuned standard DeBERTa checkpoint. Keep it zipped.
3. `CASIE_CPU_test_predictions.jsonl` (154,112 bytes), the recorded CPU reference.
   Its SHA-256 must be
   `1d170c66909d1704fc0e8a3eca500beaebbcd4160bb6b24f578aca3b18fca3db`.

The last two are separately supplied local artifacts, not files in a fresh
clone. The original CPU file is retained locally at
`runs/deberta_test_cpu_20261101T105313Z/test_predictions.jsonl`; a copy was
named `CASIE_CPU_test_predictions.jsonl` for this comparison. Keep both
private working inputs out of Git. The notebook obtains public code and
dataset inputs from the fixed repository commit recorded inside it.

## Steps

1. Open [Google Colab](https://colab.research.google.com/) and upload the notebook.
2. Choose **Runtime → Change runtime type → T4 GPU**, save and connect.
3. Run cells individually from top to bottom. The first installs the recorded
   package versions. If a session restart is requested, restart and rerun from
   the top before continuing.
4. At the upload cell, select the checkpoint ZIP and CPU prediction JSONL.
   Wait until both uploads finish. The cell verifies the CPU checksum; the
   following cells verify weights, source and test data.
5. Continue through checkpoint loading, inference on all 736 events, scoring,
   exact per-event comparison and the final download cell. Preserve **batch
   size 32, max length 512, and the repository's CUDA float16 autocast**.
   If any cell fails, return the complete error and cell number rather than
   changing settings to force a score match.
6. Download `results_..._CASIE_verification.zip` before disconnecting and return
   it to the team. Also keep an executed copy of the verification notebook.

The return archive contains `gpu_test_predictions.jsonl`,
`cpu_test_predictions.jsonl`, `comparison_report.json`,
`event_differences.json`, and `event_differences.csv`. It excludes weights
and dataset files. Colab runtime files are temporary; see the
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Interpretation

CPU F1 is **0.5927835051546392**; historical F1 is **0.5932521087160262**.
Report whatever the GPU produces. Backend, precision and batch size change
together, so this comparison cannot isolate their individual causal effects.
A score match also cannot prove equality to the original missing GPU artifact.
Until returned artifacts have been reviewed, GPU verification remains pending.
