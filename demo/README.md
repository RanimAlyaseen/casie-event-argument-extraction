# CASIE demo: source-highlighted event arguments

This is the team's original Gradio demonstration with a portable launcher.
Paste an English cybersecurity report and inspect proposed event triggers and
argument spans highlighted in the source. The table lists extracted role spans.
The interface highlights source spans and lists extracted arguments. It does
not perform entity linking, coreference resolution, human correction or export
of reviewed event records.

## Run

1. Obtain a complete fine-tuned **RoBERTa checkpoint** from the team and place
   it under ignored `runs/`. RoBERTa and standard DeBERTa were restored locally;
   neither is included in this public repository. A base pretrained model is
   not a substitute. The directory must include trained weights, `config.json`,
   `tokenizer_config.json`, and tokenizer vocabulary files.
2. Install the repository's dependencies in a suitable Python environment:
   `pip install -r requirements.txt`.
3. From the repository root, run:

   ```bash
   python demo/launch.py --checkpoint runs/roberta_base_muhammet --model-label RoBERTa
   ```

   To share a temporary Gradio URL during a presentation, add `--share`.
   The local URL is the default. On Colab, upload or otherwise provide the
   repository and checkpoint in the runtime, then run the same command with
   the actual checkpoint directory. No personal Drive path is required.

`--model-label` controls the displayed model name. If omitted or blank, the
checkpoint folder name is used, preventing a DeBERTa run from being displayed
as RoBERTa. `--lexicon PATH` selects a separate lexicon; otherwise the launcher
uses the original `demo/trigger_lexicon.json`.

```bash
# Use standard DeBERTa instead of RoBERTa:
python demo/launch.py --checkpoint runs/deberta_v3_base_run/outputs/deberta_v3_base --model-label DeBERTa-v3-base

# Optional separately obtained lexicon:
python demo/launch.py --checkpoint runs/roberta_base_muhammet --model-label RoBERTa --lexicon runs/roberta_base_muhammet/trigger_lexicon.json
```

These example directories are local restoration locations, not files supplied
by a clone. Run one instance at a time. The original tracked lexicon has 2,061
phrases; a separately received 2,062-phrase variant was tested locally and adds
`were hacked`. It has not replaced the tracked lexicon.

## Model coverage

The application loads **one standard transformer checkpoint per launch**.
The five event-type choices are not a model selector. RoBERTa is the primary
demo model because it was selected using validation F1. Standard DeBERTa has
also passed loading and sample-inference checks through this path.

CRF and BiLSTM-CRF require different model loaders. Role-conditioned DeBERTa
requires `team_code/role_prime_experiment.py`, its role inventory and multiple
role queries; its checkpoint is rejected by this launcher. The demo does not
provide a full multi-model comparison interface. Use the benchmark tables
and executed notebook to present the complete project comparison.

The launcher uses `team_code/casie_demo.py`, the shared transformer and scorer,
and `demo/trigger_lexicon.json`. The lexicon came from the training split.
Do not upload a private checkpoint to a public repository without team approval.

## Reading the output

The automatic mode finds candidate trigger phrases with a lexicon. The analyst
can instead enter a trigger phrase and choose one of the five event subtypes.
The same phrase is selected at **every** matching occurrence, so repeated
phrases may produce multiple event suggestions. Highlights show extracted spans;
they do not link aliases or resolve entities across sentences.

The selected RoBERTa model scored **0.6196 exact argument F1 on test with
gold event triggers and subtypes**. The lexicon alone scored about **0.501
trigger F1** on the test documents. Full raw-text trigger-to-argument F1 has
not been measured. The demo may miss events and cannot inherit the 0.6196
score as an end-to-end claim. Its current UI also has no human correction or
export step.

The restored RoBERTa has now been compared against the supplied predictions:
734 of 736 event prediction sets match exactly on CPU, with F1 0.618787 versus
the historical 0.619557. Standard DeBERTa CPU F1 is 0.592784 versus historical
0.593252. Both full CPU runs have no scorer warnings. Backend, precision and
batching differences are plausible causes, but have not been isolated. See
the [CPU verification report](../reports/verification/checkpoint_cpu_verification.md).
The historical 0.6196 is the benchmark result, not an exact reproduction or
an end-to-end score for the local demo.
