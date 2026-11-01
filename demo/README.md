# CASIE demo: source-highlighted event arguments

This is the team's original Gradio demonstration with a portable launcher.
Paste an English cybersecurity report and inspect proposed event triggers and
argument spans highlighted in the source. The table lists extracted role spans.
The interface **does not yet edit or export approved event cards**; those are
proposed startup features.

## Run

1. Get the fine-tuned **RoBERTa checkpoint** used for the team's saved
   predictions. It is not in this public repository. A base `roberta-base`
   download is not a substitute. The checkpoint directory must include the
   trained weights, `config.json`, and tokenizer files.
2. Install the repository's dependencies in a suitable Python environment:
   `pip install -r requirements.txt`.
3. From the repository root, run:

   ```bash
   python demo/launch.py --checkpoint /path/to/fine-tuned-roberta
   ```

   To share a temporary Gradio URL during a presentation, add `--share`.
   The local URL is the default. On Colab, upload or otherwise provide the
   repository and checkpoint in the runtime, then run the same command with
   the actual checkpoint directory. No personal Drive path is required.

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

Before presenting the live demo, compare checkpoint inference on the test
split with the saved `predictions/test_roberta-base.jsonl` inside the input
bundle. If outputs differ, investigate tokenizer, code version and model
selection before attributing the published score to that checkpoint.
