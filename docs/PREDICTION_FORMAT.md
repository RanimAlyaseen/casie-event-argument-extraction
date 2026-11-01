# Prediction file format

Use a UTF-8 JSONL file with **one line per event**, keyed by `event_uid` from
the corrected CASIE split. The shared scorer is `team_code/casie_eval.py`.
This guide adapts the original team's prediction-format instructions to the
current repository layout.

## Explicit spans (recommended)

```json
{"event_uid":"annotation/10001.json::0::0::E3","arguments":[{"role":"Victim","start":12,"end":14},{"role":"Compromised-Data","start":44,"end":44}]}
```

`start` and `end` are **document-level dataset token indices**, with an
**inclusive end**. A model's local window or subword positions must be mapped
back to this coordinate system. Role names must match the dataset exactly.
An event with no arguments still needs a row with `"arguments": []`.

The scorer also accepts `token_ids` instead of `start` and `end` for an
argument. It scores the span from the minimum through maximum token ID.

```python
import sys
sys.path.insert(0, "team_code")
import casie_eval as ce

predictions = {"event-1": {(12, 14, "Victim")}, "event-2": set()}
ce.save_predictions("runs/example_predictions.jsonl", predictions)
```

## BIO labels

Whole-document labels are also accepted:

```json
{"event_uid":"event-1","labels":["O","B-Victim","I-Victim","O"]}
```

For a window, include its starting dataset-token position:

```json
{"event_uid":"event-1","labels":["O","B-Victim","I-Victim"],"window_start":230}
```

Here `labels[0]` corresponds to document token 230. For subword tokenizers,
the standard repository models use the **first sub-token** to assign each
dataset word's label. `ce.word_labels_from_subwords()` provides that mapping.
Uncovered words yield no predicted argument, reducing recall where gold
arguments are outside the window.

## Validation before scoring

- Require exactly one row for every expected event and no unknown event IDs.
- Check document-level token coordinates, inclusive ends and exact role names.
- Preserve which checkpoint, tokenizer, code revision, split, precision,
  maximum length and batch size produced the file.
- Select models using validation data. Clearly distinguish later verification
  reruns from original benchmark results.
- Keep generated local predictions under ignored `runs/` unless a separate
  publication decision explicitly includes them.

The loader unions duplicate rows for an event rather than rejecting them;
therefore enforce unique rows as an input check. Exact Arg-C scoring requires
the correct **event, span and role**, not just an entity mention somewhere in
the same document.
