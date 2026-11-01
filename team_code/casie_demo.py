"""
casie_demo.py — Interactive demo: paste a cybersecurity report, see who did what to whom.

Team Ultron · NLP Course Project (COE025)

Pipeline
  1. Tokenize the text the same way the CASIE data is tokenized (words and punctuation split).
  2. Find event triggers with a lexicon learned from the TRAINING set: every trigger phrase that
     was annotated at least `min_count` times and is a trigger in at least `min_precision` of its
     occurrences, mapped to its most frequent event subtype. Longest match wins.
     The user can also type a trigger by hand.
  3. For every trigger, run the fine-tuned argument extractor (casie_transformer) and show the
     arguments it finds, coloured by role.

Argument models were trained and evaluated with annotated triggers. Automatic mode uses a
training-derived lexicon that can miss or invent events. The lexicon's trigger F1 does not
measure full raw-text argument extraction; evaluate_lexicon() scores the trigger step only.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import casie_eval as ce

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
MAX_TRIGGER_LEN = 6


# =============================================================================
# 1. TOKENIZATION
# =============================================================================

def tokenize(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    toks, offs = [], []
    for m in TOKEN_RE.finditer(text):
        toks.append(m.group()); offs.append((m.start(), m.end()))
    return toks, offs


# =============================================================================
# 2. TRIGGER LEXICON
# =============================================================================

def build_lexicon(train_records: List[dict], min_count: int = 2, min_precision: float = 0.3) -> dict:
    """
    Learn trigger phrases from the training set.
    Returns {"phrases": {"a data breach": {"subtype": "Databreach", "count": .., "precision": ..}}, ...}
    """
    as_trigger = Counter()
    subtype = defaultdict(Counter)
    docs, gold_pos = {}, defaultdict(set)
    for r in train_records:
        ids = r["trigger"]["token_ids"]
        phrase = tuple(t.lower() for t in r["tokens"][min(ids):max(ids) + 1])
        if len(phrase) > MAX_TRIGGER_LEN:
            continue
        docs[r["document_id"]] = r["tokens"]
        gold_pos[r["document_id"]].add((min(ids), max(ids)))
        subtype[phrase][r["event_subtype"]] += 1

    # how often does each phrase occur at all, and how often is that occurrence a trigger?
    candidates = set(subtype)
    occurs, hits = Counter(), Counter()
    for doc, toks in docs.items():
        low = [t.lower() for t in toks]
        for n in range(1, MAX_TRIGGER_LEN + 1):
            for i in range(len(low) - n + 1):
                p = tuple(low[i:i + n])
                if p in candidates:
                    occurs[p] += 1
                    hits[p] += (i, i + n - 1) in gold_pos[doc]

    phrases = {}
    for p in candidates:
        if hits[p] < min_count or occurs[p] == 0:
            continue
        prec = hits[p] / occurs[p]
        if prec >= min_precision:
            phrases[" ".join(p)] = {"subtype": subtype[p].most_common(1)[0][0],
                                    "count": hits[p], "precision": round(prec, 3)}
    return {"phrases": phrases, "min_count": min_count, "min_precision": min_precision}


def save_lexicon(lex: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False, indent=1)


def load_lexicon(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_triggers(tokens: List[str], lexicon: dict) -> List[Tuple[int, int, str]]:
    """Greedy longest-match, non-overlapping. Returns [(start, end_inclusive, subtype)]."""
    phrases = lexicon["phrases"]
    low = [t.lower() for t in tokens]
    out, i = [], 0
    while i < len(low):
        for n in range(min(MAX_TRIGGER_LEN, len(low) - i), 0, -1):
            p = " ".join(low[i:i + n])
            if p in phrases:
                out.append((i, i + n - 1, phrases[p]["subtype"]))
                i += n
                break
        else:
            i += 1
    return out


def evaluate_lexicon(lexicon: dict, records: List[dict]) -> dict:
    """Trigger detection precision / recall (exact span) on a split, per document."""
    gold = defaultdict(set)
    toks = {}
    for r in records:
        ids = r["trigger"]["token_ids"]
        gold[r["document_id"]].add((min(ids), max(ids)))
        toks[r["document_id"]] = r["tokens"]
    tp = n_pred = n_gold = sub_ok = 0
    sub_gold = {}
    for r in records:
        ids = r["trigger"]["token_ids"]
        sub_gold[(r["document_id"], min(ids), max(ids))] = r["event_subtype"]
    for d, t in toks.items():
        pred = find_triggers(t, lexicon)
        ps = {(s, e) for s, e, _ in pred}
        tp += len(ps & gold[d]); n_pred += len(ps); n_gold += len(gold[d])
        sub_ok += sum(1 for s, e, st in pred if sub_gold.get((d, s, e)) == st)
    p, r, f = ce._prf(tp, n_pred, n_gold)
    return {"precision": p, "recall": r, "f1": f, "subtype_accuracy_on_hits": sub_ok / tp if tp else 0.0,
            "phrases": len(lexicon["phrases"])}


# =============================================================================
# 3. ARGUMENT EXTRACTION
# =============================================================================

def extract(text: str, model, tok, enc, lexicon: dict,
            manual_trigger: str = "", manual_subtype: str = "") -> dict:
    """Run the whole pipeline on raw text. Returns tokens, offsets and a list of events."""
    import casie_transformer as ctr
    text = text or ""
    manual_trigger = manual_trigger or ""
    tokens, offsets = tokenize(text)
    if not tokens:
        return {"tokens": [], "offsets": [], "events": []}

    triggers = []
    if manual_trigger.strip():
        mt = [t.lower() for t in tokenize(manual_trigger)[0]]
        low = [t.lower() for t in tokens]
        for i in range(len(low) - len(mt) + 1):
            if low[i:i + len(mt)] == mt:
                triggers.append((i, i + len(mt) - 1, manual_subtype or "Databreach"))
    else:
        triggers = find_triggers(tokens, lexicon)

    recs = []
    for k, (s, e, sub) in enumerate(triggers):
        recs.append({"document_id": "demo", "event_uid": f"demo::E{k}", "event_subtype": sub,
                     "trigger": {"text": " ".join(tokens[s:e + 1]), "token_ids": list(range(s, e + 1))},
                     "tokens": tokens, "token_labels": [["O"]] * len(tokens), "arguments": []})
    enc._word_len_cache.pop("demo", None)  # new text every call
    items = [enc.encode(r, with_labels=False) for r in recs]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    preds = ctr.predict(model, items, model.config.id2label, pad_id) if items else {}

    events = []
    for r, (s, e, sub) in zip(recs, triggers):
        args = sorted(preds.get(r["event_uid"], set()))
        events.append({"subtype": sub, "trigger": (s, e), "trigger_text": r["trigger"]["text"],
                       "arguments": [{"role": role, "start": a, "end": b,
                                      "text": text[offsets[a][0]:offsets[b][1]]} for a, b, role in args]})
    return {"tokens": tokens, "offsets": offsets, "events": events}


def highlight(text: str, offsets, event: dict) -> List[Tuple[str, Optional[str]]]:
    """Segments for gr.HighlightedText: [(text, label or None), ...] in character space."""
    marks = [(offsets[a["start"]][0], offsets[a["end"]][1], a["role"]) for a in event["arguments"]]
    ts, te = event["trigger"]
    marks.append((offsets[ts][0], offsets[te][1], "TRIGGER"))
    marks.sort()
    out, pos = [], 0
    for s, e, lab in marks:
        if s < pos:          # overlapping (multi-role) span: skip the second mark
            continue
        if s > pos:
            out.append((text[pos:s], None))
        out.append((text[s:e], lab))
        pos = e
    if pos < len(text):
        out.append((text[pos:], None))
    return out


# =============================================================================
# 4. GRADIO APP
# =============================================================================

EXAMPLES = [
    "Hackers stole the names, addresses and card numbers of 2 million customers from Northwind Bank "
    "last month. The bank said the attackers used a phishing campaign to obtain employee passwords.",
    "Researchers at Contoso Labs discovered a critical vulnerability in the Fabrikam VPN client that "
    "allows remote code execution. Fabrikam released a patch on Tuesday and urged users to update to version 4.2.1.",
    "The ransomware encrypted the city's servers and the attackers demanded a ransom of $50,000 in Bitcoin.",
]
SUBTYPES = ["Databreach", "Phishing", "Ransom", "DiscoverVulnerability", "PatchVulnerability"]


def build_app(model, tok, enc, lexicon: dict, model_label: str = "RoBERTa", max_events: int = 8):
    import gradio as gr

    def run(text, manual_trigger, manual_subtype):
        try:
            return _run(text or "", manual_trigger or "", manual_subtype or "Databreach")
        except Exception as e:  # show the reason in the page instead of a bare "Error"
            import traceback; traceback.print_exc()
            return [f"**Error:** `{type(e).__name__}: {e}`"] + [gr.update(visible=False, value=None)] * max_events + [[]]

    def _run(text, manual_trigger, manual_subtype):
        res = extract(text, model, tok, enc, lexicon, manual_trigger, manual_subtype)
        evs = res["events"]
        if not evs:
            msg = "No event trigger found. Type a trigger word (e.g. *stole*, *patched*) in the box on the left."
            return [msg] + [gr.update(visible=False, value=None)] * max_events + [[]]
        summary_md = f"**{len(evs)} event(s) found.** Arguments are coloured by role; the trigger is marked TRIGGER."
        if len(evs) > max_events:
            summary_md += f" Showing the first {max_events}."
        rows, boxes = [], []
        for i in range(max_events):
            if i < len(evs):
                ev = evs[i]
                boxes.append(gr.update(visible=True, value=highlight(text, res["offsets"], ev),
                                       label=f"Event {i + 1}: {ev['subtype']}  ·  trigger \"{ev['trigger_text']}\""))
                for a in ev["arguments"]:
                    rows.append([i + 1, ev["subtype"], ev["trigger_text"], a["role"], a["text"]])
            else:
                boxes.append(gr.update(visible=False, value=None))
        return [summary_md] + boxes + [rows]

    with gr.Blocks(title="Cybersecurity Event Extractor") as app:
        gr.Markdown(f"# Cybersecurity Event Extractor\nPaste a security report. The model ({model_label}, fine-tuned on CASIE) "
                    "suggests event arguments and highlights their source text. Automatic trigger detection "
                    "may miss events; use the manual trigger option to inspect a specific event.")
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(lines=8, label="Security report", value=EXAMPLES[0])
                with gr.Accordion("Optional: choose the trigger yourself", open=False):
                    mtrig = gr.Textbox(value="", label="Trigger word or phrase (leave empty for automatic detection)")
                    msub = gr.Dropdown(SUBTYPES, value="Databreach", label="Event type of that trigger")
                btn = gr.Button("Extract", variant="primary")
                gr.Examples(EXAMPLES, inputs=text)
            with gr.Column(scale=3):
                summary = gr.Markdown()
                boxes = [gr.HighlightedText(visible=False, combine_adjacent=False, show_legend=True) for _ in range(max_events)]
        table = gr.Dataframe(headers=["Event", "Type", "Trigger", "Role", "Argument"], label="All arguments", wrap=True)
        btn.click(run, [text, mtrig, msub], [summary] + boxes + [table])
    return app
