"""
casie_eval.py — Evaluation and error analysis for CASIE event argument extraction.

Team Ultron · NLP Course Project (COE025)

All scoring is done at the ARGUMENT SPAN level, per event:
    an argument = (event_uid, start_token, end_token, role)     end is inclusive
A prediction is correct only if start, end AND role match a gold argument
of the SAME event exactly.

Why not seqeval?  The dataset has multi-label tokens (one span can carry two
roles in the same event) and event-conditioned labels (the same document is
labelled differently for each event).  seqeval only understands one flat tag
sequence, so we score sets of spans directly.  On single-label data the
numbers are identical to seqeval's strict span matching.

Usage (Colab / Python):
    import casie_eval as ce
    gold  = ce.load_jsonl("CASIE_test.jsonl")
    preds = ce.load_predictions("predictions_bert.jsonl")
    res   = ce.evaluate(gold, preds)
    ce.print_report(res)
    ce.save_report(res, "results/bert")

Command line:
    python casie_eval.py --gold CASIE_test.jsonl --pred predictions_bert.jsonl --out results/bert
    python casie_eval.py --gold CASIE_test.jsonl --pred a.jsonl --pred-b b.jsonl     # compare two models
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Span = Tuple[int, int, str]  # (start_token, end_token_inclusive, role)

ERROR_TYPES = [
    "role_confusion",       # exact same span, wrong role
    "boundary",             # right role, span overlaps but boundaries differ
    "boundary_and_role",    # overlaps a gold span, but boundaries AND role wrong
    "spurious",             # predicted span overlaps no gold argument (pure false positive)
    "missed",               # gold argument with no overlapping prediction (pure false negative)
]


# =============================================================================
# 1. LOADING
# =============================================================================

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_spans(record: dict) -> Set[Span]:
    """Gold arguments of one event record as a set of (start, end, role)."""
    spans = set()
    for arg in record["arguments"]:
        ids = arg["token_ids"]
        spans.add((min(ids), max(ids), arg["role"]))
    return spans


def bio_to_spans(labels: Sequence, offset: int = 0) -> Set[Span]:
    """
    Decode a BIO sequence into spans.

    `labels` can be
      - a flat list of strings:            ["O", "B-Victim", "I-Victim", ...]
      - a list of lists (multi-label):     [["O"], ["B-Victim", "B-Discoverer"], ...]
    `offset` is added to every position (use it when the model ran on a
    window that starts at document token `offset`).

    Decoding is lenient like conlleval/seqeval default: an I-X that does not
    follow B-X/I-X starts a new span.
    """
    labels = [list(l) if not isinstance(l, str) and hasattr(l, "__len__") else l for l in labels]  # numpy-safe
    if labels and isinstance(labels[0], (list, tuple)):
        layers: Dict[str, List[str]] = defaultdict(lambda: ["O"] * len(labels))
        for i, labs in enumerate(labels):
            for lab in labs:
                if lab != "O":
                    layers[lab[2:]][i] = lab
        out = set()
        for role, seq in layers.items():
            out |= bio_to_spans(seq, offset)
        return out

    spans = set()
    start, role = None, None
    for i, lab in enumerate(list(labels) + ["O"]):
        lab = lab or "O"
        tag, r = (lab[0], lab[2:]) if lab != "O" else ("O", None)
        continues = tag == "I" and role == r and start is not None
        if not continues and start is not None:
            spans.add((start + offset, i - 1 + offset, role))
            start, role = None, None
        if tag == "B" or (tag == "I" and not continues):
            start, role = i, r
    return spans


def load_predictions(path: str) -> Dict[str, Set[Span]]:
    """
    Load a predictions JSONL file. One line per event, keyed by `event_uid`.
    Each line uses ONE of these formats (token indices are DOCUMENT-level,
    the same indices as `tokens` in CASIE_*.jsonl):

      A) spans  (recommended)
         {"event_uid": "...", "arguments": [{"role": "Victim", "start": 12, "end": 14}, ...]}
         `end` is inclusive. `token_ids: [12, 13, 14]` is also accepted instead of start/end.

      B) BIO labels over the whole document
         {"event_uid": "...", "labels": ["O", "B-Victim", "I-Victim", ...]}

      C) BIO labels over a window
         {"event_uid": "...", "labels": [...], "window_start": 230}
    """
    preds: Dict[str, Set[Span]] = defaultdict(set)
    for row in load_jsonl(path):
        uid = row["event_uid"]
        preds[uid]  # register the event even when it has no predicted arguments
        if "arguments" in row:
            for a in row["arguments"]:
                if "token_ids" in a:
                    s, e = min(a["token_ids"]), max(a["token_ids"])
                else:
                    s, e = int(a["start"]), int(a["end"])
                preds[uid].add((s, e, a["role"]))
        elif "labels" in row:
            preds[uid] |= bio_to_spans(row["labels"], row.get("window_start", 0))
        else:
            raise ValueError(f"Prediction for {uid} has neither 'arguments' nor 'labels'")
    return dict(preds)


def save_predictions(path: str, preds: Dict[str, Iterable[Span]]) -> None:
    """Write predictions in format A. `preds` maps event_uid -> iterable of (start, end, role)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for uid, spans in preds.items():
            args = [{"role": r, "start": s, "end": e} for s, e, r in sorted(spans)]
            f.write(json.dumps({"event_uid": uid, "arguments": args}, ensure_ascii=False) + "\n")


def word_labels_from_subwords(pred_label_ids: Sequence[int],
                              word_ids: Sequence[Optional[int]],
                              id2label: Dict[int, str]) -> Dict[int, str]:
    """
    Helper for HuggingFace models: turn sub-word predictions into one label per
    word, using the FIRST sub-token of each word.

    pred_label_ids : argmax label ids for each sub-token (one window)
    word_ids       : tokenizer(...).word_ids() for that window
    returns        : {word_index: label}  (word_index as given in word_ids)
    """
    out = {}
    prev = None
    for lab_id, w in zip(pred_label_ids, word_ids):
        if w is not None and w != prev:
            out[w] = id2label[int(lab_id)]
        prev = w
    return out


# =============================================================================
# 2. METRICS
# =============================================================================

def _prf(tp: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _span_overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def evaluate(gold_records: List[dict],
             preds: Dict[str, Set[Span]],
             n_examples: int = 15,
             seed: int = 42) -> dict:
    """
    Score predictions against gold event records.

    Returns a dict with:
      overall            Arg-C (span+role) and Arg-I (span only) micro P/R/F1, macro-F1
      per_role           list of rows: role, support, predicted, tp, P, R, F1
      per_subtype        list of rows per event subtype
      errors             counts per error type
      confusion          Counter of (gold_role, predicted_role) for role confusions
      boundary_detail    Counter: too_long / too_short / shifted
      by_distance        recall by token distance from the trigger
      by_length          recall by gold span length
      examples           sampled error examples with context
      warnings           list of strings (missing events, unknown uids, ...)
    """
    gold_uids = {r["event_uid"] for r in gold_records}
    warnings = []
    unknown = set(preds) - gold_uids
    if unknown:
        warnings.append(f"{len(unknown)} predicted event_uids are not in the gold file (ignored).")
    missing = gold_uids - set(preds)
    if missing:
        warnings.append(f"{len(missing)} gold events have no prediction line (treated as 'predicted nothing').")

    role_tp, role_pred, role_gold = Counter(), Counter(), Counter()
    sub_tp, sub_pred, sub_gold = Counter(), Counter(), Counter()
    tp_c = n_pred = n_gold = tp_i = 0
    n_pred_i = n_gold_i = 0
    errors = Counter()
    confusion = Counter()
    boundary_detail = Counter()
    dist_hit, dist_tot = Counter(), Counter()
    len_hit, len_tot = Counter(), Counter()
    error_pool = defaultdict(list)
    per_doc = defaultdict(lambda: [0, 0, 0])  # doc -> [tp, n_pred, n_gold] for bootstrap

    for rec in gold_records:
        uid = rec["event_uid"]
        sub = rec.get("event_subtype", "?")
        g = gold_spans(rec)
        p = set(preds.get(uid, set()))
        tp = g & p

        # ---- Arg-C (span + role)
        tp_c += len(tp); n_pred += len(p); n_gold += len(g)
        sub_tp[sub] += len(tp); sub_pred[sub] += len(p); sub_gold[sub] += len(g)
        d = per_doc[rec["document_id"]]
        d[0] += len(tp); d[1] += len(p); d[2] += len(g)
        for s in tp: role_tp[s[2]] += 1
        for s in p:  role_pred[s[2]] += 1
        for s in g:  role_gold[s[2]] += 1

        # ---- Arg-I (span only, role ignored)
        g_i = {(s, e) for s, e, _ in g}
        p_i = {(s, e) for s, e, _ in p}
        tp_i += len(g_i & p_i)
        n_pred_i += len(p_i)
        n_gold_i += len(g_i)

        # ---- recall by distance from trigger / span length
        trig = rec.get("trigger", {}).get("token_ids") or [0]
        t0, t1 = min(trig), max(trig)
        for s in g:
            dist = 0 if s[1] >= t0 and s[0] <= t1 else min(abs(s[0] - t1), abs(t0 - s[1]))
            b = _distance_bucket(dist)
            dist_tot[b] += 1; dist_hit[b] += s in tp
            lb = _length_bucket(s[1] - s[0] + 1)
            len_tot[lb] += 1; len_hit[lb] += s in tp

        # ---- error categorisation (one-to-one greedy matching)
        g_left = sorted(g - tp)
        p_left = sorted(p - tp)
        used_g = set()
        cat_pred = {}
        # pass 1: same span, different role
        for ps in p_left:
            for gs in g_left:
                if gs in used_g: continue
                if gs[0] == ps[0] and gs[1] == ps[1]:
                    cat_pred[ps] = ("role_confusion", gs); used_g.add(gs); break
        # pass 2: overlapping, same role (pick largest overlap)
        for ps in p_left:
            if ps in cat_pred: continue
            cands = [gs for gs in g_left if gs not in used_g and gs[2] == ps[2] and _span_overlap(ps, gs)]
            if cands:
                gs = max(cands, key=lambda x: _span_overlap(ps, x))
                cat_pred[ps] = ("boundary", gs); used_g.add(gs)
        # pass 3: overlapping, different role
        for ps in p_left:
            if ps in cat_pred: continue
            cands = [gs for gs in g_left if gs not in used_g and _span_overlap(ps, gs)]
            if cands:
                gs = max(cands, key=lambda x: _span_overlap(ps, x))
                cat_pred[ps] = ("boundary_and_role", gs); used_g.add(gs)
        # leftovers
        for ps in p_left:
            if ps not in cat_pred:
                cat_pred[ps] = ("spurious", None)

        for ps, (etype, gs) in cat_pred.items():
            errors[etype] += 1
            if etype in ("role_confusion", "boundary_and_role"):
                confusion[(gs[2], ps[2])] += 1
            if etype == "boundary":
                if ps[0] <= gs[0] and ps[1] >= gs[1]:   boundary_detail["too_long"] += 1
                elif ps[0] >= gs[0] and ps[1] <= gs[1]: boundary_detail["too_short"] += 1
                else:                                   boundary_detail["shifted"] += 1
            error_pool[etype].append(_example(rec, etype, gs, ps))
        for gs in g_left:
            if gs not in used_g:
                errors["missed"] += 1
                error_pool["missed"].append(_example(rec, "missed", gs, None))

    P, R, F = _prf(tp_c, n_pred, n_gold)
    Pi, Ri, Fi = _prf(tp_i, n_pred_i, n_gold_i)

    per_role = []
    for role in sorted(set(role_gold) | set(role_pred)):
        p_, r_, f_ = _prf(role_tp[role], role_pred[role], role_gold[role])
        per_role.append({"role": role, "support": role_gold[role], "predicted": role_pred[role],
                         "tp": role_tp[role], "precision": p_, "recall": r_, "f1": f_})
    roles_with_support = [r for r in per_role if r["support"] > 0]
    macro_f1 = sum(r["f1"] for r in roles_with_support) / len(roles_with_support) if roles_with_support else 0.0

    per_subtype = []
    for sub in sorted(sub_gold):
        p_, r_, f_ = _prf(sub_tp[sub], sub_pred[sub], sub_gold[sub])
        per_subtype.append({"event_subtype": sub, "support": sub_gold[sub], "predicted": sub_pred[sub],
                            "tp": sub_tp[sub], "precision": p_, "recall": r_, "f1": f_})

    rng = random.Random(seed)
    examples = []
    for etype in ERROR_TYPES:
        pool = error_pool.get(etype, [])
        examples += rng.sample(pool, min(n_examples, len(pool)))

    return {
        "overall": {
            "arg_c_precision": P, "arg_c_recall": R, "arg_c_f1": F,
            "arg_i_precision": Pi, "arg_i_recall": Ri, "arg_i_f1": Fi,
            "macro_f1": macro_f1,
            "gold_arguments": n_gold, "predicted_arguments": n_pred, "correct": tp_c,
            "events": len(gold_records),
        },
        "per_role": per_role,
        "per_subtype": per_subtype,
        "errors": {k: errors.get(k, 0) for k in ERROR_TYPES},
        "confusion": confusion,
        "boundary_detail": dict(boundary_detail),
        "by_distance": _bucket_rows(dist_hit, dist_tot, DISTANCE_BUCKETS),
        "by_length": _bucket_rows(len_hit, len_tot, LENGTH_BUCKETS),
        "examples": examples,
        "warnings": warnings,
        "_per_doc": dict(per_doc),
    }


DISTANCE_BUCKETS = ["0 (overlaps trigger)", "1-5", "6-10", "11-20", "21-50", "51+"]
LENGTH_BUCKETS = ["1", "2", "3-4", "5-8", "9+"]


def _distance_bucket(d: int) -> str:
    if d == 0: return DISTANCE_BUCKETS[0]
    if d <= 5: return "1-5"
    if d <= 10: return "6-10"
    if d <= 20: return "11-20"
    if d <= 50: return "21-50"
    return "51+"


def _length_bucket(n: int) -> str:
    if n == 1: return "1"
    if n == 2: return "2"
    if n <= 4: return "3-4"
    if n <= 8: return "5-8"
    return "9+"


def _bucket_rows(hit, tot, order):
    return [{"bucket": b, "gold": tot[b], "found": hit[b],
             "recall": hit[b] / tot[b] if tot[b] else 0.0} for b in order if tot[b]]


def _example(rec: dict, etype: str, gs: Optional[Span], ps: Optional[Span], ctx: int = 12) -> dict:
    toks = rec["tokens"]
    anchor = gs or ps
    lo = max(0, min(x[0] for x in (gs, ps) if x) - ctx)
    hi = min(len(toks), max(x[1] for x in (gs, ps) if x) + ctx + 1)
    trig = rec.get("trigger", {})
    return {
        "error_type": etype,
        "event_uid": rec["event_uid"],
        "event_subtype": rec.get("event_subtype"),
        "trigger": trig.get("text"),
        "gold_role": gs[2] if gs else None,
        "gold_text": " ".join(toks[gs[0]:gs[1] + 1]) if gs else None,
        "pred_role": ps[2] if ps else None,
        "pred_text": " ".join(toks[ps[0]:ps[1] + 1]) if ps else None,
        "context": ("... " if lo > 0 else "") + " ".join(toks[lo:hi]) + (" ..." if hi < len(toks) else ""),
    }


# =============================================================================
# 3. CONFIDENCE INTERVALS AND MODEL COMPARISON (document-level bootstrap)
# =============================================================================

def bootstrap_f1(res: dict, n: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """95% CI of Arg-C micro-F1, resampling DOCUMENTS (events of a doc are not independent)."""
    docs = list(res["_per_doc"].values())
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        tp = npd = ng = 0
        for _ in docs:
            a = docs[rng.randrange(len(docs))]
            tp += a[0]; npd += a[1]; ng += a[2]
        scores.append(_prf(tp, npd, ng)[2])
    scores.sort()
    return scores[int(0.025 * n)], scores[int(0.975 * n) - 1]


def compare_models(res_a: dict, res_b: dict, n: int = 1000, seed: int = 42) -> dict:
    """
    Paired bootstrap over documents. Returns the observed F1 difference (B - A)
    and how often B beats A across resamples. 'b_better_fraction' >= 0.95 means
    B is reliably better on this test set.
    """
    docs = sorted(set(res_a["_per_doc"]) & set(res_b["_per_doc"]))
    A, B = res_a["_per_doc"], res_b["_per_doc"]
    rng = random.Random(seed)
    wins = 0
    for _ in range(n):
        sa = [0, 0, 0]; sb = [0, 0, 0]
        for _ in docs:
            d = docs[rng.randrange(len(docs))]
            for k in range(3):
                sa[k] += A[d][k]; sb[k] += B[d][k]
        wins += _prf(*sb)[2] > _prf(*sa)[2]
    return {"f1_a": res_a["overall"]["arg_c_f1"], "f1_b": res_b["overall"]["arg_c_f1"],
            "difference": res_b["overall"]["arg_c_f1"] - res_a["overall"]["arg_c_f1"],
            "b_better_fraction": wins / n}


# =============================================================================
# 4. REPORTING
# =============================================================================

def print_report(res: dict, name: str = "model", top_confusions: int = 10) -> None:
    o = res["overall"]
    for w in res["warnings"]:
        print("WARNING:", w)
    print("=" * 72)
    print(f"RESULTS — {name}   ({o['events']} events, {o['gold_arguments']} gold arguments)")
    print("=" * 72)
    print(f"Arg-C (span + role)  P {o['arg_c_precision']:.4f}  R {o['arg_c_recall']:.4f}  F1 {o['arg_c_f1']:.4f}")
    print(f"Arg-I (span only)    P {o['arg_i_precision']:.4f}  R {o['arg_i_recall']:.4f}  F1 {o['arg_i_f1']:.4f}")
    print(f"Macro-F1 over roles  {o['macro_f1']:.4f}")
    print(f"Predicted {o['predicted_arguments']}, correct {o['correct']}")

    print("\nPER ROLE (sorted by F1)")
    print(f"{'role':28s}{'support':>8s}{'pred':>7s}{'P':>8s}{'R':>8s}{'F1':>8s}")
    for r in sorted(res["per_role"], key=lambda x: x["f1"]):
        flag = "  (<10, unstable)" if r["support"] < 10 else ""
        print(f"{r['role']:28s}{r['support']:8d}{r['predicted']:7d}{r['precision']:8.3f}{r['recall']:8.3f}{r['f1']:8.3f}{flag}")

    print("\nPER EVENT SUBTYPE")
    for r in res["per_subtype"]:
        print(f"{r['event_subtype']:28s}{r['support']:8d}{r['predicted']:7d}{r['precision']:8.3f}{r['recall']:8.3f}{r['f1']:8.3f}")

    total = sum(res["errors"].values()) or 1
    print("\nERROR TYPES")
    for k in ERROR_TYPES:
        print(f"{k:22s}{res['errors'][k]:6d}  ({100 * res['errors'][k] / total:5.1f}%)")
    if res["boundary_detail"]:
        print("  boundary detail:", res["boundary_detail"])

    if res["confusion"]:
        print(f"\nTOP ROLE CONFUSIONS (gold -> predicted)")
        for (g, p), c in res["confusion"].most_common(top_confusions):
            print(f"  {g:26s} -> {p:26s}{c:5d}")

    print("\nRECALL BY DISTANCE FROM TRIGGER (tokens)")
    for r in res["by_distance"]:
        print(f"  {r['bucket']:22s}{r['found']:6d}/{r['gold']:<6d}{r['recall']:.3f}")
    print("\nRECALL BY GOLD SPAN LENGTH (tokens)")
    for r in res["by_length"]:
        print(f"  {r['bucket']:22s}{r['found']:6d}/{r['gold']:<6d}{r['recall']:.3f}")


def save_report(res: dict, out_dir: str, name: str = "model") -> None:
    """Write CSV tables + summary.json + a markdown summary for the README/slides."""
    import csv
    os.makedirs(out_dir, exist_ok=True)

    def write_csv(fname, rows):
        if not rows: return
        with open(os.path.join(out_dir, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write_csv("per_role.csv", res["per_role"])
    write_csv("per_subtype.csv", res["per_subtype"])
    write_csv("error_types.csv", [{"error_type": k, "count": v} for k, v in res["errors"].items()])
    write_csv("role_confusions.csv", [{"gold_role": g, "pred_role": p, "count": c}
                                      for (g, p), c in res["confusion"].most_common()])
    write_csv("recall_by_distance.csv", res["by_distance"])
    write_csv("recall_by_length.csv", res["by_length"])
    write_csv("error_examples.csv", res["examples"])

    summary = {k: v for k, v in res.items() if k in ("overall", "errors", "boundary_detail", "warnings")}
    summary["name"] = name
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    o = res["overall"]
    md = [f"| Model | Arg-C P | Arg-C R | Arg-C F1 | Arg-I F1 | Macro-F1 |",
          f"|---|---|---|---|---|---|",
          f"| {name} | {o['arg_c_precision']:.3f} | {o['arg_c_recall']:.3f} | {o['arg_c_f1']:.3f} "
          f"| {o['arg_i_f1']:.3f} | {o['macro_f1']:.3f} |", ""]
    with open(os.path.join(out_dir, "results_row.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def results_table(results: Dict[str, dict], with_ci: bool = True) -> str:
    """Markdown table comparing several models: {"BERT": res1, "RoBERTa": res2, ...}."""
    lines = ["| Model | Precision | Recall | F1 (micro) | 95% CI | Arg-I F1 | F1 (macro) |",
             "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        o = res["overall"]
        ci = "%.3f–%.3f" % bootstrap_f1(res) if with_ci else "–"
        lines.append(f"| {name} | {o['arg_c_precision']:.3f} | {o['arg_c_recall']:.3f} | {o['arg_c_f1']:.3f} "
                     f"| {ci} | {o['arg_i_f1']:.3f} | {o['macro_f1']:.3f} |")
    return "\n".join(lines)


# =============================================================================
# 5. CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="CASIE event argument extraction evaluation")
    ap.add_argument("--gold", required=True, help="CASIE_validation.jsonl or CASIE_test.jsonl")
    ap.add_argument("--pred", required=True, help="predictions JSONL")
    ap.add_argument("--name", default="model")
    ap.add_argument("--out", default=None, help="folder for CSV/JSON reports")
    ap.add_argument("--pred-b", default=None, help="second predictions file to compare against --pred")
    ap.add_argument("--name-b", default="model_b")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    res = evaluate(gold, load_predictions(args.pred))
    print_report(res, args.name)
    lo, hi = bootstrap_f1(res)
    print(f"\nArg-C F1 95% CI (document bootstrap): {lo:.4f} – {hi:.4f}")
    if args.out:
        save_report(res, args.out, args.name)
        print("Reports written to", args.out)
    if args.pred_b:
        res_b = evaluate(gold, load_predictions(args.pred_b))
        print_report(res_b, args.name_b)
        cmp = compare_models(res, res_b)
        print(f"\nPAIRED BOOTSTRAP: {args.name_b} - {args.name} = {cmp['difference']:+.4f} F1; "
              f"{args.name_b} better in {100 * cmp['b_better_fraction']:.1f}% of resamples")


if __name__ == "__main__":
    main()
