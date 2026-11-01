"""
casie_crf.py — Feature-based CRF baseline for CASIE event argument extraction.

Team Ultron · NLP Course Project (COE025)

A linear-chain CRF (sklearn-crfsuite) over a window of words around the trigger.
It needs no GPU and no pre-training. It sets the lower bound the transformers have to beat.

Features per word:
  - lexical:    lowercased word, prefix/suffix (3), shape (Xxxx / dddd), capitalisation, digits, punctuation
  - context:    the same for words at -2..+2
  - event:      event subtype, signed distance to the trigger (bucketed), left/right of trigger,
                is-trigger, same sentence as trigger, and conjunctions subtype x position
The event features are what make it an *argument* extractor, not plain NER: the same word
gets different features for different events of the same document.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List

import casie_eval as ce


def _shape(w: str) -> str:
    s = re.sub("[A-Z]", "X", w)
    s = re.sub("[a-z]", "x", s)
    s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def _dist_bucket(d: int) -> str:
    a = abs(d)
    b = "0" if a == 0 else "1" if a == 1 else "2" if a == 2 else "3-5" if a <= 5 else \
        "6-10" if a <= 10 else "11-20" if a <= 20 else "21-40" if a <= 40 else "41+"
    return ("-" if d < 0 else "+") + b


SENT_END = {".", "!", "?"}


def _window(rec: dict, size: int):
    t = rec["trigger"]["token_ids"]
    return max(0, min(t) - size), min(len(rec["tokens"]), max(t) + size + 1)


def record_features(rec: dict, size: int = 100):
    toks = rec["tokens"]
    t = rec["trigger"]["token_ids"]
    t0, t1 = min(t), max(t)
    lo, hi = _window(rec, size)
    sub = rec.get("event_subtype", "?")

    # sentence index per word (simple: split after . ! ?)
    sent, s = [], 0
    for w in toks:
        sent.append(s)
        if w in SENT_END:
            s += 1
    trig_sent = sent[t0]

    def lex(i, p):
        if i < 0 or i >= len(toks):
            return {f"{p}:pad": 1.0}
        w = toks[i]
        return {f"{p}:lower": w.lower(), f"{p}:shape": _shape(w)}

    feats = []
    for i in range(lo, hi):
        w = toks[i]
        d = 0 if t0 <= i <= t1 else (i - t1 if i > t1 else i - t0)
        pos = _dist_bucket(d)
        sd = sent[i] - trig_sent
        sentrel = "same" if sd == 0 else ("prev" if sd == -1 else "next" if sd == 1 else "far")
        f = {
            "bias": 1.0,
            "w.lower": w.lower(), "w.pre3": w[:3].lower(), "w.suf3": w[-3:].lower(),
            "w.shape": _shape(w), "w.istitle": w.istitle(), "w.isupper": w.isupper(),
            "w.hasdigit": any(c.isdigit() for c in w), "w.ispunct": not any(c.isalnum() for c in w),
            "ev.subtype": sub, "ev.dist": pos, "ev.side": "T" if d == 0 else ("L" if d < 0 else "R"),
            "ev.sent": sentrel,
            "ev.sub_dist": f"{sub}|{pos}", "ev.sub_sent": f"{sub}|{sentrel}",
            "ev.sub_shape": f"{sub}|{_shape(w)}",
            "ev.trigger": " ".join(toks[t0:t1 + 1]).lower(),
        }
        for k in (-2, -1, 1, 2):
            f.update(lex(i + k, f"{k:+d}"))
        f["-1:w|w"] = f"{toks[i - 1].lower() if i > 0 else '<s>'}|{w.lower()}"
        feats.append(f)
    return feats, lo


def record_labels(rec: dict, size: int = 100) -> List[str]:
    lo, hi = _window(rec, size)
    return [rec["token_labels"][i][0] for i in range(lo, hi)]


def train_crf(train_records: List[dict], window: int = 100, c1: float = 0.05, c2: float = 0.05,
              max_iterations: int = 150, verbose: bool = True):
    import sklearn_crfsuite
    t = time.time()
    X = [record_features(r, window)[0] for r in train_records]
    y = [record_labels(r, window) for r in train_records]
    if verbose:
        print(f"Features built for {len(X)} events in {time.time() - t:.0f}s, training CRF ...")
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations,
                               all_possible_transitions=True)
    t = time.time()
    crf.fit(X, y)
    if verbose:
        print(f"CRF trained in {(time.time() - t) / 60:.1f} min")
    crf.window_ = window
    return crf


def predict_crf(crf, records: List[dict]) -> Dict[str, set]:
    window = getattr(crf, "window_", 100)
    feats = [record_features(r, window) for r in records]
    labels = crf.predict([f for f, _ in feats])
    return {r["event_uid"]: ce.bio_to_spans(l, offset=lo) for r, l, (_, lo) in zip(records, labels, feats)}


def measure_latency_crf(crf, records: List[dict], n_single: int = 100, seed: int = 42) -> dict:
    """Same measurements as casie_transformer.measure_latency, for the CRF (CPU)."""
    import random, statistics
    sample = random.Random(seed).sample(records, min(n_single, len(records)))
    times = []
    for r in sample:
        t = time.perf_counter(); predict_crf(crf, [r]); times.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter(); predict_crf(crf, records); total = time.perf_counter() - t
    return {"device": "cpu", "params_millions": round(len(crf.state_features_) / 1e6, 2),
            "single_event_ms_median": round(statistics.median(times), 1),
            "single_event_ms_p90": round(sorted(times)[int(0.9 * len(times)) - 1], 1),
            "throughput_events_per_s": round(len(records) / total, 1), "events": len(records)}
