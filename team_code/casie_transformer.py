"""
casie_transformer.py — Fine-tune a BERT-style encoder for CASIE event argument extraction.

Team Ultron · NLP Course Project (COE025)

Same code for every encoder, so model comparisons are fair. Only MODEL_NAME changes:
    "bert-base-cased"            general-domain BERT
    "roberta-base"               general-domain RoBERTa
    "ehsanaghaei/SecureBERT"     RoBERTa further pre-trained on cybersecurity text
    "markusbayer/CySecBERT"      BERT further pre-trained on cybersecurity text
    "jackaduma/SecBERT"          BERT trained from scratch on cybersecurity text

Input encoding for one event record (text pair):
    segment A:  <event subtype>  <trigger text>
    segment B:  ... words left of trigger  <t> trigger words </t>  words right of trigger ...
The window in segment B is grown outwards from the trigger until the sub-word budget
(max_len) is full, so the trigger is always inside and as much context as possible fits.
Arguments that fall outside the window cannot be predicted. They still count as missed
in the evaluation, so the reported scores stay honest.

Labels: one BIO label per dataset word (O + B-/I- for each of the 26 roles), predicted on
the first sub-token of each word. The 3 multi-label tokens in the corpus keep their first
label for training. Evaluation still uses the full gold set.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

import casie_eval as ce

TRIG_OPEN, TRIG_CLOSE = "<t>", "</t>"
BPE_PREFIX_SPACE = {"roberta", "longformer", "bart", "gpt2", "deberta", "deberta-v2", "mpnet", "modernbert"}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_label_list(records: List[dict]) -> List[str]:
    roles = sorted({a["role"] for r in records for a in r["arguments"]})
    return ["O"] + [f"{p}-{r}" for r in roles for p in ("B", "I")]


def load_tokenizer_and_model(model_name: str, labels: List[str]):
    from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
    cfg = AutoConfig.from_pretrained(model_name, num_labels=len(labels),
                                     id2label=dict(enumerate(labels)),
                                     label2id={l: i for i, l in enumerate(labels)})
    kw = {"add_prefix_space": True} if cfg.model_type in BPE_PREFIX_SPACE else {}
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kw)
    tok.add_tokens([TRIG_OPEN, TRIG_CLOSE], special_tokens=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=cfg, ignore_mismatched_sizes=True, dtype=torch.float32)
    model.resize_token_embeddings(len(tok))
    return tok, model


# =============================================================================
# ENCODING
# =============================================================================

class Encoder:
    """Turns event records into model inputs with a trigger-centred window."""

    def __init__(self, tokenizer, labels: List[str], max_len: int = 512):
        self.tok = tokenizer
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.max_len = max_len
        self._word_len_cache: Dict[str, List[int]] = {}
        self.n_special = tokenizer.num_special_tokens_to_add(pair=True)

    def word_lengths(self, rec: dict) -> List[int]:
        doc = rec["document_id"]
        if doc not in self._word_len_cache:
            enc = self.tok(rec["tokens"], is_split_into_words=True, add_special_tokens=False)
            counts = [0] * len(rec["tokens"])
            for w in enc.word_ids():
                if w is not None:
                    counts[w] += 1
            self._word_len_cache[doc] = [max(1, c) for c in counts]
        return self._word_len_cache[doc]

    def window(self, rec: dict, budget: int):
        """Return (lo, hi) word indices [lo, hi) of a window around the trigger."""
        lens = self.word_lengths(rec)
        t = rec["trigger"]["token_ids"]
        lo, hi = min(t), max(t) + 1
        used = sum(lens[lo:hi]) + 2  # + the two marker tokens
        left, right = lo - 1, hi
        while True:
            grew = False
            if left >= 0 and used + lens[left] <= budget:
                used += lens[left]; lo = left; left -= 1; grew = True
            if right < len(lens) and used + lens[right] <= budget:
                used += lens[right]; hi = right + 1; right += 1; grew = True
            if not grew:
                break
        return lo, hi

    def encode(self, rec: dict, with_labels: bool = True) -> dict:
        trig = rec["trigger"]["token_ids"]
        t0, t1 = min(trig), max(trig)
        seg_a = [rec.get("event_subtype", "event")] + rec["tokens"][t0:t1 + 1]
        len_a = len(self.tok(seg_a, is_split_into_words=True, add_special_tokens=False)["input_ids"])
        budget = self.max_len - self.n_special - len_a
        lo, hi = self.window(rec, budget)

        seg_b, doc_idx = [], []          # doc_idx: document word index per segment-B word (None = marker)
        for i in range(lo, hi):
            if i == t0:
                seg_b.append(TRIG_OPEN); doc_idx.append(None)
            seg_b.append(rec["tokens"][i]); doc_idx.append(i)
            if i == t1:
                seg_b.append(TRIG_CLOSE); doc_idx.append(None)

        enc = self.tok(seg_a, seg_b, is_split_into_words=True, truncation="only_second",
                       max_length=self.max_len)
        seq_ids = enc.sequence_ids()
        word_ids = enc.word_ids()

        # map each sub-token to a document word index if it is the FIRST sub-token of a seg-B word
        first_sub_doc = []
        prev = None
        for s, w in zip(seq_ids, word_ids):
            if s == 1 and w is not None and w != prev and doc_idx[w] is not None:
                first_sub_doc.append(doc_idx[w])
            else:
                first_sub_doc.append(None)
            prev = w if s == 1 else None

        item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "first_sub_doc": first_sub_doc, "event_uid": rec["event_uid"], "window": (lo, hi)}
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        if with_labels:
            labs = []
            for d in first_sub_doc:
                if d is None:
                    labs.append(-100)
                else:
                    lab = rec["token_labels"][d][0]
                    labs.append(self.label2id.get(lab, 0))
            item["labels"] = labs
        return item


def collate(batch: List[dict], pad_id: int) -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": torch.full((len(batch), n), pad_id, dtype=torch.long),
           "attention_mask": torch.zeros((len(batch), n), dtype=torch.long)}
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.zeros((len(batch), n), dtype=torch.long)
    if "labels" in batch[0]:
        out["labels"] = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        out["input_ids"][i, :L] = torch.tensor(b["input_ids"])
        out["attention_mask"][i, :L] = 1
        if "token_type_ids" in out:
            out["token_type_ids"][i, :L] = torch.tensor(b["token_type_ids"])
        if "labels" in out:
            out["labels"][i, :L] = torch.tensor(b["labels"])
    return out


def window_coverage(records: List[dict], encoded: List[dict]) -> float:
    """Share of gold arguments that lie completely inside the model's input window."""
    inside = total = 0
    for rec, it in zip(records, encoded):
        lo, hi = it["window"]
        seen = {d for d in it["first_sub_doc"] if d is not None}
        for s, e, _ in ce.gold_spans(rec):
            total += 1
            inside += all(i in seen for i in range(s, e + 1))
    return inside / total if total else 1.0


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def predict(model, encoded: List[dict], id2label: Dict[int, str], pad_id: int,
            batch_size: int = 32, device: Optional[str] = None) -> Dict[str, set]:
    device = device or next(model.parameters()).device
    model.eval()
    preds = {}
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["input_ids"]))  # faster batching
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        batch = collate([{x: v for x, v in encoded[i].items() if x != "labels"} for i in idx], pad_id)
        batch = {x: v.to(device) for x, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            logits = model(**batch).logits
        am = logits.argmax(-1).cpu().tolist()
        for j, i in enumerate(idx):
            it = encoded[i]
            lo, hi = it["window"]
            labels = ["O"] * (hi - lo)
            for pos, d in enumerate(it["first_sub_doc"]):
                if d is not None:
                    labels[d - lo] = id2label[am[j][pos]]
            preds[it["event_uid"]] = ce.bio_to_spans(labels, offset=lo)
    return preds


# =============================================================================
# TRAINING
# =============================================================================

def train(train_records: List[dict],
          val_records: List[dict],
          model_name: str = "bert-base-cased",
          out_dir: str = "runs/bert",
          max_len: int = 512,
          epochs: int = 5,
          batch_size: int = 8,
          grad_accum: int = 1,
          lr: float = 3e-5,
          weight_decay: float = 0.01,
          warmup_ratio: float = 0.1,
          patience: int = 2,
          seed: int = 42,
          fp16: bool = True,
          log_every: int = 100):
    """
    Fine-tune, pick the best epoch by validation Arg-C F1, save the best model to out_dir.
    Returns (model, tokenizer, encoder, history).
    """
    from transformers import get_linear_schedule_with_warmup
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = fp16 and device == "cuda"

    labels = build_label_list(train_records)
    id2label = dict(enumerate(labels))
    tok, model = load_tokenizer_and_model(model_name, labels)
    model.to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    enc = Encoder(tok, labels, max_len)
    t = time.time()
    train_items = [enc.encode(r) for r in train_records]
    val_items = [enc.encode(r) for r in val_records]
    cov_tr, cov_va = window_coverage(train_records, train_items), window_coverage(val_records, val_items)
    print(f"Encoded {len(train_items)} train / {len(val_items)} val events in {time.time() - t:.0f}s")
    print(f"Gold arguments inside the input window: train {cov_tr:.1%}, val {cov_va:.1%}  (upper bound on recall)")

    loader = DataLoader(train_items, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    steps = math.ceil(len(loader) / grad_accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = dict(model_name=model_name, max_len=max_len, epochs=epochs, batch_size=batch_size,
                  grad_accum=grad_accum, lr=lr, weight_decay=weight_decay, warmup_ratio=warmup_ratio,
                  seed=seed, fp16=use_amp, window="trigger-centred, grown outwards until max_len",
                  trigger_marking=f"text pair (subtype + trigger) and {TRIG_OPEN} {TRIG_CLOSE} markers",
                  window_coverage_train=cov_tr, window_coverage_val=cov_va)
    history, best_f1, bad = [], -1.0, 0

    for ep in range(1, epochs + 1):
        model.train()
        t, running = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum
            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if log_every and step % log_every == 0:
                print(f"  epoch {ep} step {step}/{len(loader)} loss {running / step:.4f}")

        val_preds = predict(model, val_items, id2label, pad_id)
        res = ce.evaluate(val_records, val_preds)
        o = res["overall"]
        history.append({"epoch": ep, "train_loss": running / len(loader), "val_arg_c_f1": o["arg_c_f1"],
                        "val_arg_c_p": o["arg_c_precision"], "val_arg_c_r": o["arg_c_recall"],
                        "val_macro_f1": o["macro_f1"], "minutes": (time.time() - t) / 60})
        print(f"epoch {ep}: loss {running / len(loader):.4f} | val Arg-C P {o['arg_c_precision']:.3f} "
              f"R {o['arg_c_recall']:.3f} F1 {o['arg_c_f1']:.3f} | macro {o['macro_f1']:.3f} | {(time.time() - t) / 60:.1f} min")

        if o["arg_c_f1"] > best_f1:
            best_f1, bad = o["arg_c_f1"], 0
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
            print(f"  -> new best, saved to {out_dir}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no improvement for {patience} epochs)")
                break

    config["best_val_arg_c_f1"] = best_f1
    config["best_epoch"] = max(history, key=lambda h: h["val_arg_c_f1"])["epoch"]
    with open(os.path.join(out_dir, "training_config.json"), "w") as f:
        json.dump({"config": config, "history": history}, f, indent=2)

    # reload best weights
    from transformers import AutoModelForTokenClassification
    model = AutoModelForTokenClassification.from_pretrained(out_dir).to(device)
    return model, tok, enc, history


def predict_file(model, tok, enc: Encoder, records: List[dict], out_path: str) -> Dict[str, set]:
    """Encode records, predict, and write a predictions JSONL for casie_eval."""
    items = [enc.encode(r, with_labels=False) for r in records]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    preds = predict(model, items, model.config.id2label, pad_id)
    for r in records:                       # guarantee one line per event
        preds.setdefault(r["event_uid"], set())
    ce.save_predictions(out_path, preds)
    return preds


def load_trained(out_dir: str, max_len: int = 512):
    """Load a fine-tuned model saved by train()."""
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(out_dir).to(device)
    kw = {"add_prefix_space": True} if model.config.model_type in BPE_PREFIX_SPACE else {}
    tok = AutoTokenizer.from_pretrained(out_dir, use_fast=True, **kw)
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    return model, tok, Encoder(tok, labels, max_len)


# =============================================================================
# LATENCY
# =============================================================================

@torch.no_grad()
def measure_latency(model, tok, enc: "Encoder", records: List[dict], batch_size: int = 32,
                    n_single: int = 100, warmup: int = 5, seed: int = 42) -> dict:
    """
    Inference speed on the current device.
      single_event_ms : median time to process ONE event (batch size 1), what a user of the demo waits
      throughput_eps  : events per second when the whole split is processed in batches
    Encoding (tokenization) is included in the single-event time, excluded from throughput.
    """
    import statistics
    device = next(model.parameters()).device
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    model.eval()

    rng = random.Random(seed)
    sample = rng.sample(records, min(n_single, len(records)))
    for r in sample[:warmup]:
        predict(model, [enc.encode(r, with_labels=False)], model.config.id2label, pad_id, batch_size=1)
    times = []
    for r in sample:
        enc._word_len_cache.pop(r["document_id"], None)
        sync(); t = time.perf_counter()
        predict(model, [enc.encode(r, with_labels=False)], model.config.id2label, pad_id, batch_size=1)
        sync(); times.append((time.perf_counter() - t) * 1000)

    items = [enc.encode(r, with_labels=False) for r in records]
    sync(); t = time.perf_counter()
    predict(model, items, model.config.id2label, pad_id, batch_size=batch_size)
    sync(); total = time.perf_counter() - t

    n_params = sum(p.numel() for p in model.parameters())
    return {"device": str(device), "params_millions": round(n_params / 1e6, 1),
            "single_event_ms_median": round(statistics.median(times), 1),
            "single_event_ms_p90": round(sorted(times)[int(0.9 * len(times)) - 1], 1),
            "throughput_events_per_s": round(len(items) / total, 1), "events": len(items)}
