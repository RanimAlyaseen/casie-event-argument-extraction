"""TAGPRIME-inspired role-conditioned argument extraction for the CASIE notebook.

This is an adaptation, not a reproduction of TAGPRIME. The gold event trigger and
subtype are supplied to all systems. Training additionally names one target role
in the prompt and predicts O/B-Target/I-Target for that role. The inventory of
roles permitted per event subtype comes exclusively from the training split.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import casie_eval as ce
import casie_transformer as ctr

ROLE_LABELS = ["O", "B-Target", "I-Target"]


def training_role_inventory(train_records):
    roles = defaultdict(set)
    for rec in train_records:
        for arg in rec["arguments"]:
            roles[rec["event_subtype"]].add(arg["role"])
    return {subtype: sorted(items) for subtype, items in sorted(roles.items())}


class RoleEncoder(ctr.Encoder):
    def encode_role(self, rec, role, with_labels=True):
        trig = rec["trigger"]["token_ids"]
        t0, t1 = min(trig), max(trig)
        # Natural language prompt makes the target role explicit; subtype and
        # trigger are available to the ordinary DeBERTa baseline as well.
        seg_a = [f"Event: {rec['event_subtype']}",
                 f"Trigger: {' '.join(rec['tokens'][t0:t1 + 1])}",
                 f"Target role: {role}"]
        prompt_len = len(self.tok(seg_a, is_split_into_words=True,
                                  add_special_tokens=False)["input_ids"])
        budget = self.max_len - self.n_special - prompt_len
        assert budget > 2, "The role prompt exceeds the sequence length."
        lo, hi = self.window(rec, budget)
        seg_b, doc_idx = [], []
        for d in range(lo, hi):
            if d == t0:
                seg_b.append(ctr.TRIG_OPEN); doc_idx.append(None)
            seg_b.append(rec["tokens"][d]); doc_idx.append(d)
            if d == t1:
                seg_b.append(ctr.TRIG_CLOSE); doc_idx.append(None)
        enc = self.tok(seg_a, seg_b, is_split_into_words=True,
                       truncation="only_second", max_length=self.max_len)
        first_sub_doc, previous = [], None
        for segment, word in zip(enc.sequence_ids(), enc.word_ids()):
            if segment == 1 and word is not None and word != previous and doc_idx[word] is not None:
                first_sub_doc.append(doc_idx[word])
            else:
                first_sub_doc.append(None)
            previous = word if segment == 1 else None
        assert any(d is not None for d in first_sub_doc)
        item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "first_sub_doc": first_sub_doc, "event_uid": rec["event_uid"],
                "role": role, "window": (lo, hi)}
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        if with_labels:
            # Choose one BIO path for this role. Any overlapping same-role
            # arguments are reported as a representation limitation below.
            tags = [0] * (hi - lo)
            for s, e, r in sorted(ce.gold_spans(rec)):
                if r != role or s < lo or e >= hi:
                    continue
                for d in range(s, e + 1):
                    if tags[d - lo] == 0:
                        tags[d - lo] = 1 if d == s else 2
            item["labels"] = [-100 if d is None else tags[d - lo]
                              for d in first_sub_doc]
        return item


def role_pairs(records, inventory, negatives_per_event=None, seed=42):
    """Every present role plus sampled absent roles for train; all roles for eval."""
    for index, rec in enumerate(records):
        permitted = inventory[rec["event_subtype"]]
        if negatives_per_event is None:
            selected = permitted
        else:
            present = {arg["role"] for arg in rec["arguments"]}
            assert present <= set(permitted), "Training contains a role absent from its inventory."
            absent = sorted(set(permitted) - present)
            rng = random.Random(seed + index)
            selected = sorted(present) + rng.sample(absent, min(negatives_per_event, len(absent)))
        for role in selected:
            yield rec, role


@torch.no_grad()
def predict_roles(model, items, pad_id, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    predictions = defaultdict(set)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    for k in range(0, len(order), batch_size):
        group = [items[i] for i in order[k:k + batch_size]]
        batch = ctr.collate([{key: value for key, value in item.items()
                              if key not in ("labels", "role")}
                             for item in group], pad_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=device.type == "cuda"):
            guesses = model(**batch).logits.argmax(-1).cpu().tolist()
        for item, pred_ids in zip(group, guesses):
            lo, hi = item["window"]
            labels = ["O"] * (hi - lo)
            for pos, d in enumerate(item["first_sub_doc"]):
                if d is not None:
                    labels[d - lo] = ROLE_LABELS[pred_ids[pos]]
            for s, e, _ in ce.bio_to_spans(labels, offset=lo):
                predictions[item["event_uid"]].add((s, e, item["role"]))
    return dict(predictions)


def make_items(records, inventory, encoder, negatives_per_event=None):
    return [encoder.encode_role(rec, role, with_labels=negatives_per_event is not None)
            for rec, role in role_pairs(records, inventory, negatives_per_event)]


def train(train_records, val_records, baseline_dir, out_dir,
          epochs=3, batch_size=4, grad_accum=4, lr=2e-5, seed=42):
    """Warm start from the validation-selected DeBERTa; select epoch on val F1."""
    assert torch.cuda.is_available(), "Use a Colab GPU for this experiment."
    ctr.set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = training_role_inventory(train_records)
    baseline_dir = str(baseline_dir)
    cfg = AutoConfig.from_pretrained(
        baseline_dir, num_labels=3, id2label=dict(enumerate(ROLE_LABELS)),
        label2id={label: i for i, label in enumerate(ROLE_LABELS)})
    kw = {"add_prefix_space": True} if cfg.model_type in ctr.BPE_PREFIX_SPACE else {}
    tok = AutoTokenizer.from_pretrained(baseline_dir, use_fast=True, **kw)
    model = AutoModelForTokenClassification.from_pretrained(
        baseline_dir, config=cfg, ignore_mismatched_sizes=True,
        dtype=torch.float32).to("cuda")
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    encoder = RoleEncoder(tok, ROLE_LABELS, max_len=512)
    train_items = make_items(train_records, inventory, encoder, negatives_per_event=1)
    val_items = make_items(val_records, inventory, encoder)
    print(f"Role pairs: {len(train_items)} train / {len(val_items)} validation")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    loader = DataLoader(train_items, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda items: ctr.collate(items, pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    total_steps = math.ceil(len(loader) / grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    history, best_f1 = [], -1.0
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum
            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        preds = predict_roles(model, val_items, pad_id)
        for rec in val_records:
            preds.setdefault(rec["event_uid"], set())
        res = ce.evaluate(val_records, preds, n_examples=0)
        f1 = res["overall"]["arg_c_f1"]
        row = {"epoch": epoch, "train_loss": running / len(loader),
               "val_arg_c_f1": f1, "val_macro_f1": res["overall"]["macro_f1"],
               "minutes": (time.time() - started) / 60}
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} | "
              f"val Arg-C F1 {f1:.4f} | macro {row['val_macro_f1']:.4f} | "
              f"{row['minutes']:.1f} min", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            ce.save_predictions(str(out_dir / "validation_predictions.jsonl"), preds)
            print("  -> new best checkpoint saved", flush=True)
        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        (out_dir / "role_inventory.json").write_text(json.dumps(inventory, indent=2))
    del model
    torch.cuda.empty_cache()
    return history


def predict_file(checkpoint_dir, records, output_path):
    checkpoint_dir = Path(checkpoint_dir)
    inventory = json.loads((checkpoint_dir / "role_inventory.json").read_text())
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    kw = {"add_prefix_space": True} if model.config.model_type in ctr.BPE_PREFIX_SPACE else {}
    tok = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True, **kw)
    encoder = RoleEncoder(tok, ROLE_LABELS, max_len=512)
    items = make_items(records, inventory, encoder)
    preds = predict_roles(model, items, tok.pad_token_id or 0)
    for rec in records:
        preds.setdefault(rec["event_uid"], set())
    ce.save_predictions(str(output_path), preds)
    del model
    torch.cuda.empty_cache()
    return preds
