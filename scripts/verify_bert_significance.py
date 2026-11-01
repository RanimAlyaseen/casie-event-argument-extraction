"""Audit supplied BERT predictions; no model loading, training, or Git writes.

Primary analysis: 100,000 paired document bootstrap replicates for the
percentile CI, and 100,000 whole-document label swaps for a two-sided
Monte Carlo randomization test. Both use independent random.Random(42).
The first 5,000 bootstrap replicates are a historical-method diagnostic.
Only the Python standard library and the repository's casie_eval are used.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prf(counts):
    tp, predicted, gold = counts
    return (tp / predicted if predicted else 0.0,
            tp / gold if gold else 0.0,
            2 * tp / (predicted + gold) if predicted + gold else 0.0)


def percentile(sorted_values, q):
    """Linear interpolation at (n-1)*q; same convention as NumPy linear."""
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return (sorted_values[lower] * (upper - index)
            + sorted_values[upper] * (index - lower)
            if lower != upper else sorted_values[lower])


def summarize_differences(draws):
    out = {}
    for i, metric in enumerate(("precision", "recall", "f1")):
        values = sorted(row[i] for row in draws)
        negative = sum(x < 0 for x in values)
        zero = sum(x == 0 for x in values)
        positive = len(values) - negative - zero
        out[metric] = {
            "percentile_ci_95": [percentile(values, .025), percentile(values, .975)],
            "repository_order_statistic_ci_95": [values[int(.025 * len(values))],
                                                    values[int(.975 * len(values)) - 1]],
            "negative_replicates": negative,
            "zero_replicates": zero,
            "positive_replicates": positive,
            "uncased_better_fraction": positive / len(values),
            # Descriptive tail diagnostic, explicitly NOT the primary p-value.
            "twice_smaller_bootstrap_tail_fraction": min(1., 2 * min(
                negative + zero, positive + zero) / len(values)),
        }
    return out


def paired_bootstrap(a, b, n, seed):
    rng = random.Random(seed)
    pairs = list(zip(a, b))
    draws = []
    for _ in range(n):
        at = ap = ag = bt = bp = bg = 0
        for _ in pairs:
            aa, bb = pairs[rng.randrange(len(pairs))]
            at += aa[0]; ap += aa[1]; ag += aa[2]
            bt += bb[0]; bp += bb[1]; bg += bb[2]
        pa, pb = prf((at, ap, ag)), prf((bt, bp, bg))
        draws.append(tuple(y - x for x, y in zip(pa, pb)))
    return draws


def pooled_counts(rows):
    return tuple(sum(row[i] for row in rows) for i in range(3))


def swapped_counts(a, b, mask):
    """Swap the complete paired count triples, keeping each document intact."""
    sa, sb = [0, 0, 0], [0, 0, 0]
    for i, (aa, bb) in enumerate(zip(a, b)):
        if (mask >> i) & 1:
            aa, bb = bb, aa
        for k in range(3):
            sa[k] += aa[k]
            sb[k] += bb[k]
    return sa, sb


def f1_fraction(counts):
    return Fraction(2 * counts[0], counts[1] + counts[2])


def randomization(a, b, n, seed):
    assert len(a) == len(b) and all(aa[2] == bb[2] for aa, bb in zip(a, b))
    total_a, total_b = pooled_counts(a), pooled_counts(b)
    # Exact integer comparison avoids rounding ambiguity at the observed boundary.
    observed = abs(f1_fraction(total_b) - f1_fraction(total_a))
    combined_tp = total_a[0] + total_b[0]
    combined_pred = total_a[1] + total_b[1]
    gold = total_a[2]
    changes = [(bb[0] - aa[0], bb[1] - aa[1]) for aa, bb in zip(a, b)]
    rng = random.Random(seed)
    extremes = 0
    for _ in range(n):
        mask = rng.getrandbits(len(a))
        ta, na = total_a[0], total_a[1]
        for dt, dn in changes:
            if mask & 1:
                ta += dt
                na += dn
            mask >>= 1
        tb, nb = combined_tp - ta, combined_pred - na
        da, db = na + gold, nb + gold
        numerator = abs(2 * (tb * da - ta * db))
        denominator = da * db
        extremes += numerator * observed.denominator >= observed.numerator * denominator
    raw = extremes / n
    return {
        "replicates": n, "seed": seed,
        "null": "Model labels are exchangeable within each complete test document.",
        "statistic": "Absolute difference in pooled exact event-specific Arg-C micro F1",
        "sidedness": "two-sided",
        "sampling": "Independent uniform whole-document swap masks, with replacement",
        "extreme_replicates": extremes,
        "extremeness_rule": "abs(delta_permuted) >= abs(delta_observed); exact rational comparison",
        "p_value": (extremes + 1) / (n + 1),
        "p_formula": "(extreme_replicates + 1) / (replicates + 1)",
        "monte_carlo_standard_error_approx": math.sqrt(raw * (1 - raw) / n),
        "multiple_comparisons": "Unadjusted; one focused retrospective BERT F1 comparison",
        "alpha": .05,
    }


def validate_statistical_code():
    # Exhaustive tiny problem validates the optimized count-based statistic
    # against direct swapping with Fraction arithmetic for every assignment.
    a = [(2, 5, 6), (4, 6, 7), (3, 5, 8)]
    b = [(4, 7, 6), (3, 5, 7), (5, 7, 8)]
    ca, cb = pooled_counts(a), pooled_counts(b)
    obs = abs(f1_fraction(cb) - f1_fraction(ca))
    direct_extremes = 0
    for mask in range(2 ** len(a)):
        aa, bb = swapped_counts(a, b, mask)
        direct = abs(f1_fraction(bb) - f1_fraction(aa)) >= obs
        da, db = aa[1] + aa[2], bb[1] + bb[2]
        exact = (abs(2 * (bb[0] * da - aa[0] * db)) * obs.denominator
                 >= obs.numerator * da * db)
        assert direct == exact
        direct_extremes += direct
    assert direct_extremes == 4, direct_extremes
    assert randomization(a, a, 100, 42)["p_value"] == 1.
    identical = paired_bootstrap(a, a, 100, 42)
    assert all(row == (0., 0., 0.) for row in identical)
    assert percentile([0., 10.], .25) == 2.5
    return {"exhaustive_toy_assignments": 8, "exhaustive_toy_extremes": direct_extremes,
            "identical_model_p_is_one": True, "identical_model_bootstrap_is_zero": True,
            "quantile_interpolation_checked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--team-archive", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    code_checks = validate_statistical_code()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = repo / "team_code/casie_eval.py"
    spec = importlib.util.spec_from_file_location("casie_eval_audit", source)
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)
    member_a = "predictions/test_bert-base-cased.jsonl"
    member_b = "predictions/test_bert-base-uncased.jsonl"
    gold_member = "dataset/CASIE_test.jsonl"
    bundle = repo / "data/TeamUltron_Project2_Inputs.zip"
    hashes, prediction_rows, records, documents = {}, {}, [], {}
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read("input_manifest.json"))
        for name in (member_a, member_b):
            data = z.read(name)
            hashes[name] = hashlib.sha256(data).hexdigest()
            assert hashes[name] == manifest["files"][name]
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
            ids = [r["event_uid"] for r in rows]
            assert len(ids) == len(set(ids)) == 736
            prediction_rows[name] = rows
            (out / Path(name).name).write_bytes(data)
        digest = hashlib.sha256()
        with z.open(gold_member) as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                rec = json.loads(line)
                did = rec["document_id"]
                if did in documents:
                    assert rec["tokens"] == documents[did]
                    rec["tokens"] = documents[did]
                else:
                    documents[did] = rec["tokens"]
                records.append(rec)
        hashes[gold_member] = digest.hexdigest()
        assert hashes[gold_member] == manifest["files"][gold_member]
    ids = {r["event_uid"] for r in records}
    assert len(records) == len(ids) == 736 and len(documents) == 100
    res = []
    for name in (member_a, member_b):
        predictions = ce.load_predictions(str(out / Path(name).name))
        assert set(predictions) == ids
        result = ce.evaluate(records, predictions, n_examples=0)
        assert not result["warnings"], result["warnings"]
        assert result["overall"]["gold_arguments"] == 2109
        res.append(result)
    docs = sorted(res[0]["_per_doc"])
    assert set(docs) == set(res[1]["_per_doc"]) == set(documents)
    a, b = ([r["_per_doc"][d] for d in docs] for r in res)
    for rows, result in zip((a, b), res):
        assert pooled_counts(rows) == tuple(result["overall"][k] for k in
                                           ("correct", "predicted_arguments", "gold_arguments"))
        assert abs(prf(pooled_counts(rows))[2] - result["overall"]["arg_c_f1"]) < 1e-15
    print("Verified 736 events, 100 documents, 2109 gold arguments.", flush=True)
    print("Cased:", res[0]["overall"], flush=True)
    print("Uncased:", res[1]["overall"], flush=True)
    boot = paired_bootstrap(a, b, 100000, 42)
    legacy = summarize_differences(boot[:5000])
    primary_boot = summarize_differences(boot)
    comparison = ce.compare_models(res[0], res[1], n=5000, seed=42)
    assert comparison["b_better_fraction"] == legacy["f1"]["uncased_better_fraction"]
    permutation = randomization(a, b, 100000, 42)
    # Compare archive bytes, README claim and recorded training settings, if supplied.
    archive_evidence = None
    if args.team_archive:
        with zipfile.ZipFile(args.team_archive) as z:
            base = "TeamUltron_Project2/"
            original_readme = z.read(base + "README.md").decode("utf-8-sig")
            assert "0.002" in original_readme
            configs = {}
            archive_hashes = {}
            for model, member in (("bert-base-cased", member_a), ("bert-base-uncased", member_b)):
                raw = z.read(base + member)
                archive_hashes[member] = hashlib.sha256(raw).hexdigest()
                assert archive_hashes[member] == hashes[member]
                configs[model] = json.loads(z.read(base + "runs/" + model + "/training_config.json"))["config"]
            archive_evidence = {"filename": args.team_archive.name,
                                "archive_sha256": sha256(args.team_archive),
                                "prediction_sha256": archive_hashes,
                                "training_config": configs,
                                "readme_claim": {"f1_difference": .034, "ci_95": [.013, .056], "p": .002,
                                                 "bootstrap_replicates": 5000},
                                "original_p_value_formula_and_generating_script": "Not identified in inspected package"}
    per_document = [{"document_id": d, "events": sum(r["document_id"] == d for r in records),
                     "cased": dict(zip(("correct", "predicted", "gold"), aa)),
                     "uncased": dict(zip(("correct", "predicted", "gold"), bb))}
                    for d, aa, bb in zip(docs, a, b)]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "Supplied BERT uncased minus supplied BERT cased; test split",
        "scope": "Retrospective audit of fixed supplied predictions; no inference or training",
        "sample": {"events": 736, "documents": 100, "gold_arguments": 2109},
        "scorer": "Repository casie_eval.evaluate; exact event UID, inclusive token span and role",
        "provenance": {"repository_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "scorer_sha256": sha256(source), "bundle_sha256": sha256(bundle),
            "member_sha256": hashes, "audit_script_sha256": sha256(__file__),
            "python": platform.python_version(), "random_generator": "Python random.Random (MT19937)"},
        "models": {"bert_cased": {"overall": res[0]["overall"], "warnings": res[0]["warnings"],
                                  "repository_f1_ci_1000_seed42": ce.bootstrap_f1(res[0])},
                   "bert_uncased": {"overall": res[1]["overall"], "warnings": res[1]["warnings"],
                                    "repository_f1_ci_1000_seed42": ce.bootstrap_f1(res[1])}},
        "observed_difference_uncased_minus_cased": dict(zip(("precision", "recall", "f1"),
            (y - x for x, y in zip(prf(pooled_counts(a)), prf(pooled_counts(b)))))),
        "paired_document_bootstrap": {"replicates": 100000, "seed": 42,
            "resampling": "Sample 100 document IDs with replacement; identical sampled IDs for both models",
            "statistic": "Difference of pooled corpus micro metrics, not average document metrics",
            "percentile_method": "Linear interpolation at (n-1)*q for q=.025,.975",
            "difference": primary_boot},
        "historical_5000_bootstrap_diagnostic": {"replicates": 5000, "seed": 42,
            "document_order": "Sorted document_id; matches repository compare_models",
            "difference": legacy, "repository_compare_models": comparison,
            "note": "Bootstrap tail fractions are diagnostics, not the primary randomization p-value"},
        "paired_document_randomization": permutation,
        "source_archive": archive_evidence,
        "validation": code_checks,
        "limitations": ["Conditional on these fitted models and this corrected test split; no training-seed uncertainty",
            "Whole-document resampling preserves within-document event dependence; assumes independent document clusters",
            "Cased and uncased use different pretrained models and vocabularies; not a controlled casing-only intervention",
            "Comparison is retrospective and unadjusted; not a preregistered or benchmark-wide multiple-comparison analysis",
            "Different statistical tests and random seeds need not reproduce the identical rounded historical p-value"],
        "method_references": ["https://aclanthology.org/P18-1128/", "https://aclanthology.org/W05-0908/"],
        "elapsed_seconds": time.perf_counter() - start,
    }
    (out / "per_document_counts.json").write_text(json.dumps(per_document, indent=2), encoding="utf-8")
    (out / "CASIE_BERT_Significance_Report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("OBSERVED", report["observed_difference_uncased_minus_cased"], flush=True)
    print("BOOTSTRAP", json.dumps(primary_boot, indent=2), flush=True)
    print("HISTORICAL_DIAGNOSTIC", json.dumps(legacy, indent=2), flush=True)
    print("RANDOMIZATION", json.dumps(permutation, indent=2), flush=True)
    print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
