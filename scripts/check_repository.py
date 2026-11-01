"""Check submission structure, evidence and checksums without inference/training.

Python standard library only. This validates PNG structure but does not
replace visual inspection of figures or slides.
"""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
from urllib.parse import unquote
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_png(path):
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, seen = 8, []
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        kind = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", raw[pos + 8 + length:pos + 12 + length])[0]
        assert (zlib.crc32(kind + data) & 0xffffffff) == crc, path
        seen.append(kind)
        if kind == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
            assert width > 0 and height > 0
        pos += 12 + length
    assert pos == len(raw) and seen[0] == b"IHDR" and seen[-1] == b"IEND", path


def main():
    manifest = json.loads((ROOT / "results/experiment_manifest.json").read_text())
    for path in (manifest["notebook"], manifest["input_bundle"]):
        assert (ROOT / path).is_file(), path
    assert sha(ROOT / manifest["input_bundle"]) == manifest["data_sha256"]
    assert manifest["selected_by_validation"] == "Supplied RoBERTa"

    with zipfile.ZipFile(ROOT / manifest["input_bundle"]) as z:
        inputs = json.loads(z.read("input_manifest.json"))
        actual = {n for n in z.namelist() if not n.endswith("/")}
        assert actual == set(inputs["files"]) | {"input_manifest.json"}
        for name, expected in inputs["files"].items():
            assert not name.startswith("/") and ".." not in Path(name).parts
            digest = hashlib.sha256()
            with z.open(name) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            assert digest.hexdigest() == expected, name
    print("PASS: input ZIP inventory and every manifest member checksum")

    for relative, count in (("results/validation_summary.csv", 8), ("results/test_summary.csv", 9)):
        with (ROOT / relative).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count and len({r["method"] for r in rows}) == count, relative
        for row in rows:
            p, r, f = (float(row[k]) for k in ("Arg-C P", "Arg-C R", "Arg-C F1"))
            assert 0 <= p <= 1 and 0 <= r <= 1 and 0 <= f <= 1
            assert abs(2 * p * r / (p + r) - f) < 2e-6
            low, high = ast.literal_eval(row["95% document bootstrap CI"])
            assert 0 <= low <= f <= high <= 1
    print("PASS: readable 8-row validation and 9-row test tables, metrics and intervals")

    submission = json.loads((ROOT / manifest["notebook"]).read_text(encoding="utf-8"))
    outputs = sum(bool(c.get("outputs")) for c in submission["cells"])
    assert outputs > 0, "Submission has lost executed outputs"
    assert not any(o.get("output_type") == "error" for c in submission["cells"] for o in c.get("outputs", []))
    verification = json.loads((ROOT / "notebooks/CASIE_Colab_DeBERTa_Verification.ipynb").read_text())
    for i, cell in enumerate(verification["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"colab-cell-{i}", "exec")
            assert not cell.get("outputs"), "GPU notebook must not imply a completed GPU run"
    print(f"PASS: notebook JSON; submission retains {outputs} output-bearing cells; GPU notebook syntax")

    for path in list((ROOT / "team_code").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + [ROOT / "demo/launch.py"]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    for path in (ROOT / "figures").glob("*.png"):
        check_png(path)
    print("PASS: source compilation and figure PNG integrity")

    audit = json.loads((ROOT / "results/verification/bert_significance.json").read_text())
    docs = json.loads((ROOT / "results/verification/bert_document_counts.json").read_text())
    assert len(docs) == 100 and sum(d["events"] for d in docs) == 736
    assert sha(ROOT / "scripts/verify_bert_significance.py") == audit["provenance"]["audit_script_sha256"]
    assert sha(ROOT / "team_code/casie_eval.py") == audit["provenance"]["scorer_sha256"]
    for short, label in (("cased", "bert_cased"), ("uncased", "bert_uncased")):
        counts = tuple(sum(d[short][k] for d in docs) for k in ("correct", "predicted", "gold"))
        target = audit["models"][label]["overall"]
        assert counts == tuple(target[k] for k in ("correct", "predicted_arguments", "gold_arguments"))
    test = audit["paired_document_randomization"]
    assert test["p_value"] == (test["extreme_replicates"] + 1) / (test["replicates"] + 1)
    assert test["p_value"] == manifest["subsequent_verification"]["bert_significance"]["randomization_p"]
    spec = importlib.util.spec_from_file_location("statistical_audit", ROOT / "scripts/verify_bert_significance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_statistical_code()
    checkpoint = json.loads((ROOT / "results/verification/checkpoint_cpu_verification.json").read_text())
    assert checkpoint["models"]["deberta_v3_base"]["gpu_verification_status"] == "pending"
    for model in checkpoint["models"].values():
        assert model["cpu_overall"]["events"] == 736
        assert not model["scorer_warnings"]
        o = model["cpu_overall"]
        assert abs(2 * o["correct"] / (o["predicted_arguments"] + o["gold_arguments"]) - o["arg_c_f1"]) < 1e-15
    print("PASS: audit provenance, pooled counts, statistical code checks and CPU report consistency")

    candidates = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()
    secret = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,})")
    for relative in candidates:
        path = ROOT / relative
        assert not relative.startswith(("runs/", ".venv/", "venv/", "uploads/")), relative
        assert path.suffix.lower() not in (".safetensors", ".pt", ".pth", ".bin", ".joblib", ".ckpt", ".pyc"), relative
        assert path.suffix.lower() != ".zip" or relative == manifest["input_bundle"], relative
        if path.suffix in (".md", ".py", ".json", ".ipynb"):
            text = path.read_text(encoding="utf-8")
            assert not secret.search(text), f"Possible secret in {relative}"
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                target = target.strip("<>").split("#", 1)[0]
                if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue
                assert (path.parent / unquote(target)).exists(), (relative, target)
    print("PASS: relative Markdown links, candidate-file exclusions and credential-pattern scan")


if __name__ == "__main__":
    main()
