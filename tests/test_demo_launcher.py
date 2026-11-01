"""Exercise launcher dispatch without loading weights or opening a server."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_launcher", ROOT / "demo/launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def run_launcher(self, model_label=None, custom_lexicon=False, role_conditioned=False):
        with tempfile.TemporaryDirectory(prefix="casie_launcher_test_") as directory:
            checkpoint = Path(directory) / "deberta_v3_base"
            checkpoint.mkdir()
            labels = {"0": "O", "1": "B-Target" if role_conditioned else "B-Victim",
                      "2": "I-Target" if role_conditioned else "I-Victim"}
            (checkpoint / "config.json").write_text(json.dumps({"id2label": labels}))
            (checkpoint / "tokenizer_config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"test-stub-not-real-weights")
            lexicon = Path(directory) / "custom_lexicon.json"
            lexicon.write_text('{"phrases": {"breached": {"subtype": "Databreach"}}}')
            calls = {}
            model = types.SimpleNamespace(device="cpu")

            def load_trained(path, max_len):
                calls["checkpoint"] = path
                calls["max_len"] = max_len
                return model, "tokenizer-stub", "encoder-stub"

            def load_lexicon(path):
                calls["lexicon"] = path
                return {"phrases": {"breached": {"subtype": "Databreach"}}}

            def build_app(*args, **kwargs):
                calls["app_args"] = args
                calls["model_label"] = kwargs["model_label"]
                return types.SimpleNamespace(launch=lambda **options: calls.update(launch=options))

            argv = ["launch.py", "--checkpoint", str(checkpoint)]
            if model_label is not None:
                argv += ["--model-label", model_label]
            if custom_lexicon:
                argv += ["--lexicon", str(lexicon)]
            ctr = types.SimpleNamespace(load_trained=load_trained)
            cd = types.SimpleNamespace(load_lexicon=load_lexicon, build_app=build_app)
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {
                    "casie_transformer": ctr, "casie_demo": cd}), redirect_stdout(io.StringIO()):
                launcher.main()
            return calls, str(checkpoint.resolve()), str(lexicon.resolve())

    def test_deberta_folder_name_reaches_ui_instead_of_roberta(self):
        calls, checkpoint, _ = self.run_launcher()
        self.assertEqual(calls["model_label"], "deberta_v3_base")
        self.assertEqual(calls["checkpoint"], checkpoint)
        self.assertEqual(calls["max_len"], 512)
        self.assertEqual(calls["launch"], {"share": False})
        self.assertEqual(calls["lexicon"], str(ROOT / "demo/trigger_lexicon.json"))

    def test_explicit_label_and_separate_lexicon_reach_app(self):
        calls, _, lexicon = self.run_launcher(" DeBERTa-v3-base ", custom_lexicon=True)
        self.assertEqual(calls["model_label"], "DeBERTa-v3-base")
        self.assertEqual(calls["lexicon"], lexicon)

    def test_blank_label_uses_checkpoint_folder(self):
        calls, _, _ = self.run_launcher("   ")
        self.assertEqual(calls["model_label"], "deberta_v3_base")

    def test_role_conditioned_checkpoint_stops_before_inference(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            self.run_launcher(role_conditioned=True)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("role_prime_experiment.py", stderr.getvalue())

    def test_missing_weights_produce_actionable_error(self):
        with tempfile.TemporaryDirectory(prefix="casie_missing_checkpoint_") as directory:
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["launch.py", "--checkpoint", directory]), \
                    redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                launcher.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("complete fine-tuned model directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
