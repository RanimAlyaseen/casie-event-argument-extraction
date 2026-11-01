"""Launch the original team demo from any working directory.

Example: python demo/launch.py --checkpoint /path/to/deberta_v3_base --model-label DeBERTa-v3-base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CASIE argument extraction demo")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Directory with the fine-tuned model and tokenizer files")
    parser.add_argument("--model-label",
                        help="Displayed model name (defaults to the checkpoint folder name)")
    parser.add_argument("--lexicon", type=Path,
                        help="Trigger lexicon JSON (defaults to demo/trigger_lexicon.json)")
    parser.add_argument("--share", action="store_true",
                        help="Create a temporary public Gradio link")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    has_weights = any((checkpoint / name).is_file() for name in
                      ("model.safetensors", "pytorch_model.bin",
                       "model.safetensors.index.json", "pytorch_model.bin.index.json"))
    if not checkpoint.is_dir() or missing or not has_weights:
        parser.error(f"checkpoint directory {checkpoint} is unavailable, missing {missing}, "
                     f"or has no model weights (weights present: {has_weights}). "
                     "Supply a complete fine-tuned model directory; a base pretrained model is not sufficient.")

    try:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(f"Cannot read checkpoint config.json: {error}")
    labels = set(config.get("id2label", {}).values())
    if (checkpoint / "role_inventory.json").is_file() or labels & {"B-Target", "I-Target"}:
        parser.error("Role-conditioned checkpoints require team_code/role_prime_experiment.py; "
                     "this demo accepts standard argument-role BIO checkpoints.")

    root = Path(__file__).resolve().parents[1]
    lexicon_path = (args.lexicon.expanduser().resolve() if args.lexicon
                    else root / "demo" / "trigger_lexicon.json")
    if not lexicon_path.is_file():
        parser.error(f"trigger lexicon is unavailable: {lexicon_path}")
    sys.path.insert(0, str(root / "team_code"))
    import casie_demo as cd
    import casie_transformer as ctr

    model, tokenizer, encoder = ctr.load_trained(str(checkpoint), max_len=512)
    lexicon = cd.load_lexicon(str(lexicon_path))
    print(f"Loaded model on {model.device}; {len(lexicon['phrases'])} trigger phrases.")
    model_label = (args.model_label or "").strip() or checkpoint.name
    cd.build_app(model, tokenizer, encoder, lexicon,
                 model_label=model_label).launch(share=args.share)


if __name__ == "__main__":
    main()
