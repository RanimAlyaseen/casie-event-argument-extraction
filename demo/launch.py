"""Launch the original team demo from any working directory.

Example: python demo/launch.py --checkpoint /path/to/roberta-base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CASIE argument extraction demo")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Directory with the fine-tuned model and tokenizer files")
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
                     "Supply a complete fine-tuned model directory; base roberta-base is not sufficient.")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "team_code"))
    import casie_demo as cd
    import casie_transformer as ctr

    model, tokenizer, encoder = ctr.load_trained(str(checkpoint), max_len=512)
    lexicon = cd.load_lexicon(str(root / "demo" / "trigger_lexicon.json"))
    print(f"Loaded model on {model.device}; {len(lexicon['phrases'])} trigger phrases.")
    cd.build_app(model, tokenizer, encoder, lexicon).launch(share=args.share)


if __name__ == "__main__":
    main()
