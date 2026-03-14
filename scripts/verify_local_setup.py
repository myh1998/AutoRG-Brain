#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def must_exist(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local offline assets for AutoRG-Brain baseline run.")
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--seg-pretrained", required=True)
    parser.add_argument("--radgenome-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    model_folder = Path(args.model_folder).expanduser()
    seg_pretrained = Path(args.seg_pretrained).expanduser()
    radgenome_root = Path(args.radgenome_root).expanduser()
    split_json = Path(args.split_json).expanduser()
    output_root = Path(args.output_root).expanduser()

    must_exist(model_folder, "model folder")
    must_exist(seg_pretrained, "seg_pretrained model")
    must_exist(radgenome_root, "RadGenome root")
    must_exist(split_json, "split json")

    ckpt = model_folder / f"{args.checkpoint_name}.model"
    ckpt_pkl = model_folder / f"{args.checkpoint_name}.model.pkl"
    if not ckpt.exists() and not (model_folder / args.checkpoint_name).exists():
        raise FileNotFoundError(
            f"Cannot find checkpoint '{args.checkpoint_name}'. Expected {ckpt} or a matching subfolder in {model_folder}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "features_cache").mkdir(parents=True, exist_ok=True)

    split_obj = json.loads(split_json.read_text(encoding="utf-8"))
    num_tests = None
    if isinstance(split_obj, dict) and isinstance(split_obj.get("test"), list):
        num_tests = len(split_obj["test"])
    elif isinstance(split_obj, list):
        num_tests = len(split_obj)

    print("[OK] Local setup check passed")
    print(f"- model_folder: {model_folder}")
    print(f"- checkpoint_model_exists: {ckpt.exists()}")
    print(f"- checkpoint_pkl_exists: {ckpt_pkl.exists()}")
    print(f"- seg_pretrained: {seg_pretrained}")
    print(f"- radgenome_root: {radgenome_root}")
    print(f"- split_json: {split_json}")
    if num_tests is not None:
        print(f"- split_cases_count: {num_tests}")
    print(f"- output_root: {output_root}")


if __name__ == "__main__":
    main()
