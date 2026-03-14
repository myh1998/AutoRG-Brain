#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from AutoRG_Brain.utilities.llm_metric import compute_language_model_scores


def _extract_pred(item: List[dict], pred_field: str) -> str:
    if not item:
        return ""
    head = item[0]
    if pred_field in head:
        payload = head[pred_field]
        if isinstance(payload, dict):
            return str(payload.get("report", ""))
        return str(payload)

    # fallback priority
    for key in ("pred_global_report", "pred_region_concat", "pred_report"):
        if key in head:
            payload = head[key]
            if isinstance(payload, dict):
                return str(payload.get("report", ""))
            return str(payload)
    return ""


def _load_refs(test_file: Path) -> Dict[str, str]:
    tests = json.loads(test_file.read_text(encoding="utf-8"))
    refs: Dict[str, str] = {}
    for row in tests:
        image = str(row.get("image", ""))
        ident = Path(image).name.split(".")[0]
        report = row.get("report")
        if isinstance(report, dict):
            refs[ident] = " ".join(report.keys())
        elif isinstance(report, str):
            refs[ident] = report
    return refs


def _run_optional_metric(metric_name: str, command: str, pred_json: Path, test_json: Path) -> Tuple[bool, str]:
    env = os.environ.copy()
    env["PRED_JSON"] = str(pred_json)
    env["TEST_JSON"] = str(test_json)
    try:
        out = subprocess.check_output(command, shell=True, env=env, text=True, stderr=subprocess.STDOUT)
        return True, out.strip()
    except subprocess.CalledProcessError as e:
        return False, e.output.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AutoRG outputs for local Table-2 style reporting.")
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--pred-field", default="pred_global_report")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--radgraph-cmd", default="")
    parser.add_argument("--ratescore-cmd", default="")
    parser.add_argument("--radcliq-cmd", default="")
    args = parser.parse_args()

    pred_data = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
    refs = _load_refs(Path(args.test_json))

    ys_pred: List[str] = []
    ys_ref: List[str] = []
    for ident, block in pred_data.items():
        if ident == "avg":
            continue
        if ident not in refs:
            continue
        ys_pred.append(_extract_pred(block, args.pred_field))
        ys_ref.append(refs[ident])

    if not ys_pred:
        raise RuntimeError("No overlapping prediction/reference pairs found. Check identifier mapping.")

    rouges, bleus = compute_language_model_scores(ys_pred, ys_ref)
    result: Dict[str, object] = {
        "num_pairs": len(ys_pred),
        "bleu1_avg": sum(bleus) / len(bleus),
        "rouge1_recall_avg": sum(rouges) / len(rouges),
    }

    for metric_name, cmd in (
        ("radgraph", args.radgraph_cmd),
        ("ratescore", args.ratescore_cmd),
        ("radcliq", args.radcliq_cmd),
    ):
        if cmd:
            ok, out = _run_optional_metric(metric_name, cmd, Path(args.pred_json), Path(args.test_json))
            result[metric_name] = {"ok": ok, "raw": out}
        else:
            result[metric_name] = {
                "ok": False,
                "raw": "not_configured",
                "hint": f"Pass --{metric_name}-cmd '<your offline evaluator command>'",
            }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
