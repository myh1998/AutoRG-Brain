#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


MODAL_HINTS = {
    "t1": "T1WI",
    "t1w": "T1WI",
    "t1wi": "T1WI",
    "t2": "T2WI",
    "t2w": "T2WI",
    "t2wi": "T2WI",
    "flair": "T2FLAIR",
    "t2flair": "T2FLAIR",
    "dwi": "DWI",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _guess_modal(image_path: str) -> str:
    lower = image_path.lower()
    for key, modal in MODAL_HINTS.items():
        if key in lower:
            return modal
    return "T2FLAIR"


def _normalize_case(raw: Dict[str, Any], root: Path, default_modal: Optional[str]) -> Optional[Dict[str, Any]]:
    image = raw.get("image") or raw.get("img") or raw.get("image_path")
    if image is None and raw.get("id"):
        cid = str(raw["id"])
        image = root / "images" / f"{cid}.nii.gz"
    if image is None:
        return None

    image = str(image)
    if not Path(image).is_absolute():
        image = str((root / image).resolve())

    label = raw.get("label") or raw.get("ab_label") or raw.get("mask")
    if label is None and raw.get("id"):
        cand = root / "labels" / f"{raw['id']}.nii.gz"
        if cand.exists():
            label = str(cand)
    elif label is not None and not Path(str(label)).is_absolute():
        label = str((root / str(label)).resolve())

    label2 = raw.get("label2") or raw.get("ana_label")
    if label2 is not None and not Path(str(label2)).is_absolute():
        label2 = str((root / str(label2)).resolve())

    modal = raw.get("modal") or default_modal or _guess_modal(image)

    out: Dict[str, Any] = {
        "image": image,
        "modal": modal,
    }
    if label is not None:
        out["label"] = str(label)
    if label2 is not None:
        out["label2"] = str(label2)

    report = raw.get("report")
    if report is not None:
        out["report"] = report

    return out


def _collect_test_cases(split_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(split_obj, list):
        if split_obj and isinstance(split_obj[0], dict):
            return split_obj
        return [{"id": x} for x in split_obj]

    if isinstance(split_obj, dict):
        if "test" in split_obj and isinstance(split_obj["test"], list):
            tests = split_obj["test"]
            if tests and isinstance(tests[0], dict):
                return tests
            return [{"id": x} for x in tests]

        if "test_file" in split_obj and isinstance(split_obj["test_file"], list):
            return split_obj["test_file"]

    raise ValueError("Unsupported split JSON format. Expected list or dict with `test` list.")


def _index_case_metadata(case_meta: Any) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if isinstance(case_meta, list):
        for item in case_meta:
            cid = str(item.get("id") or item.get("case_id") or item.get("image") or "")
            if cid:
                idx[cid] = item
    elif isinstance(case_meta, dict):
        for k, v in case_meta.items():
            if isinstance(v, dict):
                idx[str(k)] = v
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AutoRG test_file.json from local RadGenome metadata.")
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--radgenome-root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--case-metadata-json", default=None)
    parser.add_argument("--default-modal", default=None)
    parser.add_argument("--require-report", action="store_true")
    args = parser.parse_args()

    root = Path(args.radgenome_root).expanduser().resolve()
    split_obj = _load_json(Path(args.split_json).expanduser())
    raw_tests = _collect_test_cases(split_obj)

    case_idx: Dict[str, Dict[str, Any]] = {}
    if args.case_metadata_json:
        case_idx = _index_case_metadata(_load_json(Path(args.case_metadata_json).expanduser()))

    out_cases: List[Dict[str, Any]] = []
    missing_report = 0

    for item in raw_tests:
        candidate = dict(item)
        cid = str(candidate.get("id") or candidate.get("case_id") or "")
        if cid and cid in case_idx:
            merged = dict(case_idx[cid])
            merged.update(candidate)
            candidate = merged

        normalized = _normalize_case(candidate, root, args.default_modal)
        if normalized is None:
            continue

        if args.require_report and "report" not in normalized:
            missing_report += 1
            continue

        out_cases.append(normalized)

    out_path = Path(args.out_json).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_cases, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(out_cases)} test entries -> {out_path}")
    if missing_report:
        print(f"Skipped {missing_report} entries without report because --require-report is set")


if __name__ == "__main__":
    main()
