from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from petct_seg_data import load_volume
from petct_seg_metrics import summarize_metric


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    path: Path
    mask_voxels: int


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run k-fold cross validation for PET-CT segmentation.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory of processed case folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root directory for folds, checkpoints and summaries.")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction taken from the non-test pool.")
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument(
        "--stratify-by",
        type=str,
        default="mask_voxels",
        choices=("mask_voxels", "none"),
        help="Simple stratification key used when building folds.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="auto",
        choices=("auto", "symlink", "copy"),
        help="How fold train/val/test directories should materialize cases.",
    )
    parser.add_argument("--force-recreate-splits", action="store_true", help="Recreate split directories even if they exist.")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare fold directories and manifests.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and reuse existing checkpoints.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and reuse existing eval JSON files.")
    parser.add_argument("--save-preds", action="store_true", help="Save test predictions for each fold.")
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--eval-device", type=str, default=None, help="Optional device override passed to evaluate.py.")
    parser.add_argument(
        "--fold-indices",
        type=str,
        default=None,
        help="Comma-separated 1-based fold indices to run. Defaults to all folds.",
    )
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--train-script", type=Path, default=Path("train.py"))
    parser.add_argument("--eval-script", type=Path, default=Path("evaluate.py"))
    return parser


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = build_argparser()
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def parse_fold_indices(raw: str | None, num_folds: int) -> list[int]:
    if raw is None:
        return list(range(num_folds))

    indices: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        fold_index = int(item) - 1
        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(f"Fold index out of range: {item}. Expected 1..{num_folds}.")
        indices.append(fold_index)

    deduped = sorted(set(indices))
    if not deduped:
        raise ValueError("No valid fold indices were provided.")
    return deduped


def load_case_records(dataset_dir: Path, stratify_by: str) -> list[CaseRecord]:
    case_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No case directories found under {dataset_dir}.")

    case_records: list[CaseRecord] = []
    for case_dir in case_dirs:
        mask_voxels = 0
        if stratify_by == "mask_voxels":
            metadata_path = case_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                raw_voxels = metadata.get("nonzero_mask_voxels")
                if raw_voxels is not None:
                    mask_voxels = int(raw_voxels)
            if mask_voxels <= 0:
                mask = load_volume(case_dir / "mask.npy")
                mask_voxels = int((mask > 0).sum().item())
        case_records.append(CaseRecord(case_id=case_dir.name, path=case_dir.resolve(), mask_voxels=mask_voxels))
    return case_records


def _ordered_records(case_records: list[CaseRecord], seed: int, stratify_by: str) -> list[CaseRecord]:
    rng = random.Random(seed)
    ordered = list(case_records)
    rng.shuffle(ordered)
    if stratify_by == "mask_voxels":
        ordered.sort(key=lambda record: record.mask_voxels, reverse=True)
    return ordered


def build_outer_folds(case_records: list[CaseRecord], num_folds: int, seed: int, stratify_by: str) -> list[list[CaseRecord]]:
    ordered = _ordered_records(case_records, seed=seed, stratify_by=stratify_by)
    rng = random.Random(seed)
    folds: list[list[CaseRecord]] = [[] for _ in range(num_folds)]

    for start in range(0, len(ordered), num_folds):
        bucket = ordered[start : start + num_folds]
        fold_order = list(range(num_folds))
        rng.shuffle(fold_order)
        for record, fold_index in zip(bucket, fold_order):
            folds[fold_index].append(record)

    for fold in folds:
        fold.sort(key=lambda record: record.case_id)
    return folds


def _split_evenly(items: list[CaseRecord], num_groups: int) -> list[list[CaseRecord]]:
    groups: list[list[CaseRecord]] = []
    start = 0
    total = len(items)
    for group_index in range(num_groups):
        stop = start + (total - start + num_groups - group_index - 1) // (num_groups - group_index)
        groups.append(items[start:stop])
        start = stop
    return groups


def select_validation_subset(
    case_records: list[CaseRecord],
    val_fraction: float,
    seed: int,
    stratify_by: str,
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    if len(case_records) < 2:
        raise ValueError("Need at least 2 cases in the non-test pool to create train/val splits.")

    val_count = max(1, int(math.ceil(len(case_records) * val_fraction)))
    val_count = min(val_count, len(case_records) - 1)

    ordered = _ordered_records(case_records, seed=seed, stratify_by=stratify_by)
    buckets = _split_evenly(ordered, val_count)
    rng = random.Random(seed)

    selected_ids: set[str] = set()
    for bucket in buckets:
        if not bucket:
            continue
        chosen = bucket[rng.randrange(len(bucket))]
        selected_ids.add(chosen.case_id)

    val_cases = sorted((record for record in case_records if record.case_id in selected_ids), key=lambda record: record.case_id)
    train_cases = sorted(
        (record for record in case_records if record.case_id not in selected_ids),
        key=lambda record: record.case_id,
    )

    if not train_cases or not val_cases:
        raise ValueError("Failed to create non-empty train/val splits.")
    return train_cases, val_cases


def build_fold_assignments(
    case_records: list[CaseRecord],
    num_folds: int,
    val_fraction: float,
    seed: int,
    stratify_by: str,
) -> list[dict[str, list[CaseRecord]]]:
    outer_folds = build_outer_folds(case_records, num_folds=num_folds, seed=seed, stratify_by=stratify_by)
    assignments: list[dict[str, list[CaseRecord]]] = []

    for fold_index in range(num_folds):
        test_cases = outer_folds[fold_index]
        remaining = [record for idx, fold in enumerate(outer_folds) if idx != fold_index for record in fold]
        train_cases, val_cases = select_validation_subset(
            remaining,
            val_fraction=val_fraction,
            seed=seed + fold_index + 1,
            stratify_by=stratify_by,
        )
        assignments.append({"train": train_cases, "val": val_cases, "test": list(test_cases)})
    return assignments


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _resolve_split_mode(mode: str) -> str:
    if mode == "auto":
        return "symlink" if os.name != "nt" else "copy"
    return mode


def _materialize_case(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return

    resolved_mode = _resolve_split_mode(mode)
    if resolved_mode == "symlink":
        try:
            dst.symlink_to(src, target_is_directory=True)
            return
        except OSError:
            if mode != "auto":
                raise
            resolved_mode = "copy"

    if resolved_mode == "copy":
        shutil.copytree(src, dst)
        return

    raise ValueError(f"Unsupported split mode: {mode}")


def prepare_fold_split(
    fold_dir: Path,
    assignment: dict[str, list[CaseRecord]],
    split_mode: str,
    force_recreate: bool,
) -> dict[str, object]:
    split_root = fold_dir / "split"
    if force_recreate:
        _remove_path(split_root)

    split_root.mkdir(parents=True, exist_ok=True)
    manifest = {"fold_dir": str(fold_dir), "split_root": str(split_root), "splits": {}}

    for split_name, records in assignment.items():
        split_dir = split_root / split_name
        if force_recreate:
            _remove_path(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            _materialize_case(record.path, split_dir / record.case_id, split_mode)

        manifest["splits"][split_name] = {
            "num_cases": len(records),
            "case_ids": [record.case_id for record in records],
        }

    manifest_path = fold_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _record_to_json(record: CaseRecord) -> dict[str, object]:
    return {"case_id": record.case_id, "path": str(record.path), "mask_voxels": record.mask_voxels}


def save_assignment_manifest(
    output_dir: Path,
    assignments: list[dict[str, list[CaseRecord]]],
    *,
    dataset_dir: Path,
    num_folds: int,
    val_fraction: float,
    seed: int,
    stratify_by: str,
    split_mode: str,
) -> None:
    payload = {
        "dataset_dir": str(dataset_dir),
        "num_folds": num_folds,
        "val_fraction": val_fraction,
        "seed": seed,
        "stratify_by": stratify_by,
        "split_mode": split_mode,
        "folds": [],
    }
    for fold_index, assignment in enumerate(assignments, start=1):
        payload["folds"].append(
            {
                "fold_index": fold_index,
                "train": [_record_to_json(record) for record in assignment["train"]],
                "val": [_record_to_json(record) for record in assignment["val"]],
                "test": [_record_to_json(record) for record in assignment["test"]],
            }
        )
    (output_dir / "crossval_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_command(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str], cwd: Path) -> None:
    print(f"Running: {_format_command(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def validate_passthrough_args(args: list[str]) -> None:
    reserved = {"--train-dir", "--val-dir", "--save-dir"}
    conflicts = [arg for arg in args if arg in reserved]
    if conflicts:
        raise ValueError(f"Do not pass {conflicts} to crossval.py; they are managed per fold.")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_fold_result(fold_dir: Path, fold_index: int) -> dict[str, object]:
    if not fold_dir.exists():
        return {}

    result: dict[str, object] = {"fold_index": fold_index}
    has_content = False

    split_manifest_path = fold_dir / "split_manifest.json"
    if split_manifest_path.exists():
        has_content = True
        split_manifest = load_json(split_manifest_path)
        result["num_train"] = split_manifest["splits"]["train"]["num_cases"]
        result["num_val"] = split_manifest["splits"]["val"]["num_cases"]
        result["num_test"] = split_manifest["splits"]["test"]["num_cases"]

    best_ckpt_path = fold_dir / "checkpoints" / "best.pt"
    if best_ckpt_path.exists():
        has_content = True
        import torch

        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        result["best_epoch"] = best_ckpt.get("epoch")
        result["val_dice"] = best_ckpt.get("val_dice")
        result["val_hd95"] = best_ckpt.get("val_hd95")
        result["val_hd95_invalid"] = best_ckpt.get("val_hd95_invalid")
        result["loss_name"] = best_ckpt.get("loss_name")
        result["loss_kwargs"] = best_ckpt.get("loss_kwargs")
        result["model_kwargs"] = best_ckpt.get("model_kwargs")

    eval_json_path = fold_dir / "eval" / "results.json"
    if eval_json_path.exists():
        has_content = True
        evaluation = load_json(eval_json_path)
        summary = evaluation.get("summary", {})
        dice_summary = summary.get("dice", {})
        hd95_summary = summary.get("hd95", {})
        result["test_dice_mean"] = dice_summary.get("mean")
        result["test_dice_std"] = dice_summary.get("std")
        result["test_hd95_mean"] = hd95_summary.get("mean")
        result["test_hd95_std"] = hd95_summary.get("std")
        result["test_hd95_invalid"] = hd95_summary.get("num_invalid")
        result["eval_json"] = str(eval_json_path)
    return result if has_content else {}


def build_summary(output_dir: Path, num_folds: int) -> dict[str, object]:
    fold_results: list[dict[str, object]] = []
    pooled_dice: list[float] = []
    pooled_hd95: list[float] = []
    pooled_hd95_invalid = 0

    for fold_index in range(1, num_folds + 1):
        fold_dir = output_dir / f"fold_{fold_index:02d}"
        fold_result = collect_fold_result(fold_dir, fold_index)
        if fold_result:
            fold_results.append(fold_result)

        eval_json_path = fold_dir / "eval" / "results.json"
        if eval_json_path.exists():
            evaluation = load_json(eval_json_path)
            for case in evaluation.get("cases", []):
                if not isinstance(case, dict):
                    continue
                dice = case.get("dice")
                hd95 = case.get("hd95")
                if dice is not None:
                    pooled_dice.append(float(dice))
                if hd95 is None:
                    pooled_hd95_invalid += 1
                else:
                    pooled_hd95.append(float(hd95))

    fold_test_dice = [float(result["test_dice_mean"]) for result in fold_results if result.get("test_dice_mean") is not None]
    fold_test_hd95 = [float(result["test_hd95_mean"]) for result in fold_results if result.get("test_hd95_mean") is not None]
    fold_val_dice = [float(result["val_dice"]) for result in fold_results if result.get("val_dice") is not None]
    fold_val_hd95 = [float(result["val_hd95"]) for result in fold_results if result.get("val_hd95") is not None]

    summary = {
        "num_completed_folds": len(fold_test_dice),
        "folds": fold_results,
        "aggregate": {
            "fold_test_dice": summarize_metric(fold_test_dice),
            "fold_test_hd95": summarize_metric(fold_test_hd95),
            "fold_val_dice": summarize_metric(fold_val_dice),
            "fold_val_hd95": summarize_metric(fold_val_hd95),
            "pooled_case_dice": summarize_metric(pooled_dice),
            "pooled_case_hd95": {
                **summarize_metric(pooled_hd95),
                "num_invalid": pooled_hd95_invalid,
            },
        },
    }
    return summary


def main() -> None:
    args, train_args = parse_args()
    validate_passthrough_args(train_args)

    if args.num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    project_root = Path(__file__).resolve().parent
    train_script = (project_root / args.train_script).resolve() if not args.train_script.is_absolute() else args.train_script
    eval_script = (project_root / args.eval_script).resolve() if not args.eval_script.is_absolute() else args.eval_script

    case_records = load_case_records(args.dataset_dir, stratify_by=args.stratify_by)
    if len(case_records) < args.num_folds:
        raise ValueError(f"Need at least {args.num_folds} cases, found {len(case_records)}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments = build_fold_assignments(
        case_records,
        num_folds=args.num_folds,
        val_fraction=args.val_fraction,
        seed=args.seed,
        stratify_by=args.stratify_by,
    )
    save_assignment_manifest(
        args.output_dir,
        assignments,
        dataset_dir=args.dataset_dir,
        num_folds=args.num_folds,
        val_fraction=args.val_fraction,
        seed=args.seed,
        stratify_by=args.stratify_by,
        split_mode=args.split_mode,
    )

    selected_folds = parse_fold_indices(args.fold_indices, args.num_folds)
    for fold_index in selected_folds:
        fold_dir = args.output_dir / f"fold_{fold_index + 1:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        prepare_fold_split(
            fold_dir,
            assignments[fold_index],
            split_mode=args.split_mode,
            force_recreate=args.force_recreate_splits,
        )

        if args.prepare_only:
            continue

        split_root = fold_dir / "split"
        checkpoint_dir = fold_dir / "checkpoints"
        eval_dir = fold_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        if not args.skip_train:
            train_command = [
                args.python_executable,
                str(train_script),
                "--train-dir",
                str(split_root / "train"),
                "--val-dir",
                str(split_root / "val"),
                "--save-dir",
                str(checkpoint_dir),
                *train_args,
            ]
            run_command(train_command, cwd=project_root)

        best_checkpoint = checkpoint_dir / "best.pt"
        if not args.skip_eval:
            if not best_checkpoint.exists():
                raise FileNotFoundError(f"Expected checkpoint at {best_checkpoint} before evaluation.")
            eval_command = [
                args.python_executable,
                str(eval_script),
                "--checkpoint",
                str(best_checkpoint),
                "--dataset-dir",
                str(split_root / "test"),
                "--threshold",
                str(args.eval_threshold),
                "--output-json",
                str(eval_dir / "results.json"),
            ]
            if args.eval_device is not None:
                eval_command.extend(["--device", args.eval_device])
            if args.save_preds:
                eval_command.extend(["--save-preds-dir", str(fold_dir / "preds")])
            run_command(eval_command, cwd=project_root)

    summary = build_summary(args.output_dir, args.num_folds)
    summary.update(
        {
            "dataset_dir": str(args.dataset_dir),
            "output_dir": str(args.output_dir),
            "num_folds": args.num_folds,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "stratify_by": args.stratify_by,
            "split_mode": args.split_mode,
            "train_args": train_args,
        }
    )
    summary_path = args.output_dir / "crossval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    aggregate = summary["aggregate"]
    fold_dice = aggregate["fold_test_dice"]
    fold_hd95 = aggregate["fold_test_hd95"]
    print()
    print("Cross-validation summary")
    if fold_dice["mean"] is None:
        print("Test Dice: no completed fold evaluations yet")
    else:
        print(
            "Test Dice:",
            f"mean={fold_dice['mean']:.4f}",
            f"std={fold_dice['std']:.4f}",
            f"min={fold_dice['min']:.4f}",
            f"max={fold_dice['max']:.4f}",
        )
    if fold_hd95["mean"] is None:
        print("Test HD95: no completed fold evaluations yet")
    else:
        print(
            "Test HD95:",
            f"mean={fold_hd95['mean']:.4f}",
            f"std={fold_hd95['std']:.4f}",
            f"min={fold_hd95['min']:.4f}",
            f"max={fold_hd95['max']:.4f}",
        )
    print(f"Saved cross-validation summary to {summary_path}")


if __name__ == "__main__":
    main()
