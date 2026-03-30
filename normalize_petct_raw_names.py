from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


TARGET_FILENAMES = {
    "ct": "ct.nii.gz",
    "pet": "pet.nii.gz",
    "ct_seg": "ct_seg.nii.gz",
    "pet_seg": "pet_seg.nii.gz",
}

SOURCE_PATTERNS = {
    "ct": ("{case_id}_CT.nii.gz", "{case_id}_CT.nii"),
    "pet": ("{case_id}_PET.nii.gz", "{case_id}_PET.nii"),
    "ct_seg": ("{case_id}-CT-S.nii.gz", "{case_id}-CT-S.nii"),
    "pet_seg": ("{case_id}-PET-S.nii.gz", "{case_id}-PET-S.nii"),
}

REQUIRED_KEYS = ("ct", "pet")
OPTIONAL_KEYS = ("ct_seg", "pet_seg")


@dataclass(frozen=True)
class Operation:
    source: Path
    destination: Path
    mode: str


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize PET/CT raw case filenames to the convention expected by "
            "preprocess_pet_ct_dataset.py."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory of raw case folders.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. If omitted, files are renamed in place.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing target files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned operations without changing files.")
    return parser


def find_source_file(case_dir: Path, case_id: str, logical_name: str) -> Path | None:
    candidates = [case_dir / pattern.format(case_id=case_id) for pattern in SOURCE_PATTERNS[logical_name]]
    candidates.append(case_dir / TARGET_FILENAMES[logical_name])

    for path in candidates:
        if path.exists():
            return path
    return None


def build_case_operations(
    case_dir: Path,
    output_root: Path | None,
    overwrite: bool,
) -> tuple[list[Operation], list[str]]:
    case_id = case_dir.name
    destination_case_dir = (output_root / case_id) if output_root is not None else case_dir
    operations: list[Operation] = []
    missing_optional: list[str] = []

    for logical_name in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
        source = find_source_file(case_dir, case_id, logical_name)
        if source is None:
            if logical_name in REQUIRED_KEYS:
                raise FileNotFoundError(
                    f"Missing required file for case {case_id}: expected one of "
                    f"{SOURCE_PATTERNS[logical_name]!r} or {TARGET_FILENAMES[logical_name]!r}"
                )
            missing_optional.append(logical_name)
            continue

        destination = destination_case_dir / TARGET_FILENAMES[logical_name]
        if source.resolve() == destination.resolve():
            continue

        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Target file already exists for case {case_id}: {destination}. "
                "Use --overwrite to replace it."
            )

        mode = "copy" if output_root is not None else "rename"
        operations.append(Operation(source=source, destination=destination, mode=mode))

    return operations, missing_optional


def execute_operations(operations: list[Operation], overwrite: bool, dry_run: bool) -> None:
    for operation in operations:
        print(f"  {operation.mode}: {operation.source.name} -> {operation.destination.name}")
        if dry_run:
            continue

        operation.destination.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and operation.destination.exists():
            operation.destination.unlink()

        if operation.mode == "copy":
            shutil.copy2(operation.source, operation.destination)
        else:
            shutil.move(str(operation.source), str(operation.destination))


def main() -> None:
    args = build_argparser().parse_args()
    input_root = args.input_root
    output_root = args.output_root

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    case_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No case directories were found under {input_root}")

    success = 0
    failures: list[tuple[str, str]] = []

    print(f"Found {len(case_dirs)} case directories under {input_root}")
    if output_root is not None:
        print(f"Output root: {output_root}")
    print(f"Mode: {'dry-run' if args.dry_run else 'apply'}")

    for index, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        try:
            operations, missing_optional = build_case_operations(
                case_dir=case_dir,
                output_root=output_root,
                overwrite=args.overwrite,
            )
            print(f"[{index}/{len(case_dirs)}] {case_id}")
            execute_operations(operations, overwrite=args.overwrite, dry_run=args.dry_run)
            if not operations:
                print("  no changes needed")
            if missing_optional:
                print(f"  missing optional files: {', '.join(missing_optional)}")
            success += 1
        except Exception as exc:
            failures.append((case_id, str(exc)))
            print(f"[{index}/{len(case_dirs)}] {case_id}: failed -> {exc}")

    print(f"Finished. Successful cases: {success}, failed cases: {len(failures)}")
    if failures:
        for case_id, message in failures:
            print(f"  {case_id}: {message}")


if __name__ == "__main__":
    main()
