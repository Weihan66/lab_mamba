from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from petct_seg_data import load_volume, normalize_volume
from petct_seg_losses import binary_dice_score
from petct_seg_metrics import binary_hd95_scores, format_metric, load_case_spacing_map
from petct_seg_model import DualModalSegNet3D

GT_CMAP = ListedColormap([(0.0, 0.0, 0.0, 0.0), (1.0, 0.1, 0.1, 0.65)])
PRED_CMAP = ListedColormap([(0.0, 0.0, 0.0, 0.0), (0.1, 1.0, 0.2, 0.65)])
PLANE_NAMES = ("axial", "coronal", "sagittal")
PLANE_AXES = (0, 1, 2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize PET/CT cases with GT and prediction overlays.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory of processed case folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save PNG figures.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint used to run fresh predictions.")
    parser.add_argument("--preds-dir", type=Path, default=None, help="Prediction directory saved by evaluate.py.")
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Case id to visualize. Repeat the flag or pass comma-separated ids.",
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Optional evaluate.py JSON. Used to select cases by Dice ranking or threshold.",
    )
    parser.add_argument(
        "--top-k-worst",
        type=int,
        default=0,
        help="Select the worst-k cases by Dice from --eval-json.",
    )
    parser.add_argument(
        "--top-k-best",
        type=int,
        default=0,
        help="Select the best-k cases by Dice from --eval-json.",
    )
    parser.add_argument(
        "--min-dice",
        type=float,
        default=None,
        help="Optional Dice lower bound applied to --eval-json case selection.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold used to binarize prediction.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def _find_existing_path(case_dir: Path, stem: str) -> Path | None:
    for suffix in (".npy", ".npz", ".pt", ".pth"):
        path = case_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _parse_case_ids(values: list[str] | None) -> list[str]:
    if not values:
        return []

    case_ids: list[str] = []
    for value in values:
        for case_id in value.split(","):
            case_id = case_id.strip()
            if case_id:
                case_ids.append(case_id)
    return case_ids


def _load_eval_cases(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"Expected 'cases' list in {path}.")
    return [case for case in cases if isinstance(case, dict) and "case_id" in case]


def _dice_sort_key(case: dict[str, object]) -> float:
    dice = case.get("dice")
    if dice is None:
        return float("inf")
    return float(dice)


def _filter_eval_cases(eval_cases: list[dict[str, object]], min_dice: float | None) -> list[dict[str, object]]:
    filtered = eval_cases
    if min_dice is not None:
        filtered = [
            case
            for case in filtered
            if case.get("dice") is not None and float(case["dice"]) >= min_dice
        ]
    return filtered


def _select_case_ids(args: argparse.Namespace, dataset_case_ids: list[str]) -> list[str]:
    explicit_case_ids = _parse_case_ids(args.case_id)
    if explicit_case_ids:
        selected = explicit_case_ids
    elif args.top_k_worst > 0 or args.top_k_best > 0 or args.min_dice is not None:
        if args.eval_json is None:
            raise ValueError("--top-k-worst/--top-k-best/--min-dice require --eval-json.")
        if args.top_k_worst > 0 and args.top_k_best > 0:
            raise ValueError("--top-k-worst and --top-k-best cannot be used together.")
        eval_cases = _load_eval_cases(args.eval_json)
        eval_cases = _filter_eval_cases(eval_cases, min_dice=args.min_dice)
        if not eval_cases:
            raise ValueError("No cases matched the requested Dice filter.")

        if args.top_k_best > 0:
            eval_cases.sort(key=_dice_sort_key, reverse=True)
            selected = [str(case["case_id"]) for case in eval_cases[: args.top_k_best]]
        else:
            eval_cases.sort(key=_dice_sort_key)
            limit = args.top_k_worst if args.top_k_worst > 0 else len(eval_cases)
            selected = [str(case["case_id"]) for case in eval_cases[:limit]]
    else:
        selected = dataset_case_ids

    deduped: list[str] = []
    seen: set[str] = set()
    for case_id in selected:
        if case_id not in seen:
            deduped.append(case_id)
            seen.add(case_id)

    missing = [case_id for case_id in deduped if case_id not in dataset_case_ids]
    if missing:
        raise FileNotFoundError(f"Cases not found under {args.dataset_dir}: {missing}")
    return deduped


def _load_model(checkpoint_path: Path, device: torch.device) -> DualModalSegNet3D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = DualModalSegNet3D(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _load_prediction_from_model(
    model: DualModalSegNet3D,
    case_dir: Path,
    device: torch.device,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pet = normalize_volume(load_volume(case_dir / "pet.npy")).unsqueeze(0).to(device)
    ct = normalize_volume(load_volume(case_dir / "ct.npy")).unsqueeze(0).to(device)
    with torch.inference_mode():
        prob = model.predict_proba(pet, ct).squeeze(0).cpu()
    pred = (prob >= threshold).float()
    return prob, pred


def _load_prediction_from_dir(case_pred_dir: Path, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    prob_path = _find_existing_path(case_pred_dir, "prob")
    mask_path = _find_existing_path(case_pred_dir, "mask")
    if prob_path is None and mask_path is None:
        raise FileNotFoundError(f"Cannot find 'prob' or 'mask' under {case_pred_dir}.")

    prob = load_volume(prob_path) if prob_path is not None else None
    pred = load_volume(mask_path) if mask_path is not None else None

    if prob is None:
        prob = pred.float()
    if pred is None:
        pred = (prob >= threshold).float()
    return prob.float(), pred.float()


def _compute_focus_center(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[int, int, int]:
    focus = gt_mask > 0
    if not focus.any():
        focus = pred_mask > 0

    if focus.any():
        coords = np.argwhere(focus)
        min_corner = coords.min(axis=0)
        max_corner = coords.max(axis=0)
        center = np.rint((min_corner + max_corner) / 2.0).astype(int)
        return int(center[0]), int(center[1]), int(center[2])

    shape = np.asarray(gt_mask.shape, dtype=int)
    center = shape // 2
    return int(center[0]), int(center[1]), int(center[2])


def _extract_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    slice_2d = np.take(volume, index, axis=axis)
    return np.rot90(slice_2d)


def _robust_limits(volume: np.ndarray) -> tuple[float, float]:
    values = volume[np.nonzero(volume)]
    if values.size == 0:
        values = volume.reshape(-1)

    low, high = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or math.isclose(float(low), float(high)):
        low = float(values.min())
        high = float(values.max())
    if math.isclose(float(low), float(high)):
        high = float(low) + 1.0
    return float(low), float(high)


def _masked_binary(mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(mask <= 0, mask)


def _plot_case(
    case_id: str,
    pet_volume: np.ndarray,
    ct_volume: np.ndarray,
    gt_volume: np.ndarray,
    pred_volume: np.ndarray,
    dice: float,
    hd95: float | None,
    output_path: Path,
    dpi: int,
) -> None:
    pet_limits = _robust_limits(pet_volume)
    ct_limits = _robust_limits(ct_volume)
    center = _compute_focus_center(gt_volume, pred_volume)

    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 9), dpi=dpi, constrained_layout=True)
    column_titles = ("CT", "PET", "GT Mask", "Pred Mask", "GT on CT", "GT + Pred on CT")

    for row, (plane_name, axis, index) in enumerate(zip(PLANE_NAMES, PLANE_AXES, center)):
        ct_slice = _extract_slice(ct_volume, axis, index)
        pet_slice = _extract_slice(pet_volume, axis, index)
        gt_slice = _extract_slice(gt_volume, axis, index)
        pred_slice = _extract_slice(pred_volume, axis, index)

        axes[row, 0].imshow(ct_slice, cmap="gray", vmin=ct_limits[0], vmax=ct_limits[1], origin="lower")
        axes[row, 1].imshow(pet_slice, cmap="inferno", vmin=pet_limits[0], vmax=pet_limits[1], origin="lower")
        axes[row, 2].imshow(gt_slice, cmap="Reds", vmin=0.0, vmax=1.0, origin="lower")
        axes[row, 3].imshow(pred_slice, cmap="Greens", vmin=0.0, vmax=1.0, origin="lower")

        axes[row, 4].imshow(ct_slice, cmap="gray", vmin=ct_limits[0], vmax=ct_limits[1], origin="lower")
        axes[row, 4].imshow(_masked_binary(gt_slice), cmap=GT_CMAP, vmin=0.0, vmax=1.0, origin="lower")

        axes[row, 5].imshow(ct_slice, cmap="gray", vmin=ct_limits[0], vmax=ct_limits[1], origin="lower")
        axes[row, 5].imshow(_masked_binary(gt_slice), cmap=GT_CMAP, vmin=0.0, vmax=1.0, origin="lower")
        axes[row, 5].imshow(_masked_binary(pred_slice), cmap=PRED_CMAP, vmin=0.0, vmax=1.0, origin="lower")

        for col, axis_obj in enumerate(axes[row]):
            axis_obj.set_xticks([])
            axis_obj.set_yticks([])
            if row == 0:
                axis_obj.set_title(column_titles[col], fontsize=11)

        axes[row, 0].set_ylabel(f"{plane_name}\nidx={index}", fontsize=11)

    fig.suptitle(
        (
            f"case={case_id} | dice={dice:.4f} | hd95={format_metric(hd95)} | "
            f"gt_voxels={int(gt_volume.sum())} | pred_voxels={int(pred_volume.sum())}\n"
            "Overlay colors: GT=red, Pred=green"
        ),
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_argparser().parse_args()
    if args.checkpoint is None and args.preds_dir is None:
        raise ValueError("Provide either --checkpoint or --preds-dir.")

    dataset_case_dirs = sorted(path for path in args.dataset_dir.iterdir() if path.is_dir())
    dataset_case_ids = [case_dir.name for case_dir in dataset_case_dirs]
    selected_case_ids = _select_case_ids(args, dataset_case_ids)
    spacing_by_case = load_case_spacing_map(args.dataset_dir)

    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device) if args.checkpoint is not None else None

    for case_id in selected_case_ids:
        case_dir = args.dataset_dir / case_id
        pet = load_volume(case_dir / "pet.npy")[0].numpy()
        ct = load_volume(case_dir / "ct.npy")[0].numpy()
        gt = (load_volume(case_dir / "mask.npy") > 0).float()
        spacing = spacing_by_case.get(case_id, (1.0, 1.0, 1.0))

        if args.preds_dir is not None:
            prob, pred = _load_prediction_from_dir(args.preds_dir / case_id, args.threshold)
        else:
            assert model is not None
            prob, pred = _load_prediction_from_model(model, case_dir, device, args.threshold)

        dice = binary_dice_score(pred.unsqueeze(0), gt.unsqueeze(0), threshold=args.threshold).item()
        hd95_value = float(
            binary_hd95_scores(prob.unsqueeze(0), gt.unsqueeze(0), threshold=args.threshold, spacing=[spacing])[0].item()
        )
        hd95 = hd95_value if math.isfinite(hd95_value) else None

        output_path = args.output_dir / f"{case_id}.png"
        _plot_case(
            case_id=case_id,
            pet_volume=pet,
            ct_volume=ct,
            gt_volume=gt[0].numpy(),
            pred_volume=pred[0].numpy(),
            dice=dice,
            hd95=hd95,
            output_path=output_path,
            dpi=args.dpi,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
