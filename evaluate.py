from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from petct_seg_data import load_volume, normalize_volume, save_volume
from petct_seg_losses import binary_dice_score
from petct_seg_metrics import binary_hd95_scores, load_case_spacing_map, summarize_metric
from petct_seg_model import DualModalSegNet3D


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PET-CT segmentation checkpoint on a case directory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory of processed cases to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-preds-dir", type=Path, default=None, help="Optional directory for prob/mask outputs.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON file to save full metrics.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = DualModalSegNet3D(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    case_dirs = sorted(path for path in args.dataset_dir.iterdir() if path.is_dir())
    spacing_by_case = load_case_spacing_map(args.dataset_dir)

    results: list[dict[str, float | int | str | None]] = []
    finite_hd95: list[float] = []
    hd95_invalid = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        pet = normalize_volume(load_volume(case_dir / "pet.npy")).unsqueeze(0).to(device)
        ct = normalize_volume(load_volume(case_dir / "ct.npy")).unsqueeze(0).to(device)
        gt = (load_volume(case_dir / "mask.npy") > 0).float()
        spacing = spacing_by_case.get(case_id, (1.0, 1.0, 1.0))

        with torch.inference_mode():
            prob = model.predict_proba(pet, ct).squeeze(0).cpu()
        pred = (prob >= args.threshold).float()

        dice = binary_dice_score(pred.unsqueeze(0), gt.unsqueeze(0), threshold=args.threshold).item()
        hd95 = float(
            binary_hd95_scores(prob.unsqueeze(0), gt.unsqueeze(0), threshold=args.threshold, spacing=[spacing])[0].item()
        )
        hd95_is_finite = torch.isfinite(torch.tensor(hd95)).item()
        if hd95_is_finite:
            finite_hd95.append(hd95)
        else:
            hd95_invalid += 1

        result = {
            "case_id": case_id,
            "dice": dice,
            "hd95": hd95 if hd95_is_finite else None,
            "hd95_is_finite": bool(hd95_is_finite),
            "gt_voxels": int(gt.sum().item()),
            "pred_voxels": int(pred.sum().item()),
        }
        results.append(result)

        if args.save_preds_dir is not None:
            output_case_dir = args.save_preds_dir / case_id
            output_case_dir.mkdir(parents=True, exist_ok=True)
            save_volume(output_case_dir / "prob.npy", prob)
            save_volume(output_case_dir / "mask.npy", pred)

        hd95_text = f"{hd95:.4f}" if hd95_is_finite else "inf"
        print(
            f"{case_id}: dice={dice:.4f} hd95={hd95_text} "
            f"gt_voxels={result['gt_voxels']} pred_voxels={result['pred_voxels']}"
        )

    dice_summary = summarize_metric(result["dice"] for result in results)
    hd95_summary = summarize_metric(finite_hd95)
    summary = {
        "num_cases": len(results),
        "threshold": args.threshold,
        "dice": dice_summary,
        "hd95": {
            **hd95_summary,
            "num_invalid": hd95_invalid,
        },
    }

    print()
    print(
        "Dice:",
        f"mean={dice_summary['mean']:.4f}",
        f"std={dice_summary['std']:.4f}",
        f"min={dice_summary['min']:.4f}",
        f"max={dice_summary['max']:.4f}",
    )
    if hd95_summary["mean"] is None:
        print("HD95: no finite values", f"invalid={hd95_invalid}")
    else:
        print(
            "HD95:",
            f"mean={hd95_summary['mean']:.4f}",
            f"std={hd95_summary['std']:.4f}",
            f"min={hd95_summary['min']:.4f}",
            f"max={hd95_summary['max']:.4f}",
            f"invalid={hd95_invalid}",
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "dataset_dir": str(args.dataset_dir),
            "summary": summary,
            "cases": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved evaluation summary to {args.output_json}")


if __name__ == "__main__":
    main()
