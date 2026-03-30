from __future__ import annotations

import argparse
from pathlib import Path

import torch

from petct_seg_data import load_volume, normalize_volume, save_volume
from petct_seg_model import DualModalSegNet3D


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PET-CT 3D lesion segmentation inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pet", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--output-prob", type=Path, required=True, help="Output probability map (.npy/.pt).")
    parser.add_argument("--output-mask", type=Path, default=None, help="Optional thresholded mask (.npy/.pt).")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = DualModalSegNet3D(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pet = normalize_volume(load_volume(args.pet)).unsqueeze(0).to(device)
    ct = normalize_volume(load_volume(args.ct)).unsqueeze(0).to(device)

    with torch.inference_mode():
        prob = model.predict_proba(pet, ct).squeeze(0)
        mask = (prob >= args.threshold).float()

    args.output_prob.parent.mkdir(parents=True, exist_ok=True)
    save_volume(args.output_prob, prob)
    print(f"Saved probability map to {args.output_prob}")

    if args.output_mask is not None:
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        save_volume(args.output_mask, mask)
        print(f"Saved binary mask to {args.output_mask}")


if __name__ == "__main__":
    main()
