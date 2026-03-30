from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from petct_seg_data import PETCTVolumeDataset, load_volume
from petct_seg_losses import binary_dice_score, build_loss
from petct_seg_metrics import binary_hd95_scores, format_metric, load_case_spacing_map
from petct_seg_model import DualModalSegNet3D


def parse_int_sequence(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def resolve_mask_voxels(case_dir: Path) -> int:
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_voxels = metadata.get("nonzero_mask_voxels")
        if raw_voxels is not None:
            return int(raw_voxels)

    mask = load_volume(case_dir / "mask.npy")
    return int((mask > 0).sum().item())


def build_lesion_size_sampler(
    dataset: PETCTVolumeDataset,
    *,
    small_quantile: float,
    large_quantile: float,
    small_weight: float,
    medium_weight: float,
    large_weight: float,
) -> tuple[WeightedRandomSampler, dict[str, float | int]]:
    mask_voxel_counts = [resolve_mask_voxels(case_dir) for case_dir in dataset.case_dirs]
    quantiles = np.quantile(mask_voxel_counts, [small_quantile, large_quantile])
    small_threshold = float(quantiles[0])
    large_threshold = float(quantiles[1])

    sample_weights: list[float] = []
    small_cases = 0
    medium_cases = 0
    large_cases = 0
    for count in mask_voxel_counts:
        if count <= small_threshold:
            sample_weights.append(float(small_weight))
            small_cases += 1
        elif count >= large_threshold:
            sample_weights.append(float(large_weight))
            large_cases += 1
        else:
            sample_weights.append(float(medium_weight))
            medium_cases += 1

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    summary = {
        "small_threshold": small_threshold,
        "large_threshold": large_threshold,
        "small_cases": small_cases,
        "medium_cases": medium_cases,
        "large_cases": large_cases,
        "small_weight": float(small_weight),
        "medium_weight": float(medium_weight),
        "large_weight": float(large_weight),
    }
    return sampler, summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the dual-modal PET-CT 3D segmentation network.")
    parser.add_argument("--train-dir", type=Path, required=True, help="Directory of training cases.")
    parser.add_argument("--val-dir", type=Path, default=None, help="Directory of validation cases.")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"), help="Checkpoint output directory.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels", type=str, default="32,64,128,256,512")
    parser.add_argument("--pet-depths", type=str, default="2,2,2,2,2")
    parser.add_argument("--ct-depths", type=str, default="2,2,2,2")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ct-encoder-type", type=str, default="official_segmamba_v2")
    parser.add_argument("--official-segmamba-path", type=Path, default=None)
    parser.add_argument("--official-segmamba-source", type=str, default="brats23")
    parser.add_argument("--disable-ct-fallback", action="store_true")
    parser.add_argument(
        "--train-sampler",
        type=str,
        default="uniform",
        choices=("uniform", "lesion_size_bins"),
        help="Training-set case sampling strategy.",
    )
    parser.add_argument(
        "--small-case-quantile",
        type=float,
        default=0.3,
        help="Cases at or below this lesion-size quantile are treated as small when using lesion_size_bins.",
    )
    parser.add_argument(
        "--large-case-quantile",
        type=float,
        default=0.7,
        help="Cases at or above this lesion-size quantile are treated as large when using lesion_size_bins.",
    )
    parser.add_argument(
        "--small-case-weight",
        type=float,
        default=2.0,
        help="Sampling weight for small-lesion cases when using lesion_size_bins.",
    )
    parser.add_argument(
        "--medium-case-weight",
        type=float,
        default=1.0,
        help="Sampling weight for medium-lesion cases when using lesion_size_bins.",
    )
    parser.add_argument(
        "--large-case-weight",
        type=float,
        default=0.7,
        help="Sampling weight for large-lesion cases when using lesion_size_bins.",
    )
    parser.add_argument("--augment", action="store_true", help="Enable lightweight training-time augmentation.")
    parser.add_argument("--aug-flip-prob", type=float, default=0.5, help="Per-axis flip probability when --augment is set.")
    parser.add_argument(
        "--aug-intensity-scale",
        type=float,
        default=0.1,
        help="Uniform intensity scaling range when --augment is set.",
    )
    parser.add_argument(
        "--aug-intensity-shift",
        type=float,
        default=0.1,
        help="Uniform intensity shift range when --augment is set.",
    )
    parser.add_argument("--aug-noise-std", type=float, default=0.03, help="Gaussian noise std when --augment is set.")
    parser.add_argument(
        "--loss",
        type=str,
        default="dice_bce",
        choices=("dice_bce", "tversky", "focal_tversky"),
        help="Training loss to optimize.",
    )
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice weight used by dice_bce.")
    parser.add_argument("--bce-weight", type=float, default=1.0, help="BCE weight used by dice_bce.")
    parser.add_argument("--bce-pos-weight", type=float, default=1.0, help="Positive-class weight used by dice_bce BCE.")
    parser.add_argument("--tversky-alpha", type=float, default=0.3, help="FP weight used by Tversky-style losses.")
    parser.add_argument("--tversky-beta", type=float, default=0.7, help="FN weight used by Tversky-style losses.")
    parser.add_argument(
        "--focal-tversky-gamma",
        type=float,
        default=0.75,
        help="Gamma exponent used by focal_tversky.",
    )
    parser.add_argument("--metric-threshold", type=float, default=0.5, help="Threshold used for Dice/HD95 reporting.")
    parser.add_argument("--compute-val-hd95", action="store_true", help="Compute HD95 during validation.")
    return parser


def run_epoch(
    model: DualModalSegNet3D,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    metric_threshold: float,
    compute_hd95: bool = False,
    spacing_by_case: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, float | int | None]:
    training = optimizer is not None
    model.train(training)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    total_loss = 0.0
    total_dice = 0.0
    total_batches = 0
    hd95_values: list[float] = []
    hd95_invalid = 0

    for batch in loader:
        pet = batch["pet"].to(device)
        ct = batch["ct"].to(device)
        mask = batch["mask"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits = model(pet, ct)
            loss = criterion(logits, mask)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dice = binary_dice_score(probs, mask, threshold=metric_threshold)
            if compute_hd95:
                case_ids = batch["case_id"]
                if isinstance(case_ids, str):
                    case_ids = [case_ids]
                spacings = None
                if spacing_by_case is not None:
                    spacings = [spacing_by_case.get(case_id, (1.0, 1.0, 1.0)) for case_id in case_ids]
                hd95_batch = binary_hd95_scores(
                    probs,
                    mask,
                    threshold=metric_threshold,
                    spacing=spacings,
                )
                for value in hd95_batch.tolist():
                    if math.isfinite(value):
                        hd95_values.append(float(value))
                    else:
                        hd95_invalid += 1

        total_loss += loss.item()
        total_dice += dice.item()
        total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "dice": 0.0, "hd95": None, "hd95_invalid": 0}
    return {
        "loss": total_loss / total_batches,
        "dice": total_dice / total_batches,
        "hd95": (sum(hd95_values) / len(hd95_values)) if hd95_values else None,
        "hd95_invalid": hd95_invalid,
    }


def main() -> None:
    args = build_argparser().parse_args()
    if args.dice_weight < 0.0 or args.bce_weight < 0.0:
        raise ValueError("dice_weight and bce_weight must be non-negative.")
    if args.bce_pos_weight <= 0.0:
        raise ValueError("bce_pos_weight must be positive.")
    if args.train_sampler == "lesion_size_bins":
        if not 0.0 < args.small_case_quantile < args.large_case_quantile < 1.0:
            raise ValueError("small_case_quantile and large_case_quantile must satisfy 0 < small < large < 1.")
        if args.small_case_weight <= 0.0 or args.medium_case_weight <= 0.0 or args.large_case_weight <= 0.0:
            raise ValueError("case sampling weights must be positive.")
    if not 0.0 <= args.aug_flip_prob <= 1.0:
        raise ValueError("aug_flip_prob must be between 0 and 1.")
    if args.aug_intensity_scale < 0.0 or args.aug_intensity_shift < 0.0 or args.aug_noise_std < 0.0:
        raise ValueError("augmentation ranges/std must be non-negative.")
    if args.tversky_alpha < 0.0 or args.tversky_beta < 0.0 or args.tversky_alpha + args.tversky_beta <= 0.0:
        raise ValueError("tversky_alpha and tversky_beta must be non-negative and sum to a positive value.")
    if args.focal_tversky_gamma <= 0.0:
        raise ValueError("focal_tversky_gamma must be positive.")

    channels = parse_int_sequence(args.channels)
    pet_depths = parse_int_sequence(args.pet_depths)
    ct_depths = parse_int_sequence(args.ct_depths)

    device = torch.device(args.device)
    train_dataset = PETCTVolumeDataset(
        args.train_dir,
        augment=args.augment,
        flip_probability=args.aug_flip_prob,
        intensity_scale_range=args.aug_intensity_scale,
        intensity_shift_range=args.aug_intensity_shift,
        noise_std=args.aug_noise_std,
    )
    train_sampler = None
    train_sampler_summary = None
    if args.train_sampler == "lesion_size_bins":
        train_sampler, train_sampler_summary = build_lesion_size_sampler(
            train_dataset,
            small_quantile=args.small_case_quantile,
            large_quantile=args.large_case_quantile,
            small_weight=args.small_case_weight,
            medium_weight=args.medium_case_weight,
            large_weight=args.large_case_weight,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    val_spacing_by_case = None
    if args.val_dir is not None:
        val_dataset = PETCTVolumeDataset(args.val_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if args.compute_val_hd95:
            val_spacing_by_case = load_case_spacing_map(args.val_dir)

    model = DualModalSegNet3D(
        channels=channels,
        pet_depths=pet_depths,
        ct_depths=ct_depths,
        dropout=args.dropout,
        ct_encoder_type=args.ct_encoder_type,
        official_segmamba_path=str(args.official_segmamba_path) if args.official_segmamba_path is not None else None,
        official_segmamba_source=args.official_segmamba_source,
        allow_ct_encoder_fallback=not args.disable_ct_fallback,
    ).to(device)
    criterion = build_loss(
        args.loss,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        bce_pos_weight=args.bce_pos_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        focal_tversky_gamma=args.focal_tversky_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0

    print(f"Training on {device} with {len(train_dataset)} training cases.")
    print(f"Train sampler: {args.train_sampler}")
    if train_sampler_summary is not None:
        print(
            "Sampler bins:",
            f"small<= {train_sampler_summary['small_threshold']:.1f} ({train_sampler_summary['small_cases']} cases, w={train_sampler_summary['small_weight']}),",
            f"medium ({train_sampler_summary['medium_cases']} cases, w={train_sampler_summary['medium_weight']}),",
            f"large>= {train_sampler_summary['large_threshold']:.1f} ({train_sampler_summary['large_cases']} cases, w={train_sampler_summary['large_weight']})",
        )
    if args.augment:
        print(
            "Augment:",
            f"flip_prob={args.aug_flip_prob}, scale={args.aug_intensity_scale},",
            f"shift={args.aug_intensity_shift}, noise_std={args.aug_noise_std}",
        )
    print(
        "Loss:",
        args.loss,
        f"(dice_weight={args.dice_weight}, bce_weight={args.bce_weight}, bce_pos_weight={args.bce_pos_weight},",
        f"tversky_alpha={args.tversky_alpha}, tversky_beta={args.tversky_beta},",
        f"focal_tversky_gamma={args.focal_tversky_gamma})",
    )
    if val_loader is not None:
        print(f"Validation cases: {len(val_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.amp,
            metric_threshold=args.metric_threshold,
        )
        train_loss = float(train_metrics["loss"])
        train_dice = float(train_metrics["dice"])
        message = f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_dice={train_dice:.4f}"

        val_loss = None
        val_dice = None
        val_hd95 = None
        val_hd95_invalid = 0
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    None,
                    device,
                    args.amp,
                    metric_threshold=args.metric_threshold,
                    compute_hd95=args.compute_val_hd95,
                    spacing_by_case=val_spacing_by_case,
                )
            val_loss = val_metrics["loss"]
            val_dice = val_metrics["dice"]
            val_hd95 = val_metrics["hd95"]
            val_hd95_invalid = int(val_metrics["hd95_invalid"] or 0)
            message += f" val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
            if args.compute_val_hd95:
                message += f" val_hd95={format_metric(val_hd95)}"
                if val_hd95_invalid:
                    message += f" val_hd95_invalid={val_hd95_invalid}"

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_path = args.save_dir / "best.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_kwargs": {
                            "channels": channels,
                            "pet_depths": pet_depths,
                            "ct_depths": ct_depths,
                            "dropout": args.dropout,
                            "ct_encoder_type": model.ct_encoder_type,
                            "official_segmamba_path": str(args.official_segmamba_path)
                            if args.official_segmamba_path is not None
                            else None,
                            "official_segmamba_source": args.official_segmamba_source,
                            "allow_ct_encoder_fallback": not args.disable_ct_fallback,
                        },
                        "epoch": epoch,
                        "val_dice": val_dice,
                        "val_hd95": val_hd95,
                        "val_hd95_invalid": val_hd95_invalid,
                        "loss_name": args.loss,
                        "loss_kwargs": {
                            "dice_weight": args.dice_weight,
                            "bce_weight": args.bce_weight,
                            "bce_pos_weight": args.bce_pos_weight,
                            "tversky_alpha": args.tversky_alpha,
                            "tversky_beta": args.tversky_beta,
                            "focal_tversky_gamma": args.focal_tversky_gamma,
                        },
                        "train_sampler": args.train_sampler,
                        "train_sampler_kwargs": {
                            "small_case_quantile": args.small_case_quantile,
                            "large_case_quantile": args.large_case_quantile,
                            "small_case_weight": args.small_case_weight,
                            "medium_case_weight": args.medium_case_weight,
                            "large_case_weight": args.large_case_weight,
                        },
                    },
                    best_path,
                )

        print(message)

        last_path = args.save_dir / "last.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_kwargs": {
                    "channels": channels,
                    "pet_depths": pet_depths,
                    "ct_depths": ct_depths,
                    "dropout": args.dropout,
                    "ct_encoder_type": model.ct_encoder_type,
                    "official_segmamba_path": str(args.official_segmamba_path)
                    if args.official_segmamba_path is not None
                    else None,
                    "official_segmamba_source": args.official_segmamba_source,
                    "allow_ct_encoder_fallback": not args.disable_ct_fallback,
                },
                "epoch": epoch,
                "loss_name": args.loss,
                "loss_kwargs": {
                    "dice_weight": args.dice_weight,
                    "bce_weight": args.bce_weight,
                    "bce_pos_weight": args.bce_pos_weight,
                    "tversky_alpha": args.tversky_alpha,
                    "tversky_beta": args.tversky_beta,
                    "focal_tversky_gamma": args.focal_tversky_gamma,
                },
                "train_sampler": args.train_sampler,
                "train_sampler_kwargs": {
                    "small_case_quantile": args.small_case_quantile,
                    "large_case_quantile": args.large_case_quantile,
                    "small_case_weight": args.small_case_weight,
                    "medium_case_weight": args.medium_case_weight,
                    "large_case_weight": args.large_case_weight,
                },
                "metrics": {
                    "train_loss": train_loss,
                    "train_dice": train_dice,
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_hd95": val_hd95,
                    "val_hd95_invalid": val_hd95_invalid,
                },
            },
            last_path,
        )

    config_path = args.save_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"Training finished. Checkpoints saved to {args.save_dir}")


if __name__ == "__main__":
    main()
