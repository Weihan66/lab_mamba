from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence
import json

import numpy as np
from scipy import ndimage
import torch


def _flatten_binary_scores(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if preds.shape != targets.shape:
        raise ValueError(f"Prediction and target shapes must match, got {preds.shape} and {targets.shape}.")

    preds = preds.float().reshape(preds.shape[0], -1)
    targets = targets.float().reshape(targets.shape[0], -1)
    return preds, targets


def binary_dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    preds = (preds > threshold).float()
    preds, targets = _flatten_binary_scores(preds, targets)

    intersection = (preds * targets).sum(dim=1)
    cardinality = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return dice.mean()


def _as_batch_binary_mask(volume: torch.Tensor) -> torch.Tensor:
    volume = torch.as_tensor(volume).detach().cpu().float()
    if volume.ndim == 3:
        volume = volume.unsqueeze(0)
    elif volume.ndim == 4:
        pass
    elif volume.ndim == 5:
        if volume.shape[1] != 1:
            raise ValueError(f"Expected a single-channel tensor for HD95, got shape {tuple(volume.shape)}.")
        volume = volume[:, 0]
    else:
        raise ValueError(f"Unsupported tensor rank for HD95: shape={tuple(volume.shape)}")
    return volume > 0


def _normalize_spacing(
    spacing: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor | None,
    batch_size: int,
) -> list[tuple[float, float, float]]:
    default_spacing = (1.0, 1.0, 1.0)
    if spacing is None:
        return [default_spacing for _ in range(batch_size)]

    if isinstance(spacing, torch.Tensor):
        spacing = spacing.detach().cpu().tolist()

    if len(spacing) == 3 and not isinstance(spacing[0], (list, tuple)):
        return [tuple(float(v) for v in spacing) for _ in range(batch_size)]

    spacing_list: list[tuple[float, float, float]] = []
    for item in spacing:  # type: ignore[arg-type]
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().tolist()
        if len(item) != 3:
            raise ValueError(f"Each spacing entry must contain 3 values, got {item!r}.")
        spacing_list.append(tuple(float(v) for v in item))

    if len(spacing_list) != batch_size:
        raise ValueError(f"Expected {batch_size} spacing entries, got {len(spacing_list)}.")
    return spacing_list


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    surface = np.logical_xor(mask, eroded)
    return surface if surface.any() else mask


def _compute_single_hd95(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    spacing: tuple[float, float, float],
    empty_value: float,
) -> float:
    pred_mask = pred_mask.astype(bool, copy=False)
    target_mask = target_mask.astype(bool, copy=False)

    pred_nonzero = bool(pred_mask.any())
    target_nonzero = bool(target_mask.any())

    if not pred_nonzero and not target_nonzero:
        return 0.0
    if pred_nonzero != target_nonzero:
        return empty_value

    pred_surface = _surface_mask(pred_mask)
    target_surface = _surface_mask(target_mask)

    pred_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)[pred_surface]
    target_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[target_surface]
    all_surface_distances = np.concatenate([pred_to_target, target_to_pred])
    return float(np.percentile(all_surface_distances, 95))


def binary_hd95_scores(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    spacing: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor | None = None,
    empty_value: float = float("inf"),
) -> torch.Tensor:
    pred_masks = _as_batch_binary_mask((preds > threshold).float())
    target_masks = _as_batch_binary_mask(targets)
    if pred_masks.shape != target_masks.shape:
        raise ValueError(
            f"Prediction and target shapes must match for HD95, got {tuple(pred_masks.shape)} and {tuple(target_masks.shape)}."
        )

    spacing_list = _normalize_spacing(spacing, pred_masks.shape[0])
    values = [
        _compute_single_hd95(pred_masks[idx].numpy(), target_masks[idx].numpy(), spacing_list[idx], empty_value)
        for idx in range(pred_masks.shape[0])
    ]
    return torch.tensor(values, dtype=torch.float32)


def binary_hd95(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    spacing: Sequence[float] | Sequence[Sequence[float]] | torch.Tensor | None = None,
    empty_value: float = float("inf"),
) -> torch.Tensor:
    scores = binary_hd95_scores(preds, targets, threshold=threshold, spacing=spacing, empty_value=empty_value)
    return scores.mean()


def summarize_metric(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(v) for v in values]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "num_values": 0}

    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "num_values": int(array.size),
    }


def load_case_spacing_map(root: str | Path) -> dict[str, tuple[float, float, float]]:
    root = Path(root)
    spacing_by_case: dict[str, tuple[float, float, float]] = {}
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_spacing = metadata.get("target_spacing_xyz")
        if raw_spacing is None or len(raw_spacing) != 3:
            continue

        # Arrays are stored in z, y, x order after SimpleITK -> NumPy conversion.
        spacing_by_case[case_dir.name] = tuple(float(v) for v in reversed(raw_spacing))
    return spacing_by_case


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.4f}"
