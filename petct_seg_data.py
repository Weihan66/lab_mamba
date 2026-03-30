from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _find_existing_path(case_dir: Path, stem: str) -> Path:
    candidates = [
        case_dir / f"{stem}.npy",
        case_dir / f"{stem}.npz",
        case_dir / f"{stem}.pt",
        case_dir / f"{stem}.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find '{stem}' volume under {case_dir}.")


def load_volume(path: str | Path) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        array = np.load(path)
        tensor = torch.from_numpy(array)
    elif suffix == ".npz":
        data = np.load(path)
        key = data.files[0]
        tensor = torch.from_numpy(data[key])
    elif suffix in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
    else:
        raise ValueError(f"Unsupported volume format: {path}")

    tensor = tensor.float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(f"Expected a 3D volume or [C, D, H, W] tensor, got shape {tuple(tensor.shape)} for {path}.")
    return tensor


def save_volume(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    tensor = tensor.detach().cpu()

    if path.suffix.lower() == ".npy":
        np.save(path, tensor.numpy())
    elif path.suffix.lower() in {".pt", ".pth"}:
        torch.save(tensor, path)
    else:
        raise ValueError(f"Unsupported output format: {path}. Use .npy, .pt or .pth.")


def normalize_volume(volume: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = volume.ne(0)
    if mask.any():
        values = volume[mask]
    else:
        values = volume
    mean = values.mean()
    std = values.std().clamp_min(eps)
    return (volume - mean) / std


def _sample_flip_dims(dims: tuple[int, ...], probability: float) -> tuple[int, ...]:
    selected_dims: list[int] = []
    for dim in dims:
        if torch.rand(1).item() < probability:
            selected_dims.append(dim)
    return tuple(selected_dims)


def _augment_intensity(
    volume: torch.Tensor,
    foreground_mask: torch.Tensor,
    scale_range: float,
    shift_range: float,
    noise_std: float,
) -> torch.Tensor:
    augmented = volume
    if scale_range > 0.0:
        scale = 1.0 + float(torch.empty(1).uniform_(-scale_range, scale_range).item())
        augmented = torch.where(foreground_mask, augmented * scale, augmented)
    if shift_range > 0.0:
        shift = float(torch.empty(1).uniform_(-shift_range, shift_range).item())
        augmented = torch.where(foreground_mask, augmented + shift, augmented)
    if noise_std > 0.0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = torch.where(foreground_mask, augmented + noise, augmented)
    return augmented


class PETCTVolumeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        normalize: bool = True,
        augment: bool = False,
        flip_probability: float = 0.5,
        intensity_scale_range: float = 0.1,
        intensity_shift_range: float = 0.1,
        noise_std: float = 0.03,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.case_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        if not self.case_dirs:
            raise ValueError(f"No case directories found under {self.root}")

        self.normalize = normalize
        self.augment = augment
        self.flip_probability = float(flip_probability)
        self.intensity_scale_range = float(intensity_scale_range)
        self.intensity_shift_range = float(intensity_shift_range)
        self.noise_std = float(noise_std)

        if not 0.0 <= self.flip_probability <= 1.0:
            raise ValueError(f"flip_probability must be in [0, 1], got {self.flip_probability}.")
        if self.intensity_scale_range < 0.0:
            raise ValueError(f"intensity_scale_range must be non-negative, got {self.intensity_scale_range}.")
        if self.intensity_shift_range < 0.0:
            raise ValueError(f"intensity_shift_range must be non-negative, got {self.intensity_shift_range}.")
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}.")

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_dir = self.case_dirs[index]
        pet = load_volume(_find_existing_path(case_dir, "pet"))
        ct = load_volume(_find_existing_path(case_dir, "ct"))
        mask = load_volume(_find_existing_path(case_dir, "mask"))

        if pet.shape[1:] != ct.shape[1:] or pet.shape[1:] != mask.shape[1:]:
            raise ValueError(
                f"PET, CT and mask must share the same spatial size in {case_dir}, "
                f"got {pet.shape}, {ct.shape}, {mask.shape}."
            )

        pet_foreground = pet.ne(0)
        ct_foreground = ct.ne(0)

        if self.normalize:
            pet = normalize_volume(pet)
            ct = normalize_volume(ct)

        mask = (mask > 0).float()

        if self.augment:
            spatial_dims = (1, 2, 3)
            flip_dims = _sample_flip_dims(spatial_dims, probability=self.flip_probability)
            if flip_dims:
                pet = torch.flip(pet, dims=flip_dims)
                ct = torch.flip(ct, dims=flip_dims)
                mask = torch.flip(mask, dims=flip_dims)
                pet_foreground = torch.flip(pet_foreground, dims=flip_dims)
                ct_foreground = torch.flip(ct_foreground, dims=flip_dims)

            pet = _augment_intensity(
                pet,
                foreground_mask=pet_foreground,
                scale_range=self.intensity_scale_range,
                shift_range=self.intensity_shift_range,
                noise_std=self.noise_std,
            )
            ct = _augment_intensity(
                ct,
                foreground_mask=ct_foreground,
                scale_range=self.intensity_scale_range,
                shift_range=self.intensity_shift_range,
                noise_std=self.noise_std,
            )

        return {
            "pet": pet,
            "ct": ct,
            "mask": mask,
            "case_id": case_dir.name,
        }
