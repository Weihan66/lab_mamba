from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


REQUIRED_IMAGE_NAMES = ("ct.nii.gz", "pet.nii.gz")
OPTIONAL_LABEL_NAMES = ("ct_seg.nii.gz", "pet_seg.nii.gz")


@dataclass
class CasePaths:
    case_id: str
    case_dir: Path
    ct_path: Path
    pet_path: Path
    ct_seg_path: Path | None
    pet_seg_path: Path | None


def parse_3tuple(raw: str, cast_type: type[int] | type[float]) -> tuple[int, int, int] | tuple[float, float, float]:
    values = tuple(cast_type(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != 3:
        raise ValueError(f"Expected exactly 3 comma-separated values, got {raw!r}.")
    return values


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int(((value + multiple - 1) // multiple) * multiple)


def discover_cases(input_root: Path) -> list[CasePaths]:
    case_paths: list[CasePaths] = []
    for case_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        ct_path = case_dir / "ct.nii.gz"
        pet_path = case_dir / "pet.nii.gz"
        if not ct_path.exists() or not pet_path.exists():
            continue

        ct_seg_path = case_dir / "ct_seg.nii.gz"
        pet_seg_path = case_dir / "pet_seg.nii.gz"
        case_paths.append(
            CasePaths(
                case_id=case_dir.name,
                case_dir=case_dir,
                ct_path=ct_path,
                pet_path=pet_path,
                ct_seg_path=ct_seg_path if ct_seg_path.exists() else None,
                pet_seg_path=pet_seg_path if pet_seg_path.exists() else None,
            )
        )

    if not case_paths:
        raise ValueError(
            f"No valid cases found under {input_root}. Each case directory must contain {REQUIRED_IMAGE_NAMES}."
        )
    return case_paths


def read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def build_reference_image(image: sitk.Image, target_spacing: Sequence[float]) -> sitk.Image:
    spacing = tuple(float(v) for v in target_spacing)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    output_size = [
        max(1, int(round(original_size[i] * (original_spacing[i] / spacing[i]))))
        for i in range(3)
    ]

    reference = sitk.Image(output_size, image.GetPixelIDValue())
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing(spacing)
    return reference


def resample_image(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float = 0.0,
    transform: sitk.Transform | None = None,
) -> sitk.Image:
    transform = sitk.Transform(3, sitk.sitkIdentity) if transform is None else transform
    return sitk.Resample(image, reference, transform, interpolator, default_value, image.GetPixelIDValue())


def register_pet_to_ct(
    fixed_ct: sitk.Image,
    moving_pet: sitk.Image,
    transform_type: str = "rigid",
    iterations: int = 150,
    sampling_percentage: float = 0.2,
) -> tuple[sitk.Transform, float, str]:
    fixed = sitk.Cast(fixed_ct, sitk.sitkFloat32)
    moving = sitk.Cast(moving_pet, sitk.sitkFloat32)

    if transform_type == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    elif transform_type == "affine":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}. Use 'rigid' or 'affine'.")

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed, moving)
    return final_transform, float(registration.GetMetricValue()), registration.GetOptimizerStopConditionDescription()


def combine_labels(
    ct_mask: np.ndarray | None,
    pet_mask: np.ndarray | None,
    label_mode: str,
) -> np.ndarray | None:
    if ct_mask is None and pet_mask is None:
        return None
    if label_mode == "ct":
        return ct_mask
    if label_mode == "pet":
        return pet_mask

    ct_mask = np.zeros_like(pet_mask, dtype=bool) if ct_mask is None and pet_mask is not None else ct_mask
    pet_mask = np.zeros_like(ct_mask, dtype=bool) if pet_mask is None and ct_mask is not None else pet_mask

    if ct_mask is None or pet_mask is None:
        return ct_mask if pet_mask is None else pet_mask

    if label_mode == "union":
        return np.logical_or(ct_mask, pet_mask)
    if label_mode == "intersection":
        return np.logical_and(ct_mask, pet_mask)
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == sizes.argmax()


def build_body_mask_from_ct(ct_volume: np.ndarray, body_threshold: float) -> np.ndarray:
    mask = ct_volume > body_threshold
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    return largest_connected_component(mask)


def compute_bbox(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if mask is None or not np.any(mask):
        return None
    coords = np.where(mask)
    return tuple((int(coords[i].min()), int(coords[i].max()) + 1) for i in range(3))


def derive_crop_shape(
    base_crop_size: Sequence[int],
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None,
    crop_margin: int,
    size_divisor: int,
    allow_expand_crop: bool,
) -> tuple[int, int, int]:
    crop_shape = list(int(v) for v in base_crop_size)
    if bbox is not None and allow_expand_crop:
        needed = [(end - start) + 2 * crop_margin for start, end in bbox]
        crop_shape = [max(crop_shape[i], needed[i]) for i in range(3)]

    return tuple(round_up_to_multiple(v, size_divisor) for v in crop_shape)


def choose_crop_center(
    volume_shape: Sequence[int],
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int]:
    if bbox is None:
        return tuple(int(dim // 2) for dim in volume_shape)
    return tuple(int((start + end) // 2) for start, end in bbox)


def crop_or_pad(
    volume: np.ndarray,
    center: Sequence[int],
    crop_shape: Sequence[int],
    constant_value: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    slices: list[slice] = []
    padding: list[tuple[int, int]] = []
    start_indices: list[int] = []

    for dim_size, dim_center, target_size in zip(volume.shape, center, crop_shape):
        start = int(round(dim_center - target_size / 2))
        end = start + int(target_size)
        pad_before = max(0, -start)
        pad_after = max(0, end - dim_size)
        src_start = max(0, start)
        src_end = min(dim_size, end)

        slices.append(slice(src_start, src_end))
        padding.append((pad_before, pad_after))
        start_indices.append(start)

    cropped = volume[tuple(slices)]
    if any(before > 0 or after > 0 for before, after in padding):
        cropped = np.pad(cropped, padding, mode="constant", constant_values=constant_value)
    return cropped, tuple(start_indices)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32, copy=False))


def maybe_save_mask(path: Path, mask: np.ndarray | None) -> None:
    if mask is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.float32, copy=False))


def process_case(
    case: CasePaths,
    output_root: Path,
    target_spacing: Sequence[float],
    base_crop_size: Sequence[int],
    crop_mode: str,
    label_mode: str,
    transform_type: str,
    body_threshold: float,
    crop_margin: int,
    size_divisor: int,
    allow_expand_crop: bool,
    skip_registration: bool,
    overwrite: bool,
) -> dict[str, object]:
    output_case_dir = output_root / case.case_id
    metadata_path = output_case_dir / "metadata.json"
    if metadata_path.exists() and not overwrite:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    ct_image = read_image(case.ct_path)
    pet_image = read_image(case.pet_path)
    ct_seg_image = read_image(case.ct_seg_path) if case.ct_seg_path is not None else None
    pet_seg_image = read_image(case.pet_seg_path) if case.pet_seg_path is not None else None

    if skip_registration:
        final_transform = sitk.Transform(3, sitk.sitkIdentity)
        registration_metric = None
        registration_stop = "registration_skipped"
    else:
        final_transform, registration_metric, registration_stop = register_pet_to_ct(
            fixed_ct=ct_image,
            moving_pet=pet_image,
            transform_type=transform_type,
        )

    reference_image = build_reference_image(ct_image, target_spacing)
    ct_resampled = resample_image(ct_image, reference_image, interpolator=sitk.sitkLinear, default_value=0.0)
    pet_resampled = resample_image(
        pet_image,
        reference_image,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
        transform=final_transform,
    )

    ct_seg_resampled = (
        resample_image(ct_seg_image, reference_image, interpolator=sitk.sitkNearestNeighbor, default_value=0.0)
        if ct_seg_image is not None
        else None
    )
    pet_seg_resampled = (
        resample_image(
            pet_seg_image,
            reference_image,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0.0,
            transform=final_transform,
        )
        if pet_seg_image is not None
        else None
    )

    ct_volume = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    pet_volume = sitk.GetArrayFromImage(pet_resampled).astype(np.float32)
    ct_mask = (
        sitk.GetArrayFromImage(ct_seg_resampled).astype(np.float32) > 0
        if ct_seg_resampled is not None
        else None
    )
    pet_mask = (
        sitk.GetArrayFromImage(pet_seg_resampled).astype(np.float32) > 0
        if pet_seg_resampled is not None
        else None
    )

    final_mask = combine_labels(ct_mask, pet_mask, label_mode=label_mode)
    label_bbox = compute_bbox(final_mask) if final_mask is not None else None
    body_mask = build_body_mask_from_ct(ct_volume, body_threshold=body_threshold)
    body_bbox = compute_bbox(body_mask)

    if crop_mode == "lesion":
        crop_bbox = label_bbox
        crop_source = "lesion"
    elif crop_mode == "body":
        crop_bbox = body_bbox
        crop_source = "body"
    elif crop_mode == "auto":
        if label_bbox is not None:
            crop_bbox = label_bbox
            crop_source = "lesion"
        else:
            crop_bbox = body_bbox
            crop_source = "body"
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    crop_shape = derive_crop_shape(
        base_crop_size=base_crop_size,
        bbox=crop_bbox,
        crop_margin=crop_margin,
        size_divisor=size_divisor,
        allow_expand_crop=allow_expand_crop,
    )
    crop_center = choose_crop_center(ct_volume.shape, crop_bbox)

    ct_crop, crop_start = crop_or_pad(ct_volume, crop_center, crop_shape, constant_value=0.0)
    pet_crop, _ = crop_or_pad(pet_volume, crop_center, crop_shape, constant_value=0.0)
    mask_crop, _ = crop_or_pad(
        final_mask.astype(np.float32) if final_mask is not None else np.zeros_like(ct_volume, dtype=np.float32),
        crop_center,
        crop_shape,
        constant_value=0.0,
    )
    ct_label_crop, _ = crop_or_pad(
        ct_mask.astype(np.float32) if ct_mask is not None else np.zeros_like(ct_volume, dtype=np.float32),
        crop_center,
        crop_shape,
        constant_value=0.0,
    )
    pet_label_crop, _ = crop_or_pad(
        pet_mask.astype(np.float32) if pet_mask is not None else np.zeros_like(ct_volume, dtype=np.float32),
        crop_center,
        crop_shape,
        constant_value=0.0,
    )

    output_case_dir.mkdir(parents=True, exist_ok=True)
    save_array(output_case_dir / "ct.npy", ct_crop)
    save_array(output_case_dir / "pet.npy", pet_crop)
    maybe_save_mask(output_case_dir / "mask.npy", mask_crop if final_mask is not None else None)
    maybe_save_mask(output_case_dir / "ct_label.npy", ct_label_crop if ct_mask is not None else None)
    maybe_save_mask(output_case_dir / "pet_label.npy", pet_label_crop if pet_mask is not None else None)
    sitk.WriteTransform(final_transform, str(output_case_dir / "pet_to_ct.tfm"))

    metadata = {
        "case_id": case.case_id,
        "input_case_dir": str(case.case_dir),
        "original_ct_size_xyz": list(ct_image.GetSize()),
        "original_pet_size_xyz": list(pet_image.GetSize()),
        "original_ct_spacing_xyz": list(ct_image.GetSpacing()),
        "original_pet_spacing_xyz": list(pet_image.GetSpacing()),
        "target_spacing_xyz": list(reference_image.GetSpacing()),
        "resampled_size_xyz": list(reference_image.GetSize()),
        "crop_shape_zyx": list(crop_shape),
        "crop_center_zyx": list(crop_center),
        "crop_start_zyx": list(crop_start),
        "crop_source": crop_source,
        "label_mode": label_mode,
        "transform_type": "identity" if skip_registration else transform_type,
        "registration_metric": registration_metric,
        "registration_stop_condition": registration_stop,
        "transform_parameters": list(final_transform.GetParameters()),
        "transform_fixed_parameters": list(final_transform.GetFixedParameters()),
        "has_ct_label": ct_mask is not None,
        "has_pet_label": pet_mask is not None,
        "output_mask_written": final_mask is not None,
        "nonzero_mask_voxels": int(mask_crop.sum()) if final_mask is not None else 0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register raw PET/CT NIfTI volumes, crop them to a model-friendly 3D size, and export .npy cases."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory like nii_1_1_gz/<case_id>/")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory of processed cases.")
    parser.add_argument("--target-spacing", type=str, default="2.0,2.0,2.0", help="Target spacing in x,y,z (mm).")
    parser.add_argument("--crop-size", type=str, default="128,128,128", help="Base crop size in z,y,x.")
    parser.add_argument("--crop-mode", type=str, default="auto", choices=["auto", "lesion", "body"])
    parser.add_argument("--label-mode", type=str, default="union", choices=["union", "intersection", "ct", "pet"])
    parser.add_argument("--transform-type", type=str, default="rigid", choices=["rigid", "affine"])
    parser.add_argument("--body-threshold", type=float, default=-600.0, help="CT threshold used to estimate body mask.")
    parser.add_argument("--crop-margin", type=int, default=16, help="Extra lesion margin in voxels when expanding crop.")
    parser.add_argument("--size-divisor", type=int, default=16, help="Round crop sizes up to a multiple of this value.")
    parser.add_argument("--allow-expand-crop", action="store_true", help="Expand the crop if lesion bbox is larger.")
    parser.add_argument("--skip-registration", action="store_true", help="Skip PET-to-CT registration.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite processed cases if they already exist.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_root = args.input_root
    output_root = args.output_root
    target_spacing = parse_3tuple(args.target_spacing, float)
    crop_size = parse_3tuple(args.crop_size, int)

    cases = discover_cases(input_root)
    print(f"Found {len(cases)} cases under {input_root}")

    success = 0
    failures: list[tuple[str, str]] = []

    for index, case in enumerate(cases, start=1):
        try:
            metadata = process_case(
                case=case,
                output_root=output_root,
                target_spacing=target_spacing,
                base_crop_size=crop_size,
                crop_mode=args.crop_mode,
                label_mode=args.label_mode,
                transform_type=args.transform_type,
                body_threshold=args.body_threshold,
                crop_margin=args.crop_margin,
                size_divisor=args.size_divisor,
                allow_expand_crop=args.allow_expand_crop,
                skip_registration=args.skip_registration,
                overwrite=args.overwrite,
            )
            success += 1
            print(
                f"[{index}/{len(cases)}] {case.case_id}: done "
                f"(crop={metadata['crop_shape_zyx']}, source={metadata['crop_source']}, "
                f"mask_voxels={metadata['nonzero_mask_voxels']})"
            )
        except Exception as exc:
            failures.append((case.case_id, str(exc)))
            print(f"[{index}/{len(cases)}] {case.case_id}: failed -> {exc}")

    print(f"Finished. Successful cases: {success}, failed cases: {len(failures)}")
    if failures:
        failure_path = output_root / "failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps([{"case_id": case_id, "error": error} for case_id, error in failures], indent=2),
            encoding="utf-8",
        )
        print(f"Failure details were saved to {failure_path}")


if __name__ == "__main__":
    main()
