from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


class OfficialSegMambaV2UnavailableError(RuntimeError):
    pass


def resolve_default_official_segmamba_v2_path(source: str = "brats23") -> Path:
    return Path(__file__).resolve().parent / "SegMamba-V2-main" / source / "models_segmamba" / "segmambav2.py"


def _load_official_module(module_path: str | Path) -> object:
    module_path = Path(module_path)
    if not module_path.exists():
        raise OfficialSegMambaV2UnavailableError(f"Official SegMamba-V2 source file was not found: {module_path}")

    module_name = f"petct_official_segmamba_v2_{abs(hash(module_path.resolve()))}"
    cached_module = sys.modules.get(module_name)
    if cached_module is not None:
        return cached_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise OfficialSegMambaV2UnavailableError(f"Failed to load SegMamba-V2 module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown dependency"
        raise OfficialSegMambaV2UnavailableError(
            "Official SegMamba-V2 dependencies are missing. "
            f"While importing {module_path}, Python could not find '{missing_module}'."
        ) from exc
    except Exception as exc:
        raise OfficialSegMambaV2UnavailableError(
            f"Failed to import official SegMamba-V2 from {module_path}: {exc}"
        ) from exc

    sys.modules[module_name] = module
    return module


class StateFeedbackProjector3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OfficialSegMambaV2StagewiseCTEncoder(nn.Module):
    """
    Stagewise adapter around the official SegMamba-V2 implementation.

    We instantiate the official network and reuse its encoder stem, hierarchical
    transition blocks, and per-scale refinement blocks so the CT branch can be
    interleaved with PET-CT fusion at every scale.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        official_module_path: str | Path | None = None,
        official_source: str = "brats23",
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()

        if len(channels) != 5 or len(depths) != 4:
            raise ValueError(
                "Official SegMamba-V2 encoder expects five exposed channel scales "
                "(encoder1-encoder5) and four internal stage depths."
            )

        official_module_path = (
            Path(official_module_path)
            if official_module_path is not None
            else resolve_default_official_segmamba_v2_path(official_source)
        )
        module = _load_official_module(official_module_path)
        hidden_size = channels[-1] if hidden_size is None else hidden_size
        if hidden_size != channels[-1]:
            raise ValueError(
                f"channels[-1] ({channels[-1]}) must match hidden_size ({hidden_size}) "
                "so the exposed fifth CT scale aligns with the decoder."
            )

        official_model = module.SegMamba(
            in_chans=in_channels,
            out_chans=1,
            depths=list(depths),
            feat_size=list(channels[:-1]),
            hidden_size=hidden_size,
        )

        self.stage_channels = tuple(channels)
        self.depths = tuple(depths)
        self.official_module_path = str(official_module_path)

        self.stem = official_model.encoder1
        self.transitions = nn.ModuleList(
            [
                nn.Sequential(official_model.vit.downsample_layers[0], official_model.vit.gscs[0]),
                nn.Sequential(official_model.vit.downsample_layers[1], official_model.vit.gscs[1]),
                nn.Sequential(
                    official_model.vit.downsample_layers[2],
                    official_model.vit.gscs[2],
                    official_model.vit.stages[2],
                ),
                nn.Sequential(
                    official_model.vit.downsample_layers[3],
                    official_model.vit.gscs[3],
                    official_model.vit.stages[3],
                ),
            ]
        )
        self.output_heads = nn.ModuleList(
            [
                official_model.encoder2,
                official_model.encoder3,
                official_model.encoder4,
                official_model.encoder5,
            ]
        )
        self.feedback_to_state = nn.ModuleList(
            [
                StateFeedbackProjector3D(channels[1], channels[0]),
                StateFeedbackProjector3D(channels[2], channels[1]),
                StateFeedbackProjector3D(channels[3], channels[2]),
            ]
        )

    def stem_stage(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.stem(x)
        return feature, feature

    def transition_stage(self, state: torch.Tensor, stage_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 1 <= stage_index <= 4:
            raise ValueError(f"Stage index must be in [1, 4] for transition stages, got {stage_index}.")

        latent = self.transitions[stage_index - 1](state)
        feature = self.output_heads[stage_index - 1](latent)
        return feature, latent

    def feedback_to_next_state(
        self,
        latent: torch.Tensor,
        refined_feature: torch.Tensor,
        stage_index: int,
    ) -> torch.Tensor:
        if stage_index == 0:
            return refined_feature
        if stage_index in (1, 2, 3):
            return latent + self.feedback_to_state[stage_index - 1](refined_feature)
        if stage_index == 4:
            return latent
        raise ValueError(f"Stage index must be in [0, 4], got {stage_index}.")
