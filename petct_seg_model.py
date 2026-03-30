from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from petct_official_segmamba_v2 import (
    OfficialSegMambaV2StagewiseCTEncoder,
    OfficialSegMambaV2UnavailableError,
    StateFeedbackProjector3D,
)


def make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvNormAct3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3d(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class PETEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        blocks = [ResidualConvBlock3d(in_channels, out_channels)]
        blocks.extend(ResidualConvBlock3d(out_channels, out_channels) for _ in range(depth - 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SegMambaBlock3D(nn.Module):
    """
    Self-contained SegMamba-style block used to keep the project runnable without
    external Mamba dependencies. It preserves the expected encoder interface and
    can be replaced later by an official SegMamba block if needed.
    """

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        squeeze_channels = max(hidden_channels // 4, 8)

        self.norm = make_group_norm(channels)
        self.in_proj = nn.Conv3d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels)
        self.axial_d = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=hidden_channels,
        )
        self.axial_h = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 3, 1),
            padding=(0, 1, 0),
            groups=hidden_channels,
        )
        self.axial_w = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            groups=hidden_channels,
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_channels, squeeze_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(squeeze_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=1)
        x = self.dw_conv(x) + self.axial_d(x) + self.axial_h(x) + self.axial_w(x)
        x = F.gelu(x) * torch.sigmoid(gate)
        x = x * self.channel_gate(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return residual + x


class SegMambaStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = ConvNormAct3d(in_channels, out_channels, kernel_size=3)
        self.blocks = nn.Sequential(*[SegMambaBlock3D(out_channels, dropout=dropout) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.blocks(x)


class FallbackSegMambaStyleCTEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        depths: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != 5 or len(depths) != 4:
            raise ValueError(
                "Fallback CT encoder expects five exposed channel scales and four internal stage depths."
            )

        self.stage_channels = tuple(channels)
        self.stem = ResidualConvBlock3d(in_channels, channels[0])
        self.transitions = nn.ModuleList(
            [
                nn.Sequential(
                    DownsampleBlock3D(channels[0], channels[0]),
                    SegMambaStage(channels[0], channels[0], depths[0], dropout=dropout),
                ),
                nn.Sequential(
                    DownsampleBlock3D(channels[0], channels[1]),
                    SegMambaStage(channels[1], channels[1], depths[1], dropout=dropout),
                ),
                nn.Sequential(
                    DownsampleBlock3D(channels[1], channels[2]),
                    SegMambaStage(channels[2], channels[2], depths[2], dropout=dropout),
                ),
                nn.Sequential(
                    DownsampleBlock3D(channels[2], channels[3]),
                    SegMambaStage(channels[3], channels[3], depths[3], dropout=dropout),
                ),
            ]
        )
        self.output_heads = nn.ModuleList(
            [
                ResidualConvBlock3d(channels[0], channels[1]),
                ResidualConvBlock3d(channels[1], channels[2]),
                ResidualConvBlock3d(channels[2], channels[3]),
                ResidualConvBlock3d(channels[3], channels[4]),
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


class DownsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvNormAct3d(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrossModalFusionBlock3D(nn.Module):
    def __init__(self, pet_channels: int, ct_channels: int, fusion_channels: int) -> None:
        super().__init__()
        self.pet_align = nn.Conv3d(pet_channels, fusion_channels, kernel_size=1, bias=False)
        self.ct_align = nn.Conv3d(ct_channels, fusion_channels, kernel_size=1, bias=False)

        self.pet_guidance = nn.Sequential(
            make_group_norm(fusion_channels),
            nn.GELU(),
            nn.Conv3d(fusion_channels, fusion_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            ResidualConvBlock3d(fusion_channels * 2, fusion_channels),
            ResidualConvBlock3d(fusion_channels, fusion_channels),
        )

        self.pet_feedback = nn.Sequential(
            nn.Conv3d(pet_channels + fusion_channels, pet_channels, kernel_size=1, bias=False),
            make_group_norm(pet_channels),
            nn.GELU(),
        )
        self.ct_feedback = nn.Sequential(
            nn.Conv3d(ct_channels + fusion_channels, ct_channels, kernel_size=1, bias=False),
            make_group_norm(ct_channels),
            nn.GELU(),
        )

    def forward(self, pet_feat: torch.Tensor, ct_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pet_aligned = self.pet_align(pet_feat)
        ct_aligned = self.ct_align(ct_feat)

        guide = self.pet_guidance(pet_aligned)
        ct_enhanced = ct_aligned * (1.0 + guide)
        fused = self.fuse(torch.cat([pet_aligned, ct_enhanced], dim=1))

        pet_refined = pet_feat + self.pet_feedback(torch.cat([pet_feat, fused], dim=1))
        ct_refined = ct_feat + self.ct_feedback(torch.cat([ct_feat, fused], dim=1))
        return fused, pet_refined, ct_refined


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pre_conv = ConvNormAct3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.block = nn.Sequential(
            ResidualConvBlock3d(out_channels + skip_channels, out_channels),
            ResidualConvBlock3d(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.pre_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SegmentationDecoder3D(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"Decoder expects 5 scales, got {len(channels)}.")

        self.blocks = nn.ModuleList(
            [
                DecoderBlock3D(channels[4], channels[3], channels[3]),
                DecoderBlock3D(channels[3], channels[2], channels[2]),
                DecoderBlock3D(channels[2], channels[1], channels[1]),
                DecoderBlock3D(channels[1], channels[0], channels[0]),
            ]
        )

    def forward(self, fused_features: Sequence[torch.Tensor]) -> torch.Tensor:
        m1, m2, m3, m4, m5 = fused_features
        x = m5
        x = self.blocks[0](x, m4)
        x = self.blocks[1](x, m3)
        x = self.blocks[2](x, m2)
        x = self.blocks[3](x, m1)
        return x


@dataclass
class ModelOutputs:
    logits: torch.Tensor
    pet_features: list[torch.Tensor]
    ct_features: list[torch.Tensor]
    fused_features: list[torch.Tensor]


class DualModalSegNet3D(nn.Module):
    def __init__(
        self,
        pet_in_channels: int = 1,
        ct_in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        pet_depths: Sequence[int] = (2, 2, 2, 2, 2),
        ct_depths: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.0,
        ct_encoder_type: str = "official_segmamba_v2",
        official_segmamba_path: str | None = None,
        official_segmamba_source: str = "brats23",
        allow_ct_encoder_fallback: bool = True,
    ) -> None:
        super().__init__()

        if len(channels) != 5:
            raise ValueError(f"Exactly five channel scales are required, got {len(channels)}.")
        if len(pet_depths) != 5 or len(ct_depths) != 4:
            raise ValueError("PET depths must define five stages and CT depths must define four official stages.")

        self.channels = tuple(channels)
        self.pet_depths = tuple(pet_depths)
        self.ct_depths = tuple(ct_depths)
        self.ct_encoder_type = ct_encoder_type
        self.official_segmamba_path = official_segmamba_path
        self.official_segmamba_source = official_segmamba_source
        self.allow_ct_encoder_fallback = allow_ct_encoder_fallback

        pet_stage_inputs = [pet_in_channels, channels[1], channels[2], channels[3], channels[4]]

        self.pet_stages = nn.ModuleList(
            [
                PETEncoderStage(pet_stage_inputs[0], channels[0], pet_depths[0]),
                PETEncoderStage(pet_stage_inputs[1], channels[1], pet_depths[1]),
                PETEncoderStage(pet_stage_inputs[2], channels[2], pet_depths[2]),
                PETEncoderStage(pet_stage_inputs[3], channels[3], pet_depths[3]),
                PETEncoderStage(pet_stage_inputs[4], channels[4], pet_depths[4]),
            ]
        )
        self.ct_encoder = self._build_ct_encoder(
            ct_in_channels=ct_in_channels,
            channels=self.channels,
            ct_depths=self.ct_depths,
            dropout=dropout,
        )
        ct_stage_channels = self.ct_encoder.stage_channels

        self.pet_downsamples = nn.ModuleList(
            [
                DownsampleBlock3D(channels[0], channels[1]),
                DownsampleBlock3D(channels[1], channels[2]),
                DownsampleBlock3D(channels[2], channels[3]),
                DownsampleBlock3D(channels[3], channels[4]),
            ]
        )

        self.fusion_blocks = nn.ModuleList(
            [
                CrossModalFusionBlock3D(channels[0], ct_stage_channels[0], channels[0]),
                CrossModalFusionBlock3D(channels[1], ct_stage_channels[1], channels[1]),
                CrossModalFusionBlock3D(channels[2], ct_stage_channels[2], channels[2]),
                CrossModalFusionBlock3D(channels[3], ct_stage_channels[3], channels[3]),
                CrossModalFusionBlock3D(channels[4], ct_stage_channels[4], channels[4]),
            ]
        )

        self.decoder = SegmentationDecoder3D(channels)
        self.seg_head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def _build_ct_encoder(
        self,
        ct_in_channels: int,
        channels: Sequence[int],
        ct_depths: Sequence[int],
        dropout: float,
    ) -> nn.Module:
        if self.ct_encoder_type == "fallback_segmamba_style":
            return FallbackSegMambaStyleCTEncoder(
                in_channels=ct_in_channels,
                channels=channels,
                depths=ct_depths,
                dropout=dropout,
            )

        if self.ct_encoder_type != "official_segmamba_v2":
            raise ValueError(
                "ct_encoder_type must be 'official_segmamba_v2' or 'fallback_segmamba_style', "
                f"got {self.ct_encoder_type!r}."
            )

        try:
            return OfficialSegMambaV2StagewiseCTEncoder(
                in_channels=ct_in_channels,
                channels=tuple(channels),
                depths=tuple(ct_depths),
                official_module_path=self.official_segmamba_path,
                official_source=self.official_segmamba_source,
            )
        except OfficialSegMambaV2UnavailableError as exc:
            if not self.allow_ct_encoder_fallback:
                raise

            warnings.warn(
                "Falling back to the local SegMamba-style CT encoder because the official "
                f"SegMamba-V2 encoder could not be loaded: {exc}",
                stacklevel=2,
            )
            self.ct_encoder_type = "fallback_segmamba_style"
            return FallbackSegMambaStyleCTEncoder(
                in_channels=ct_in_channels,
                channels=channels,
                depths=ct_depths,
                dropout=dropout,
            )

    def forward(
        self,
        pet: torch.Tensor,
        ct: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | ModelOutputs:
        if pet.ndim != 5 or ct.ndim != 5:
            raise ValueError(f"Expected PET/CT tensors of shape [B, C, D, H, W], got {pet.shape} and {ct.shape}.")
        if pet.shape[0] != ct.shape[0] or pet.shape[2:] != ct.shape[2:]:
            raise ValueError(f"PET and CT must share batch size and spatial size, got {pet.shape} and {ct.shape}.")

        pet_input = pet

        pet_features: list[torch.Tensor] = []
        ct_features: list[torch.Tensor] = []
        fused_features: list[torch.Tensor] = []
        ct_state: torch.Tensor | None = None

        for idx in range(len(self.channels)):
            pet_feature = self.pet_stages[idx](pet_input)

            if idx == 0:
                ct_feature, ct_latent = self.ct_encoder.stem_stage(ct)
            else:
                if ct_state is None:
                    raise RuntimeError("CT encoder state was not initialized before transition stage execution.")
                ct_feature, ct_latent = self.ct_encoder.transition_stage(ct_state, idx)

            fused_feature, pet_refined, ct_refined = self.fusion_blocks[idx](pet_feature, ct_feature)

            pet_features.append(pet_feature)
            ct_features.append(ct_feature)
            fused_features.append(fused_feature)

            if idx < len(self.channels) - 1:
                pet_input = self.pet_downsamples[idx](pet_refined)
                ct_state = self.ct_encoder.feedback_to_next_state(ct_latent, ct_refined, idx)

        decoded = self.decoder(fused_features)
        logits = self.seg_head(decoded)

        if return_features:
            return ModelOutputs(
                logits=logits,
                pet_features=pet_features,
                ct_features=ct_features,
                fused_features=fused_features,
            )
        return logits

    @torch.inference_mode()
    def predict_proba(self, pet: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        logits = self.forward(pet, ct, return_features=False)
        return torch.sigmoid(logits)
