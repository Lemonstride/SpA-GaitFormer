from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .alignment import pool_three_frame_features, validate_three_to_one_lengths


class TinyFrameEncoder(nn.Module):
    """Small software-test encoder; never used for reported experiments."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class ViTB16FrameEncoder(nn.Module):
    """Original ViT-B/16 architecture with a local-only checkpoint option."""

    def __init__(self, checkpoint: str | None = None) -> None:
        super().__init__()
        from torchvision.models import vit_b_16

        self.network = vit_b_16(weights=None)
        self.output_dim = int(self.network.hidden_dim)
        self.network.heads = nn.Identity()
        if checkpoint:
            path = Path(checkpoint).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"ViT checkpoint does not exist: {path}")
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.network.load_state_dict(state, strict=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class RGBBranch(nn.Module):
    def __init__(self, cfg: dict[str, Any], shared_dim: int) -> None:
        super().__init__()
        backend = cfg.get("backend", "vit_b_16")
        if backend == "vit_b_16":
            self.encoder = ViTB16FrameEncoder(cfg.get("checkpoint"))
        elif backend == "tiny":
            self.encoder = TinyFrameEncoder(shared_dim)
        else:
            raise ValueError(f"Unsupported RGB backend: {backend}")
        self.projection = nn.Linear(self.encoder.output_dim, shared_dim)
        self.train(bool(cfg.get("trainable", True)))
        for parameter in self.parameters():
            parameter.requires_grad = bool(cfg.get("trainable", True))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.size(2) != 3:
            raise ValueError(f"RGB input must be [B,T,3,H,W], got {tuple(rgb.shape)}")
        batch, frames, channels, height, width = rgb.shape
        features = self.encoder(rgb.reshape(batch * frames, channels, height, width))
        features = self.projection(features).reshape(batch, frames, -1)
        return pool_three_frame_features(features)


def load_skeletongait_pp_class(opengait_root: Path) -> type[nn.Module]:
    opengait_package = opengait_root / "opengait"
    source = opengait_package / "modeling" / "models" / "skeletongait++.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official SkeletonGait++ source does not exist: {source}")
    if str(opengait_package) not in sys.path:
        sys.path.insert(0, str(opengait_package))
    import modeling.base_model  # noqa: F401

    if "modeling.models" not in sys.modules:
        models_package = types.ModuleType("modeling.models")
        models_package.__path__ = [str(opengait_package / "modeling" / "models")]
        sys.modules["modeling.models"] = models_package
    module_name = "modeling.models.skeletongait_pp_spa"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official SkeletonGait++ source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].SkeletonGaitPP


class SkeletonGaitPPFrameEncoder(nn.Module):
    """Trainable official SkeletonGait++ P3D backbone with frame-level taps."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_class = load_skeletongait_pp_class(
            Path(cfg.get("opengait_root", "third_party/OpenGait")).expanduser().resolve()
        )
        network = model_class.__new__(model_class)
        nn.Module.__init__(network)
        channel_multiplier = int(cfg.get("channel_multiplier", 2))
        network.build_network(
            {
                "Backbone": {
                    "in_channels": 3,
                    "blocks": list(cfg.get("blocks", [1, 1, 1, 1])),
                    "C": channel_multiplier,
                },
                "SeparateBNNecks": {"class_num": 2},
                "use_emb2": False,
            }
        )
        self.network = network
        self.output_dim = 16 * 128 * channel_multiplier
        checkpoint = cfg.get("checkpoint")
        if checkpoint:
            path = Path(checkpoint).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"SkeletonGait++ checkpoint does not exist: {path}")
            state = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                state = state.get("model", state.get("state_dict", state))
            self.network.load_state_dict(state, strict=False)

    def forward(self, pose_sil: torch.Tensor) -> torch.Tensor:
        if pose_sil.ndim != 5 or pose_sil.shape[2:] != (3, 64, 44):
            raise ValueError(
                "SkeletonGait++ input must be [B,T,3,64,44], "
                f"got {tuple(pose_sil.shape)}"
            )
        pose = pose_sil.transpose(1, 2).contiguous()
        maps = pose[:, :2]
        silhouettes = pose[:, 2].unsqueeze(1)
        map_features = self.network.map_layer1(self.network.map_layer0(maps))
        silhouette_features = self.network.sil_layer1(self.network.sil_layer0(silhouettes))
        features = self.network.fusion(silhouette_features, map_features)
        features = self.network.layer4(self.network.layer3(self.network.layer2(features)))
        batch, channels, frames, height, width = features.shape
        frame_maps = features.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        frame_features = self.network.FCs(self.network.HPP(frame_maps))
        return frame_features.reshape(batch, frames, -1)


class SkeletonFeatureBranch(nn.Module):
    """Encodes pose-silhouette maps or projects cached SkeletonGait++ features."""

    def __init__(self, cfg: dict[str, Any], shared_dim: int) -> None:
        super().__init__()
        self.backend = cfg.get("backend", "frame_features")
        if self.backend == "skeletongait_pp":
            self.encoder: nn.Module | None = SkeletonGaitPPFrameEncoder(cfg)
            feature_dim = int(self.encoder.output_dim)
        elif self.backend == "frame_features":
            self.encoder = None
            feature_dim = int(cfg["feature_dim"])
        else:
            raise ValueError(f"Unsupported skeleton backend: {self.backend}")
        configured_dim = int(cfg.get("feature_dim", feature_dim))
        if configured_dim != feature_dim:
            raise ValueError(
                f"Configured skeleton feature_dim={configured_dim}, encoder emits {feature_dim}"
            )
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, shared_dim),
        )
        trainable = bool(cfg.get("trainable", True))
        for parameter in self.parameters():
            parameter.requires_grad = trainable

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            frame_features = self.encoder(frame_features)
        elif frame_features.ndim != 3:
            raise ValueError(
                "Cached skeleton input must be frame features [B,T,D] exported from "
                f"SkeletonGait++, got {tuple(frame_features.shape)}"
            )
        return pool_three_frame_features(self.projection(frame_features))


class RadarTemporalBranch(nn.Module):
    def __init__(self, cfg: dict[str, Any], shared_dim: int, dropout: float) -> None:
        super().__init__()
        channels = int(cfg.get("cnn_channels", 64))
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, shared_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=int(cfg["transformer_heads"]),
            dim_feedforward=shared_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(cfg.get("transformer_layers", 2)),
            enable_nested_tensor=False,
        )

    def forward(self, rd_maps: torch.Tensor) -> torch.Tensor:
        if rd_maps.ndim != 5 or rd_maps.size(2) != 1:
            raise ValueError(f"RD input must be [B,T,1,H,W], got {tuple(rd_maps.shape)}")
        batch, frames, channels, height, width = rd_maps.shape
        tokens = self.frame_encoder(rd_maps.reshape(batch * frames, channels, height, width))
        return self.temporal_encoder(tokens.reshape(batch, frames, -1))


class SpAGaitformer(nn.Module):
    def __init__(self, config: dict[str, Any], num_classes: int) -> None:
        super().__init__()
        cfg = config["model"]
        shared_dim = int(cfg["shared_dim"])
        dropout = float(cfg.get("dropout", 0.1))
        self.rgb_branch = RGBBranch(cfg["rgb"], shared_dim)
        self.skeleton_branch = SkeletonFeatureBranch(cfg["skeleton"], shared_dim)
        self.radar_branch = RadarTemporalBranch(cfg["radar"], shared_dim, dropout)

        self.modality_embedding = nn.Parameter(torch.empty(3, shared_dim))
        self.cls_token = nn.Parameter(torch.empty(1, 1, shared_dim))
        self.time_projection = nn.Sequential(
            nn.Linear(1, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )
        fusion_cfg = cfg["fusion"]
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=shared_dim,
            nhead=int(fusion_cfg["heads"]),
            dim_feedforward=shared_dim * int(fusion_cfg.get("mlp_ratio", 4)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer,
            num_layers=int(fusion_cfg.get("layers", 3)),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        rgb: torch.Tensor,
        skeleton_features: torch.Tensor,
        rd_maps: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        validate_three_to_one_lengths(rgb.size(1), skeleton_features.size(1), rd_maps.size(1))
        rgb_tokens = self.rgb_branch(rgb)
        skeleton_tokens = self.skeleton_branch(skeleton_features)
        radar_tokens = self.radar_branch(rd_maps)
        modalities = torch.stack([rgb_tokens, skeleton_tokens, radar_tokens], dim=2)

        batch, time_steps, modality_count, dim = modalities.shape
        time = torch.linspace(0.0, 1.0, time_steps, device=modalities.device, dtype=modalities.dtype)
        time_embedding = self.time_projection(time[:, None]).view(1, time_steps, 1, dim)
        modalities = modalities + self.modality_embedding.view(1, 1, modality_count, dim) + time_embedding

        if modality_mask is not None:
            if modality_mask.shape != (batch, modality_count):
                raise ValueError(
                    f"modality_mask must be [B,3], got {tuple(modality_mask.shape)}"
                )
            modalities = modalities * modality_mask[:, None, :, None].to(modalities.dtype)

        tokens = modalities.reshape(batch, time_steps * modality_count, dim)
        cls = self.cls_token.expand(batch, -1, -1)
        fused = self.fusion(torch.cat([cls, tokens], dim=1))
        clip_embedding = fused[:, 0]
        return {
            "logits": self.classifier(clip_embedding),
            "clip_embedding": clip_embedding,
            "rgb_tokens": rgb_tokens,
            "skeleton_tokens": skeleton_tokens,
            "radar_tokens": radar_tokens,
        }
