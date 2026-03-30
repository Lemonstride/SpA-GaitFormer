from __future__ import annotations

import torch
import torch.nn as nn

from spa_gaitformer.config import DataConfig, ModelConfig, TaskConfig


def _build_transformer(
    embed_dim: int,
    num_heads: int,
    depth: int,
    mlp_ratio: float,
    dropout: float,
) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=depth,
        enable_nested_tensor=False,
    )


class RGBEncoder(nn.Module):
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        if data_cfg.image_size % model_cfg.rgb_patch_size != 0:
            raise ValueError("image_size must be divisible by model.rgb_patch_size")

        self.patch_embed = nn.Conv2d(
            3,
            model_cfg.embed_dim,
            kernel_size=model_cfg.rgb_patch_size,
            stride=model_cfg.rgb_patch_size,
        )
        patches_per_side = data_cfg.image_size // model_cfg.rgb_patch_size
        num_patches = patches_per_side * patches_per_side

        self.spatial_pos = nn.Parameter(
            torch.zeros(1, num_patches, model_cfg.embed_dim)
        )
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, data_cfg.num_frames, model_cfg.embed_dim)
        )
        self.spatial_encoder = _build_transformer(
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            depth=model_cfg.rgb_spatial_depth,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
        )
        self.temporal_encoder = _build_transformer(
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            depth=model_cfg.rgb_temporal_depth,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
        )
        self.norm = nn.LayerNorm(model_cfg.embed_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = rgb.shape
        rgb = rgb.reshape(batch * frames, channels, height, width)
        rgb = self.patch_embed(rgb).flatten(2).transpose(1, 2)
        rgb = rgb + self.spatial_pos[:, : rgb.shape[1]]
        rgb = self.spatial_encoder(rgb)
        rgb = rgb.mean(dim=1).reshape(batch, frames, -1)
        rgb = rgb + self.temporal_pos[:, :frames]
        rgb = self.temporal_encoder(rgb)
        return self.norm(rgb)


class SkeletonEncoder(nn.Module):
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.joint_proj = nn.Linear(data_cfg.joint_dim, model_cfg.embed_dim)
        self.joint_pos = nn.Parameter(
            torch.zeros(1, data_cfg.max_joints, model_cfg.embed_dim)
        )
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, data_cfg.num_frames, model_cfg.embed_dim)
        )
        self.joint_encoder = _build_transformer(
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            depth=model_cfg.skeleton_joint_depth,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
        )
        self.temporal_encoder = _build_transformer(
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            depth=model_cfg.skeleton_temporal_depth,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
        )
        self.norm = nn.LayerNorm(model_cfg.embed_dim)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, coords = skeleton.shape
        skeleton = skeleton.reshape(batch * frames, joints, coords)
        skeleton = self.joint_proj(skeleton)
        skeleton = skeleton + self.joint_pos[:, :joints]
        skeleton = self.joint_encoder(skeleton)
        skeleton = skeleton.mean(dim=1).reshape(batch, frames, -1)
        skeleton = skeleton + self.temporal_pos[:, :frames]
        skeleton = self.temporal_encoder(skeleton)
        return self.norm(skeleton)


class DualStreamSpAGaitFormer(nn.Module):
    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        task_cfg: TaskConfig,
    ) -> None:
        super().__init__()
        if model_cfg.input_mode not in {"fusion", "rgb", "skeleton"}:
            raise ValueError("model.input_mode must be one of fusion/rgb/skeleton.")

        self.task_cfg = task_cfg
        self.input_mode = model_cfg.input_mode
        self.rgb_encoder = RGBEncoder(data_cfg, model_cfg)
        self.skeleton_encoder = SkeletonEncoder(data_cfg, model_cfg)

        max_tokens = 1 + (2 * data_cfg.num_frames)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_cfg.embed_dim))
        self.rgb_modality = nn.Parameter(torch.zeros(1, 1, model_cfg.embed_dim))
        self.skeleton_modality = nn.Parameter(torch.zeros(1, 1, model_cfg.embed_dim))
        self.fusion_pos = nn.Parameter(
            torch.zeros(1, max_tokens, model_cfg.embed_dim)
        )
        self.fusion_encoder = _build_transformer(
            embed_dim=model_cfg.embed_dim,
            num_heads=model_cfg.num_heads,
            depth=model_cfg.fusion_depth,
            mlp_ratio=model_cfg.mlp_ratio,
            dropout=model_cfg.dropout,
        )
        self.norm = nn.LayerNorm(model_cfg.embed_dim)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.disease_head = nn.Linear(model_cfg.embed_dim, 1)

        if task_cfg.severity_mode == "classification":
            self.severity_head = nn.Linear(
                model_cfg.embed_dim, task_cfg.severity_num_classes
            )
        elif task_cfg.severity_mode == "regression":
            self.severity_head = nn.Linear(model_cfg.embed_dim, 1)
        else:
            raise ValueError(
                "task.severity_mode must be 'classification' or 'regression'."
            )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.trunc_normal_(parameter, std=0.02)
        nn.init.zeros_(self.disease_head.bias)
        nn.init.zeros_(self.severity_head.bias)

    def forward(self, rgb: torch.Tensor, skeleton: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = []
        if self.input_mode in {"fusion", "rgb"}:
            tokens.append(self.rgb_encoder(rgb) + self.rgb_modality)
        if self.input_mode in {"fusion", "skeleton"}:
            tokens.append(self.skeleton_encoder(skeleton) + self.skeleton_modality)

        batch = tokens[0].shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        fused = torch.cat([cls] + tokens, dim=1)
        fused = fused + self.fusion_pos[:, : fused.shape[1]]
        fused = self.fusion_encoder(fused)
        cls_feature = self.dropout(self.norm(fused[:, 0]))

        outputs = {
            "disease_logits": self.disease_head(cls_feature).squeeze(-1),
            "fusion_feature": cls_feature,
        }
        severity = self.severity_head(cls_feature)
        if self.task_cfg.severity_mode == "regression":
            severity = severity.squeeze(-1)
        outputs["severity_logits"] = severity
        return outputs
