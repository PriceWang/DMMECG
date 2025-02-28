"""
Author: Guoxin Wang
Date: 2023-08-17 11:06:06
LastEditors: Guoxin Wang
LastEditTime: 2025-02-28 16:28:43
FilePath: /DMMECG/utils/dns_models.py
Description: Models

Copyright (c) 2024 by Guoxin Wang, All Rights Reserved. 
"""

from functools import partial
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model, register_model
from timm.models._builder import load_pretrained
from timm.models.vision_transformer import Block


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 480,
        hidden_dims: Union[list, int] = 4,
        activation_layer: nn.Module = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dims[:-1], hidden_dims)
        )
        self.activations = nn.ModuleList(
            activation_layer() for _ in range(self.num_layers - 1)
        )

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = (
                F.dropout(self.activations[i](layer(x)), p=self.dropout)
                if i < self.num_layers - 1
                else layer(x)
            )
        return x


class PatchEmbed1D(nn.Module):
    def __init__(
        self,
        signal_length: int = 480,
        patch_size: int = 32,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: nn.Module = None,
        channel_last: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.grid_size = signal_length // patch_size
        self.num_patches = self.grid_size
        self.channel_last = channel_last
        self.proj = nn.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor):
        _, _, L = x.shape
        assert (
            L == self.signal_length
        ), f"Input signal length ({L}) doesn't match model ({self.signal_length})."
        x = self.proj(x)
        if self.channel_last:
            x = x.transpose(1, 2)  # BCL -> BLC
        x = self.norm(x)
        return x


class ViT1D(nn.Module):
    def __init__(
        self,
        mlp_sizes: Union[list, int] = 4,
        signal_length: int = 480,
        patch_size: int = 32,
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_embed = PatchEmbed1D(signal_length, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.init_head(embed_dim, mlp_sizes)
        self.apply(self._init_weights)

    def init_head(self, embed_dim: int, mlp_sizes: Union[list, int]):
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), MLP(embed_dim, mlp_sizes)
        )

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x.transpose(1, 2))
        return x

    def freeze_backbone(self):
        for _, p in self.named_parameters():
            p.requires_grad = False
        for _, p in self.head.named_parameters():
            p.requires_grad = True

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x: torch.Tensor):
        N, C, L = x.shape
        p = self.patch_embed.patch_size
        assert L % p == 0
        l = L // p
        x = x.reshape(shape=(N, C, l, p))
        x = torch.einsum("nclp->nlpc", x)
        x = x.reshape(shape=(N, l, p * C))
        return x

    def unpatchify(self, x: torch.Tensor, C: int = 1, channel_last: bool = True):
        N, L, _ = x.shape
        p = self.patch_embed.patch_size
        x = x.reshape(shape=(N, L, p, C))
        if channel_last:
            signals = x.reshape(shape=(N, L * p, C))
        else:
            x = torch.einsum("nlpc->nclp", x)
            signals = x.reshape(shape=(N, C, L * p))
        return signals


@register_model
def vit_xxatto_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=96,
        depth=1,
        num_heads=2,
        mlp_ratio=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xxatto_af.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_xatto_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=96,
        depth=6,
        num_heads=2,
        mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xatto_af.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_atto_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=96,
        depth=12,
        num_heads=2,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_atto_af.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_tiny_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_tiny_af.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_small_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_small_af.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_base_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_base_af.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_large_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = None
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_huge_af(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[4],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = None
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_xxatto_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=96,
        depth=1,
        num_heads=2,
        mlp_ratio=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xxatto_id.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_xatto_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=96,
        depth=6,
        num_heads=2,
        mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xatto_id.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_atto_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=96,
        depth=12,
        num_heads=2,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_atto_id.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_tiny_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = (
                "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_tiny_id.pth"
            )
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_small_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = None
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_base_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = None
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_large_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = None
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_huge_id(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    model = ViT1D(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[90],
        **kwargs,
    )
    if pretrained:
        if pretrained_cfg_overlay.get("path", None):
            pretrained_cfg["file"] = pretrained_cfg_overlay["path"]
        else:
            pretrained_cfg["url"] = None
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def router(
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict[str, Any]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if pretrained_cfg is None:
        pretrained_cfg = {}
    if pretrained_cfg_overlay is None:
        pretrained_cfg_overlay = {}
    n_expert = (
        pretrained_cfg_overlay["n_expert"]
        if pretrained_cfg_overlay.get("n_expert", None)
        else 3
    )
    n_class = (
        pretrained_cfg_overlay["n_class"]
        if pretrained_cfg_overlay.get("n_class", None)
        else 4
    )
    model = ViT1D(
        embed_dim=96,
        depth=1,
        num_heads=2,
        mlp_ratio=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mlp_sizes=[n_expert * n_class],
        **kwargs,
    )
    checkpoint = create_model(
        (
            pretrained_cfg_overlay["router"]
            if pretrained_cfg_overlay.get("router", None)
            else "vit_xxatto_af"
        ),
        pretrained=pretrained,
    ).state_dict()
    remove_keys = []
    for k in checkpoint.keys():
        if k.startswith("head"):
            remove_keys.append(k)
    for k in remove_keys:
        del checkpoint[k]
    model.load_state_dict(checkpoint, strict=False)
    return model
