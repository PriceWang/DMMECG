"""
Author: Guoxin Wang
Date: 2023-08-17 11:06:06
LastEditors: Guoxin Wang
LastEditTime: 2025-02-18 15:17:28
FilePath: /DMMECG/utils/dns_models.py
Description: Models

Copyright (c) 2024 by Guoxin Wang, All Rights Reserved. 
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from timm.models._builder import load_pretrained
from timm.models.vision_transformer import Block


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self, input_dim=480, hidden_dims=[4], activation_layer=nn.ReLU, dropout=0.0
    ):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + hidden_dims[:-1], hidden_dims)
        )
        self.activations = nn.ModuleList(
            activation_layer() for _ in range(self.num_layers - 1)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # print(layer)
            # print(x.shape)
            x = (
                F.dropout(self.activations[i](layer(x)), p=self.dropout)
                if i < self.num_layers - 1
                else layer(x)
            )
        return x


class PatchEmbed1D(nn.Module):
    """1D Signal to Patch Embedding
    Reference: https://github.com/rwightman/pytorch-image-models/blob/main/timm/layers/patch_embed.py
    """

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
        B, C, L = x.shape
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
        mlp_sizes=[128, 128, 1],
        signal_length: int = 480,
        patch_size: int = 32,
        in_chans: int = 1,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: int = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(signal_length, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        self.init_head(embed_dim, mlp_sizes)
        self.apply(self._init_weights)

    def init_head(self, embed_dim, mlp_sizes):
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), MLP(embed_dim, mlp_sizes)
        )

    def forward(self, signals):
        # signals = signals.transpose(1, 2)

        # embed patches
        x = self.patch_embed(signals)

        # add pos embed
        x = x + self.pos_embed
        # apply Transformer blocks
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, signals):
        """
        signals: (N, C, L)
        x: (N, L, patch_size *C)
        """
        N, C, L = signals.shape
        p = self.patch_embed.patch_size
        assert L % p == 0

        l = L // p

        x = signals.reshape(shape=(N, C, l, p))
        x = torch.einsum("nclp->nlpc", x)
        x = x.reshape(shape=(N, l, p * C))
        return x

    def unpatchify(self, x, C: int = 1, channel_last=True):
        """
        x: (N, L, patch_size*C)
        signals: (N, C, L)
        """
        N, L, _ = x.shape
        p = self.patch_embed.patch_size
        x = x.reshape(shape=(N, L, p, C))
        if channel_last:
            signals = x.reshape(shape=(N, L * p, C))
        else:
            x = torch.einsum("nlpc->nclp", x)
            signals = x.reshape(shape=(N, C, L * p))
        return signals


class Router(nn.Module):
    def __init__(
        self,
        n_expert=3,
        n_class=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.router = nn.ModuleList(
            [
                MLP(input_dim=n_class, hidden_dims=[64, 16, n_class])
                for _ in range(n_expert)
            ]
        )

    def forward(self, x):
        x = torch.stack(
            [self.router[n](x[:, n, :]) for n in range(len(self.router))],
            dim=1,
        ).sum(dim=1)
        return x


@register_model
def vit_xxatto_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xxatto_af.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_xatto_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xatto_af.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_atto_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_atto_af.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_tiny_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": (
                    "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_tiny_af.pth"
                )
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_small_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_small_af.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_base_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_base_af.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_large_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {"url": None}
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_huge_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {"url": None}
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_xxatto_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xxatto_id.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_xatto_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_xatto_id.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_atto_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_atto_id.pth"
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_tiny_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {
                "url": (
                    "https://huggingface.co/PriceWang/model/resolve/main/dnsecg/vit_tiny_id.pth"
                    if pretrained_cfg_overlay["task"] == "af_beat"
                    else None
                )
            }
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_small_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {"url": None}
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_base_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {"url": None}
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_large_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {"url": None}
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def vit_huge_id(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
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
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {"url": None}
        load_pretrained(model, pretrained_cfg, strict=False)
    return model


@register_model
def router_af(
    pretrained: bool = False,
    pretrained_cfg: str = None,
    pretrained_cfg_overlay: str = None,
    cache_dir: str = None,
    **kwargs,
) -> nn.Module:
    model = (
        Router(
            n_expert=pretrained_cfg_overlay["n_expert"],
            n_class=pretrained_cfg_overlay["n_class"],
            **kwargs,
        )
        if pretrained_cfg_overlay.get("n_expert", None)
        and pretrained_cfg_overlay.get("n_class", None)
        else Router(**kwargs)
    )
    if pretrained:
        if pretrained_cfg_overlay and pretrained_cfg_overlay.get("path", None):
            pretrained_cfg = {"file": pretrained_cfg_overlay["path"]}
        else:
            pretrained_cfg = {"url": None}
        load_pretrained(model, pretrained_cfg, strict=False)
    return model
