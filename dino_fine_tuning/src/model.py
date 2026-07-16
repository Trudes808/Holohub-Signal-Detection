"""DINOv3 ViT-B/16 backbone + lightweight multi-scale segmentation head.

Binary signal/noise segmentation. Two adaptation modes:
  - "frozen":    backbone frozen (eval), only the decoder head trains.
  - "ft_lastN":  additionally unfreeze the last N transformer blocks + final norm.

Input to forward(): float tensor [B, 1, H, W] in [0,1] (grayscale spectrogram).
It is repeated to 3 channels and ImageNet-normalized (matching the deployed
detector's imagenet_mean/std) before the backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import dinov3.hub.backbones as B

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _conv_gn(cin, cout, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.GroupNorm(min(32, cout), cout),
        nn.GELU(),
    )


class SegHead(nn.Module):
    """Fuse 4 ViT feature maps (B,768,h,w) -> full-res 1-channel logits."""

    def __init__(self, embed_dim=768, n_layers=4, proj=128):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, proj, 1), nn.GELU()) for _ in range(n_layers)
        ])
        c = proj * n_layers
        self.dec = nn.ModuleList([
            _conv_gn(c, 256), _conv_gn(256, 128), _conv_gn(128, 64), _conv_gn(64, 32),
        ])
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, feats: list[torch.Tensor], out_hw):
        x = torch.cat([p(f) for p, f in zip(self.proj, feats)], dim=1)  # B, proj*L, h, w
        H, W = out_hw
        # progressive x2 upsampling (16x64 -> ... -> 256x1024)
        for i, blk in enumerate(self.dec):
            x = blk(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.out(x)  # B,1,H,W


class DinoSegmenter(nn.Module):
    def __init__(self, weights_path: str, feat_layers=(2, 5, 8, 11),
                 mode="frozen", unfreeze_last_n=4):
        super().__init__()
        self.backbone = B.dinov3_vitb16(pretrained=True, weights=weights_path)
        self.feat_layers = list(feat_layers)
        self.mode = mode
        self.unfreeze_last_n = unfreeze_last_n
        self.head = SegHead(embed_dim=768, n_layers=len(feat_layers))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self._configure_freeze()

    def _configure_freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.trainable_backbone = []
        if self.mode == "ft_lastN" and self.unfreeze_last_n > 0:
            n = self.unfreeze_last_n
            for blk in self.backbone.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
                self.trainable_backbone += list(blk.parameters())
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)
            self.trainable_backbone += list(self.backbone.norm.parameters())

    def train(self, mode=True):
        """Keep frozen backbone in eval; put unfrozen blocks in train."""
        super().train(mode)
        self.backbone.eval()  # frozen parts: no stochasticity
        if mode and self.mode == "ft_lastN" and self.unfreeze_last_n > 0:
            for blk in self.backbone.blocks[-self.unfreeze_last_n:]:
                blk.train()
            self.backbone.norm.train()
        return self

    def _prep(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        xn = self._prep(x)
        grad_ctx = torch.enable_grad() if (self.training and self.trainable_backbone) else torch.no_grad()
        with grad_ctx:
            feats = self.backbone.get_intermediate_layers(
                xn, n=self.feat_layers, reshape=True, norm=True)
        feats = [f.float() for f in feats]
        return self.head(list(feats), (H, W))

    def param_groups(self, lr_head, lr_backbone):
        groups = [{"params": self.head.parameters(), "lr": lr_head}]
        if self.trainable_backbone:
            groups.append({"params": self.trainable_backbone, "lr": lr_backbone})
        return groups


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class DiceBCELoss(nn.Module):
    """BCE-with-logits + soft Dice. pos_weight handles signal/noise imbalance."""

    def __init__(self, pos_weight=1.0, dice_w=1.0, bce_w=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.dice_w, self.bce_w = dice_w, bce_w

    def forward(self, logits, target):
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = 1 - ((2 * inter + 1.0) / (denom + 1.0)).mean()
        return self.bce_w * bce + self.dice_w * dice
