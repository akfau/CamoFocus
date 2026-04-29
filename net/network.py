"""
network.py — Main network definition for:
  "Boundary-Guided Camouflaged Object Detection"
  IEEE Transactions on Circuits and Systems for Video Technology, 2024
  DOI: 10.1109/TCSVT.2024.10483928

Architecture overview:
  - Backbone  : PVT-v2-B4 (pretrained on ImageNet)
  - EAM       : Edge Attention Module  — generates an initial foreground mask
  - Fuser     : Focal-Modulation-based feature fusion guided by the EAM mask
  - CAM       : Context-Aware Module   — progressive multi-scale decoder
  - Output    : three side-output maps (o3, o2, o1) + the EAM mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

from net.pvtv2_encoder import pvt_v2_b4
from torch.nn import GroupNorm


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class ConvBNR(nn.Module):
    """Conv → BN → ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=False
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1×1 Conv → BN → ReLU (channel projection)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Mlp(nn.Module):
    """Two-layer MLP used inside the Focal Modulation block."""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features

        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ---------------------------------------------------------------------------
# Focal Modulation — foreground / background decoupled fusion
# ---------------------------------------------------------------------------

class FocalModulation(nn.Module):
    """
    Focal modulation that separately aggregates foreground (f) and background (b)
    context at multiple focal levels, then combines them.

    Args:
        dim           : feature channel dimension
        focal_level   : number of focal (depthwise) convolution levels
        focal_window  : base kernel size for the first focal level
        focal_factor  : kernel-size growth factor per level
        use_postln    : whether to apply LayerNorm after projection
    """

    def __init__(self, dim, proj_drop=0.0,
                 focal_level=2, focal_window=7, focal_factor=2,
                 use_postln=False):
        super().__init__()
        self.dim          = dim
        self.focal_level  = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln   = use_postln

        # Learnable blend weights for fg / bg streams
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))

        self.gn  = GroupNorm(dim, dim)
        self.act = nn.ReLU()

        # Projects input to (query, context, gates) for each stream
        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=True)
        # Context aggregator (1×1 conv, shared across streams)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        # Separate focal-level depthwise conv stacks for fg and bg
        self.focal_layers_fg = nn.ModuleList()
        self.focal_layers_bg = nn.ModuleList()

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            shared_cfg  = dict(kernel_size=kernel_size, stride=1,
                               groups=dim, padding=kernel_size // 2, bias=False)
            self.focal_layers_fg.append(
                nn.Sequential(nn.Conv2d(dim, dim, **shared_cfg), nn.ReLU()))
            self.focal_layers_bg.append(
                nn.Sequential(nn.Conv2d(dim, dim, **shared_cfg), nn.ReLU()))

    def forward(self, x, mask):
        """
        Args:
            x    : (B, H, W, C) feature tensor
            mask : (B, H, W, C) foreground probability map (values in [0, 1])
        Returns:
            x_out: (B, H, W, C) fused feature tensor
        """
        # Split into fg / bg streams using the EAM mask
        fg = x * mask
        bg = x * (1.0 - mask)

        B, H, W, C = x.shape

        # Project each stream → query, context, gates
        fg = self.f(fg).permute(0, 3, 1, 2).contiguous()   # (B, 2C+L+1, H, W)
        bg = self.f(bg).permute(0, 3, 1, 2).contiguous()

        q_fg, ctx_fg, gates_fg = torch.split(fg, (C, C, self.focal_level + 1), dim=1)
        q_bg, ctx_bg, gates_bg = torch.split(bg, (C, C, self.focal_level + 1), dim=1)

        # Aggregate multi-level context for each stream
        ctx_agg_fg = ctx_fg.new_zeros(ctx_fg.shape)
        ctx_agg_bg = ctx_bg.new_zeros(ctx_bg.shape)

        for lvl in range(self.focal_level):
            ctx_fg = self.focal_layers_fg[lvl](ctx_fg)
            ctx_agg_fg = ctx_agg_fg + ctx_fg * gates_fg[:, lvl:lvl + 1]

            ctx_bg = self.focal_layers_bg[lvl](ctx_bg)
            ctx_agg_bg = ctx_agg_bg + ctx_bg * gates_bg[:, lvl:lvl + 1]

        # Global context (average-pool residual)
        ctx_global_fg = self.act(ctx_fg.mean(dim=[2, 3], keepdim=True))
        ctx_global_bg = self.act(ctx_bg.mean(dim=[2, 3], keepdim=True))

        ctx_agg_fg = ctx_agg_fg + ctx_global_fg * gates_fg[:, self.focal_level:]
        ctx_agg_bg = ctx_agg_bg + ctx_global_bg * gates_bg[:, self.focal_level:]

        # Modulate queries, normalise, and combine streams
        fg_out = self.gn(self.act(q_fg * self.h(ctx_agg_fg) * self.alpha))
        bg_out = self.gn(self.act(q_bg * self.h(ctx_agg_bg) * self.beta))
        x_out  = fg_out + bg_out                             # (B, C, H, W)

        x_out = x_out.permute(0, 2, 3, 1).contiguous()      # (B, H, W, C)
        if self.use_postln:
            x_out = self.ln(x_out)
        x_out = self.proj_drop(self.proj(x_out))
        return x_out


class FocalModulationBlock(nn.Module):
    """
    Transformer-style block wrapping FocalModulation with
    layer-norm, residual connections, and an MLP.
    """

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 act_layer=nn.ReLU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=7,
                 use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.norm1      = norm_layer(dim)
        self.modulation = FocalModulation(
            dim, focal_window=focal_window,
            focal_level=focal_level, proj_drop=drop
        )
        self.drop_path  = nn.Identity()          # DropPath removed for simplicity
        self.norm2      = norm_layer(dim)
        self.mlp        = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(
                layerscale_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                layerscale_value * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W, mask, mH, mW):
        """
        Args:
            x    : (B, H*W, C) flattened feature
            H, W : spatial dimensions of x
            mask : (B, mH*mW, C) flattened EAM mask (same spatial size after interpolation)
        """
        B, L, C = x.shape
        mB, mL, mC = mask.shape
        shortcut = x

        # Focal modulation
        x    = self.norm1(x).view(B, H, W, C)
        mask = mask.view(mB, mH, mW, mC)
        x    = self.modulation(x, mask).view(B, H * W, C)

        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Fuser: applies FocalModulationBlock to each feature pyramid level
# ---------------------------------------------------------------------------

class Fuser(nn.Module):
    """
    Guides all four feature-pyramid levels with the EAM mask via
    focal modulation, aligning the mask spatially for each level.
    """

    def __init__(self):
        super().__init__()
        self.fmb1 = FocalModulationBlock(64,  mlp_ratio=3)
        self.fmb2 = FocalModulationBlock(128, mlp_ratio=3)
        self.fmb3 = FocalModulationBlock(256, mlp_ratio=3)
        self.fmb4 = FocalModulationBlock(256, mlp_ratio=3)

    def forward(self, x1, x2, x3, x4, mask):
        """
        Args:
            x1..x4 : feature maps at strides 4, 8, 16, 32
            mask   : EAM foreground probability map
        Returns:
            Tuple of four fused feature maps (same shapes as inputs).
        """
        def _fuse(fmb, feat, m):
            b, c, h, w = feat.shape
            m_resized  = F.interpolate(m, size=(h, w), mode='bilinear', align_corners=False)
            out = fmb(
                feat.flatten(2).transpose(1, 2), h, w,
                m_resized.flatten(2).transpose(1, 2), h, w
            )
            return out.transpose(1, 2).reshape(b, c, h, w)

        x1_out = _fuse(self.fmb1, x1, mask)
        x2_out = _fuse(self.fmb2, x2, mask)
        x3_out = _fuse(self.fmb3, x3, mask)
        x4_out = _fuse(self.fmb4, x4, mask)

        return x1_out, x2_out, x3_out, x4_out


# ---------------------------------------------------------------------------
# CAM: Context-Aware Module (multi-scale dilated decoder)
# ---------------------------------------------------------------------------

class CAM(nn.Module):
    """
    Aggregates a lower-resolution high-level feature (hf) into a
    higher-resolution feature (lf) using grouped dilated convolutions
    that capture context at progressively larger receptive fields.

    Args:
        hchannel : channels of the high-level (coarser) input
        channel  : channels of the low-level (finer) input / output
        groups   : number of channel groups for the dilated conv branches
    """

    def __init__(self, hchannel, channel, groups=4):
        super().__init__()
        self.groups   = groups
        g_ch          = channel // groups

        self.merge    = ConvBNR(hchannel + channel, channel, kernel_size=1)
        self.spec     = ConvBNR(channel, g_ch, kernel_size=3, padding=1, groups=groups)
        self.conv3    = ConvBNR(g_ch, g_ch, kernel_size=3, padding=1,  groups=groups)
        self.dconv5   = ConvBNR(g_ch, g_ch, kernel_size=3, padding=4,  dilation=4, groups=groups)
        self.dconv7   = ConvBNR(g_ch, g_ch, kernel_size=3, padding=6,  dilation=6, groups=groups)
        self.dconv9   = ConvBNR(g_ch, g_ch, kernel_size=3, padding=8,  dilation=8, groups=groups)
        self.refine1  = ConvBNR(channel, channel, kernel_size=1)
        self.refine2  = ConvBNR(channel, channel, kernel_size=3, padding=1)

    def forward(self, lf, hf):
        """
        Args:
            lf : lower-level (finer) feature  — (B, channel,  H,  W)
            hf : higher-level (coarser) feature — (B, hchannel, H', W')
        """
        if lf.shape[2:] != hf.shape[2:]:
            hf = F.interpolate(hf, size=lf.shape[2:], mode='bilinear', align_corners=False)

        x     = self.merge(torch.cat([lf, hf], dim=1))
        x_sp  = self.spec(x)

        # Four parallel dilated branches with residual chain
        chunks = torch.chunk(x, self.groups, dim=1)
        b0 = self.conv3 (chunks[0] + x_sp)
        b1 = self.dconv5(chunks[1] + b0)
        b2 = self.dconv7(chunks[2] + b1)
        b3 = self.dconv9(chunks[3] + b2)

        multi_scale = self.refine1(torch.cat([b0, b1, b2, b3], dim=1))
        out         = self.refine2(x + multi_scale)
        return out


# ---------------------------------------------------------------------------
# EAM: Edge Attention Module — initial foreground mask estimation
# ---------------------------------------------------------------------------

class EAM(nn.Module):
    """
    Produces a coarse foreground probability map from the two
    middle-level features (x3, x2) of the backbone.
    This mask is used to guide the Fuser and as an auxiliary output.
    """

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNR(256 + 128, 256, kernel_size=3, padding=1),
            ConvBNR(256,       256, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x3, x2):
        """
        Args:
            x3 : stride-16 feature (B, 256, H/16, W/16)
            x2 : stride-8  feature (B, 128, H/8,  W/8)
        Returns:
            mask: (B, 1, H/8, W/8) foreground probability map
        """
        target_size = x2.shape[2:]
        x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)
        out = self.sigmoid(self.block(torch.cat([x3, x2], dim=1)))
        return out


# ---------------------------------------------------------------------------
# Network: full model
# ---------------------------------------------------------------------------

class Network(nn.Module):
    """
    BGNet — Boundary-Guided Camouflaged Object Detection Network.

    Input  : RGB image  (B, 3, H, W),  H = W = 416 during training
    Output : (o3, o2, o1, mask)
        o3   — prediction from x34  (stride-16 decoder output, upsampled ×16)
        o2   — prediction from x234 (stride-8  decoder output, upsampled ×8)
        o1   — prediction from x1234(stride-4  decoder output, upsampled ×4)
        mask — EAM foreground map   (upsampled to full resolution)

    All outputs are *logits* (apply sigmoid for probability maps),
    except `mask` which is already passed through sigmoid in EAM.
    """

    def __init__(self, pretrained_backbone='./models/pvt_v2_b4.pth'):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = pvt_v2_b4()
        if pretrained_backbone:
            state_dict = torch.load(pretrained_backbone, map_location='cpu')
            # Keep only matching keys (handles minor version mismatches)
            state_dict = {
                k: v for k, v in state_dict.items()
                if k in self.backbone.state_dict()
            }
            self.backbone.load_state_dict(state_dict)
            print(f'[Network] Loaded pretrained backbone: {pretrained_backbone}')

        # ── Channel reduction (backbone → unified dims) ───────────────────
        # PVT-v2-B4 output channels: 64 / 128 / 320 / 512
        self.reduce1 = Conv1x1(64,  64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)

        # ── Core modules ──────────────────────────────────────────────────
        self.eam  = EAM()
        self.fuse = Fuser()

        # Decoder: top-down path via CAM
        self.cam3 = CAM(hchannel=256, channel=256)   # x4 → x3
        self.cam2 = CAM(hchannel=256, channel=128)   # x34 → x2
        self.cam1 = CAM(hchannel=128, channel=64)    # x234 → x1

        # Final 1×1 prediction heads
        self.pred3 = nn.Conv2d(256, 1, kernel_size=1)
        self.pred2 = nn.Conv2d(128, 1, kernel_size=1)
        self.pred1 = nn.Conv2d(64,  1, kernel_size=1)

    def forward(self, x):
        # ── Backbone feature extraction ───────────────────────────────────
        # PVT-v2-B4 returns features from coarse to fine: x4, x3, x2, x1
        x4, x3, x2, x1 = self.backbone(x)

        # ── Channel reduction ─────────────────────────────────────────────
        x1r = self.reduce1(x1)   # (B,  64, H/4,  W/4)
        x2r = self.reduce2(x2)   # (B, 128, H/8,  W/8)
        x3r = self.reduce3(x3)   # (B, 256, H/16, W/16)
        x4r = self.reduce4(x4)   # (B, 256, H/32, W/32)

        # ── EAM: coarse foreground mask ───────────────────────────────────
        mask = self.eam(x3r, x2r)   # (B, 1, H/8, W/8)

        # ── Fuser: mask-guided focal modulation ───────────────────────────
        x1r, x2r, x3r, x4r = self.fuse(x1r, x2r, x3r, x4r, mask)

        # ── Decoder: top-down CAM aggregation ────────────────────────────
        x34   = self.cam3(x3r, x4r)
        x234  = self.cam2(x2r, x34)
        x1234 = self.cam1(x1r, x234)

        # ── Side outputs (upsample to full resolution) ────────────────────
        o3   = F.interpolate(self.pred3(x34),   scale_factor=16, mode='bilinear', align_corners=False)
        o2   = F.interpolate(self.pred2(x234),  scale_factor=8,  mode='bilinear', align_corners=False)
        o1   = F.interpolate(self.pred1(x1234), scale_factor=4,  mode='bilinear', align_corners=False)
        mask = F.interpolate(mask,              scale_factor=8,  mode='bilinear', align_corners=False)

        return o3, o2, o1, mask
