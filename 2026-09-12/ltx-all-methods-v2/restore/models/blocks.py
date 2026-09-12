"""Shared temporal NAF / residual-fusion building blocks; original arithmetic."""

import torch
from torch import nn


class LayerNorm3d(nn.Module):
    """Channel-wise LayerNorm for B,C,T,H,W tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class SpatialPixelShuffle(nn.Module):
    """PixelShuffle on H/W while preserving T."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        r = self.scale
        if c % (r * r):
            raise ValueError(f"Channel count {c} is not divisible by scale^2={r*r}")
        x = x.reshape(b, c // (r * r), r, r, t, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        return x.reshape(b, c // (r * r), t, h * r, w * r)


class TemporalNAFBlock(nn.Module):
    """NAFBlock with an identity-initialized temporal depthwise convolution."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.conv1 = nn.Conv3d(channels, dw_channels, 1)
        self.conv2 = nn.Conv3d(
            dw_channels,
            dw_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=dw_channels,
        )
        self.temporal_conv = nn.Conv3d(
            dw_channels,
            dw_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=dw_channels,
        )
        self.conv3 = nn.Conv3d(dw_channels // 2, channels, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.sg = SimpleGate()

        self.conv4 = nn.Conv3d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv3d(ffn_channels // 2, channels, 1)
        self.norm1 = LayerNorm3d(channels)
        self.norm2 = LayerNorm3d(channels)
        self.dropout1 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.reset_temporal_identity()

    def reset_temporal_identity(self) -> None:
        with torch.no_grad():
            self.temporal_conv.weight.zero_()
            self.temporal_conv.weight[:, 0, 1, 0, 0] = 1.0
            if self.temporal_conv.bias is not None:
                self.temporal_conv.bias.zero_()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.norm1(inp))
        x = self.temporal_conv(self.conv2(x))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.dropout1(self.conv3(x))
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.dropout2(self.conv5(x))
        return y + x * self.gamma


class TemporalResidualFusion(nn.Module):
    """Fuse bidirectional temporal residuals without downsampling time.

    The block sees the current, previous and following features together with
    both signed neighbor residuals.  Its learned residual scale controls
    how much temporal information is added.
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.dilation = max(1, int(dilation))
        self.norm = LayerNorm3d(channels)
        self.project_in = nn.Conv3d(channels * 5, channels * 2, 1)
        self.spatial = nn.Conv3d(
            channels * 2,
            channels * 2,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels * 2,
        )
        self.gate = SimpleGate()
        self.project_out = nn.Conv3d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm(inp)
        distance = min(self.dilation, max(1, x.shape[2] - 1))
        previous = torch.cat(
            (x[:, :, :1].expand(-1, -1, distance, -1, -1), x[:, :, :-distance]),
            dim=2,
        )
        following = torch.cat(
            (x[:, :, distance:], x[:, :, -1:].expand(-1, -1, distance, -1, -1)),
            dim=2,
        )
        context = torch.cat(
            (x, previous, following, x - previous, x - following), dim=1
        )
        update = self.project_out(self.gate(self.spatial(self.project_in(context))))
        return inp + self.beta * update
