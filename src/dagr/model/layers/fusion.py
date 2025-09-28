import torch
import torch.nn as nn

from dagr.model.layers.spike_cross_attention import CrossAttention


class SpikeCAFR(nn.Module):
    """
    CAFR-like bidirectional fusion with spike-driven cross-attention.

    Inputs
      - rgb: [B,C,H,W]
      - evt: [T,B,C,H,W]

    Output
      - fused rgb-like feature: [B,C_out,H,W]
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_rgb_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_rgb_in = nn.BatchNorm2d(in_channels)

        self.conv_evt_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_evt_in = nn.BatchNorm2d(in_channels)

        self.rgb_centric_attn = CrossAttention(embed_dims=in_channels, num_heads=num_heads)
        self.evt_centric_attn = CrossAttention(embed_dims=in_channels, num_heads=num_heads)

        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    @staticmethod
    def _bchw_to_tbnc(x: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [1,B,N,C]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1).unsqueeze(0).contiguous()
        return x

    @staticmethod
    def _tbnc_to_bchw(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # [T,B,N,C] -> mean over T -> [B,N,C] -> [B,C,H,W]
        x = x.mean(dim=0)  # [B,N,C]
        B, N, C = x.shape
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return x

    @staticmethod
    def _tbchw_to_tbnc(x: torch.Tensor) -> torch.Tensor:
        # [T,B,C,H,W] -> [T,B,N,C]
        T, B, C, H, W = x.shape
        x = x.view(T, B, C, H * W).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, rgb: torch.Tensor, evt: torch.Tensor) -> torch.Tensor:
        # pre-project
        rgb = self.bn_rgb_in(self.conv_rgb_in(rgb))
        T, B, C, H, W = evt.shape
        evt = self.bn_evt_in(self.conv_evt_in(evt.flatten(0, 1))).view(T, B, C, H, W)

        # element-wise interaction (REFusion-like)
        evt_mean = evt.mean(dim=0)  # [B,C,H,W]
        mul = rgb * evt_mean
        rgb_enh = rgb + mul
        evt_enh = evt + mul.unsqueeze(0)

        # spike-driven bidirectional cross-attention
        q_rgb = self._bchw_to_tbnc(rgb_enh)
        kv_evt = self._tbchw_to_tbnc(evt_enh)
        out_rgb = self.rgb_centric_attn(q_rgb, kv_evt, kv_evt)  # [T,B,N,C]
        out_rgb = self._tbnc_to_bchw(out_rgb, H, W)

        q_evt = kv_evt
        kv_rgb = self._bchw_to_tbnc(rgb_enh)
        out_evt = self.evt_centric_attn(q_evt, kv_rgb.expand_as(q_evt), kv_rgb.expand_as(q_evt))  # [T,B,N,C]
        out_evt = self._tbnc_to_bchw(out_evt, H, W)

        fused = out_rgb + out_evt
        fused = self.bn_out(self.conv_out(fused))
        return fused + rgb


