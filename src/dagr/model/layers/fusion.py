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

    def __init__(self, rgb_in_channels: int, evt_in_channels: int, out_channels: int, num_heads: int = 8):
        super().__init__()
        self.rgb_in_channels = rgb_in_channels
        self.evt_in_channels = evt_in_channels
        self.out_channels = out_channels
        self.debug_eval = False
        self._debug_call_count = 0

        self.conv_rgb_in = nn.Conv2d(rgb_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_rgb_in = nn.BatchNorm2d(out_channels)

        self.conv_evt_in = nn.Conv2d(evt_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_evt_in = nn.BatchNorm2d(out_channels)

        self.rgb_centric_attn = CrossAttention(embed_dims=out_channels, num_heads=num_heads)
        self.evt_centric_attn = CrossAttention(embed_dims=out_channels, num_heads=num_heads)

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
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
        # 预投射
        rgb = self.bn_rgb_in(self.conv_rgb_in(rgb))
        T, B, C_evt, H, W = evt.shape
        evt = self.bn_evt_in(self.conv_evt_in(evt.flatten(0, 1))).view(T, B, self.out_channels, H, W)

        # Debug: feature stats before attention/fusion
        if self.debug_eval and self._debug_call_count < 5:
            try:
                def _stat(x: torch.Tensor):
                    return (
                        tuple(x.shape),
                        float(x.mean().detach().cpu()),
                        float(x.std().detach().cpu()),
                        float(x.min().detach().cpu()),
                        float(x.max().detach().cpu()),
                    )
                rgb_s = _stat(rgb)
                # evt mean over time for readable stats
                evt_mean_stat = _stat(evt.mean(dim=0))
                print(f"[FusionDebug] rgb_in: shape={rgb_s[0]}, mean={rgb_s[1]:.4f}, std={rgb_s[2]:.4f}, min={rgb_s[3]:.4f}, max={rgb_s[4]:.4f}", flush=True)
                print(f"[FusionDebug] evt_in: shape={evt.shape}, mean={evt_mean_stat[1]:.4f}, std={evt_mean_stat[2]:.4f}, min={evt_mean_stat[3]:.4f}, max={evt_mean_stat[4]:.4f}", flush=True)
            except Exception as _e:
                print(f"[FusionDebug][WARN] pre-fusion stats failed: {repr(_e)}", flush=True)

        # element-wise interaction (REFusion-like)
        evt_mean = evt.mean(dim=0)  # [B,C,H,W]
        mul = rgb * evt_mean
        rgb_enh = rgb + mul
        evt_enh = evt + mul.unsqueeze(0)

        # spike-driven bidirectional cross-attention
        q_rgb = self._bchw_to_tbnc(rgb_enh)   # [1,B,N,C]
        kv_evt = self._tbchw_to_tbnc(evt_enh) # [T,B,N,C]
        out_rgb = self.rgb_centric_attn(q_rgb, kv_evt, kv_evt)  # [T,B,N,C] (T aligned inside)
        out_rgb = self._tbnc_to_bchw(out_rgb, H, W)

        q_evt = kv_evt                              # [T,B,N,C]
        kv_rgb = self._bchw_to_tbnc(rgb_enh)        # [1,B,N,C]
        out_evt = self.evt_centric_attn(q_evt, kv_rgb, kv_rgb)  # [T,B,N,C] (T aligned inside)
        out_evt = self._tbnc_to_bchw(out_evt, H, W)

        fused = out_rgb + out_evt
        fused = self.bn_out(self.conv_out(fused))
        out = fused + rgb

        if self.debug_eval and self._debug_call_count < 5:
            try:
                delta = (out - rgb).abs().mean().detach().cpu().item()
                def _stat(x: torch.Tensor):
                    return (
                        tuple(x.shape),
                        float(x.mean().detach().cpu()),
                        float(x.std().detach().cpu()),
                        float(x.min().detach().cpu()),
                        float(x.max().detach().cpu()),
                    )
                out_s = _stat(out)
                print(f"[FusionDebug] fused_out: shape={out_s[0]}, mean={out_s[1]:.4f}, std={out_s[2]:.4f}, min={out_s[3]:.4f}, max={out_s[4]:.4f}", flush=True)
                print(f"[FusionDebug] delta_abs_mean={delta:.6f}", flush=True)
            except Exception as _e2:
                print(f"[FusionDebug][WARN] post-fusion stats failed: {repr(_e2)}", flush=True)

        self._debug_call_count += 1
        return out


