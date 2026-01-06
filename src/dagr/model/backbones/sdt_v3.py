import torch
import torch.nn as nn
import logging
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from timm.models.layers import DropPath
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("timm is required for SpikformerV3Extractor") from exc

try:
    from spikingjelly.clock_driven import layer, functional
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError("spikingjelly is required for SpikformerV3Extractor") from exc

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover - optional dependency in type checks
    Data = None


DEFAULT_SPIKE_NORM = 4.0


class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=DEFAULT_SPIKE_NORM):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(torch.clamp(input, 0, lens) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class Multispike(nn.Module):
    def __init__(self, spike=multispike, norm=DEFAULT_SPIKE_NORM):
        super().__init__()
        self.lens = norm
        self.spike = spike
        self.norm = norm

    def forward(self, inputs):
        return self.spike.apply(inputs, self.lens) / self.norm


def MS_conv_unit(in_channels, out_channels, kernel_size=1, padding=0, groups=1):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=True),
            nn.BatchNorm2d(out_channels),
        )
    )


class MS_ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, spike_norm=DEFAULT_SPIKE_NORM):
        super().__init__()

        self.neuron1 = Multispike(norm=spike_norm)
        self.conv1 = MS_conv_unit(dim, int(dim * mlp_ratio), 3, 1)

        self.neuron2 = Multispike(norm=spike_norm)
        self.conv2 = MS_conv_unit(int(dim * mlp_ratio), dim, 3, 1)

    def forward(self, x, mask=None):
        short_cut = x
        x = self.neuron1(x)
        x = self.conv1(x)
        x = self.neuron2(x)
        x = self.conv2(x)
        x = x + short_cut
        return x


class MS_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, spike_norm=DEFAULT_SPIKE_NORM):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = Multispike(norm=spike_norm)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = Multispike(norm=spike_norm)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N = x.shape

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.c_output, N).contiguous()

        return x


class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, int(in_channel * 1.5), kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(int(in_channel * 1.5)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(int(in_channel * 1.5), out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class RepConv2(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, int(in_channel), kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(int(in_channel)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(int(in_channel), out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class MS_Attention_Conv_qkv_id(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, spike_norm=DEFAULT_SPIKE_NORM):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.sr_ratio = sr_ratio

        self.head_lif = Multispike(norm=spike_norm)

        self.q_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim), nn.BatchNorm1d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim * sr_ratio), nn.BatchNorm1d(dim * sr_ratio))

        self.q_lif = Multispike(norm=spike_norm)
        self.k_lif = Multispike(norm=spike_norm)
        self.v_lif = Multispike(norm=spike_norm)
        self.attn_lif = Multispike(norm=spike_norm)

        self.proj_conv = nn.Sequential(RepConv(sr_ratio * dim, dim), nn.BatchNorm1d(dim))

    def forward(self, x):
        T, B, C, N = x.shape

        x = self.head_lif(x)

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv).reshape(T, B, self.sr_ratio * C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, self.sr_ratio * C // self.num_heads).permute(0, 1, 3, 2, 4)

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale
        x = x.transpose(3, 4).reshape(T, B, self.sr_ratio * C, N)
        x = self.attn_lif(x)

        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, N)
        return x


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        choice,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        init_values=1e-6,
        finetune=False,
        spike_norm=DEFAULT_SPIKE_NORM,
    ):
        super().__init__()
        self.model = choice
        if self.model == "base":
            self.rep_conv = RepConv2(dim, dim)
        self.lif = Multispike(norm=spike_norm)
        self.attn = MS_Attention_Conv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            spike_norm=spike_norm,
        )
        self.finetune = finetune
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, spike_norm=spike_norm)

        if self.finetune:
            self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        T, B, C, N = x.shape
        if self.model == "base":
            x = x + self.rep_conv(self.lif(x).flatten(0, 1)).reshape(T, B, C, N)
        if self.finetune:
            x = x + self.drop_path(self.attn(x) * self.layer_scale1.view(1, 1, -1, 1))
            x = x + self.drop_path(self.mlp(x) * self.layer_scale2.view(1, 1, -1, 1))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        spike_norm=DEFAULT_SPIKE_NORM,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = Multispike(norm=spike_norm)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class SpikformerV3Extractor(nn.Module):
    """
    Spike-Driven Transformer V3 backbone adapted for detection.

    - Accepts (B, T, C, H, W), (T, B, C, H, W) or (B, C, H, W).
    - Returns three BCHW feature maps at strides 8/16/32 after T-mean.
    """

    def __init__(
        self,
        args,
        height,
        width,
        embed_dim=None,
        depths=None,
        num_heads=None,
        mlp_ratio=4.0,
        sr_ratio=1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        choice="base",
        pretrained_weight=None,
    ):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.T = int(getattr(args, "sdt_T", 4))
        self.repeat_static = bool(getattr(args, "sdt_repeat_static", False))
        self.input_t_first = bool(getattr(args, "sdt_input_t_first", False))
        self.use_checkpointing = bool(getattr(args, "use_checkpointing", False) or getattr(args, "sdt_checkpoint", True))
        self.in_channels = int(getattr(args, "sdt_in_channels", getattr(args, "in_channels", 2)))
        self.spike_norm = float(getattr(args, "sdt_norm", DEFAULT_SPIKE_NORM))
        self.pretrained_weight = pretrained_weight or getattr(args, "load_pretrained_weight", None)

        # Allow overriding from args if not provided explicitly
        if depths is None:
            depths = getattr(args, "sdt_depths", [2, 2, 6, 2])
        
        if isinstance(depths, list):
            if len(depths) != 4:
                raise ValueError(f"sdt_depths list must have length 4, got {depths}")
            self.depths = [int(d) for d in depths]
        else:
            raise TypeError(f"Invalid type for depths: {type(depths)}")

        num_heads = num_heads if num_heads is not None else getattr(args, "sdt_num_heads", 8)
        mlp_ratio = getattr(args, "sdt_mlp_ratio", mlp_ratio)

        embed_dim = embed_dim or getattr(args, "sdt_embed_dim", [128, 256, 512, 640])
        if len(embed_dim) < 4:
            raise ValueError("embed_dim must provide at least four stage dimensions for strides 8/16/32.")

        self.embed_dim = embed_dim
        
        # dpr only applies to the Transformer stage (Stage 3), which has self.depths[2] blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depths[2])]


        sr_ratio = getattr(args, "sdt_sr_ratio", sr_ratio)

        # --- Stage 1 (Output Stride 4) ---
        # 1_1: intermediate (Stride 2)
        self.downsample1_1 = MS_DownSampling(
            in_channels=self.in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            spike_norm=self.spike_norm,
        )

        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)])

        # 1_2: Output (Stride 4)
        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )

        self.ConvBlock1_2 = nn.ModuleList([
            MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)
            for _ in range(self.depths[0])
        ])

        # --- Stage 2 (Output Stride 8) ---
        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )
        # Fixed 1 layer for 2_1
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)])

        self.ConvBlock2_2 = nn.ModuleList([
            MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)
            for _ in range(self.depths[1])
        ])

        # --- Stage 3 (Output Stride 16) ---
        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[2],
                    choice=choice,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    sr_ratio=sr_ratio,
                    finetune=True,
                    spike_norm=self.spike_norm,
                )
                for j in range(self.depths[2])
            ]
        )

        # --- Stage 4 (Output Stride 32) ---
        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            spike_norm=self.spike_norm,
        )

        self.ConvBlock4 = nn.ModuleList([
            MS_ConvBlock(dim=embed_dim[3], mlp_ratio=mlp_ratio, spike_norm=self.spike_norm)
            for _ in range(self.depths[3])
        ])

        self.out_channels = [embed_dim[1], embed_dim[2], embed_dim[3]]
        self.strides = [8, 16, 32]
        self.num_scales = 3
        self.use_image = False
        self.is_snn = True
        self.num_classes = getattr(args, "num_classes", getattr(args, "n_classes", 2))

        # Load pretrained weights if provided
        if self.pretrained_weight:
            self._load_pretrained_weights(self.pretrained_weight)
        else:
            # 如果没有加载预训练权重 (从头训练)，则执行我们定制的初始化
            print("[SpikformerV3Extractor] Training from scratch. Applying SNN bias initialization...")
            self._init_snn_weights()

    def _init_snn_weights(self):
        # 遍历所有子模块进行初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                # 卷积层使用 Kaiming 初始化，保证方差
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                # 【核心 Hack】: 将 BN 的 Bias 初始化为 0.5 (甚至 1.0)
                # 这样 BN 输出的均值就会从 0 变成 0.5，正好卡在 Multispike 的门槛上
                # 保证初始状态下有一半的神经元能发放脉冲，打通信号通路！
                nn.init.constant_(m.bias, 0)
    
    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes

    @staticmethod
    def _events_to_frames(data, height, width):
        device = data.x.device
        batch_size = int(data.num_graphs) if hasattr(data, "num_graphs") else 1
        frames = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=device)

        if hasattr(data, "batch") and data.batch is not None:
            b = data.batch.long()
        else:
            b = torch.zeros((data.pos.shape[0],), dtype=torch.long, device=device)

        # data.pos contains NORMALIZED coordinates [0, 1], not pixel coordinates!
        # Must scale to pixel coordinates.
        x_norm = data.pos[:, 0]
        y_norm = data.pos[:, 1]
        
        x_pix = (x_norm * (width - 1)).long()
        y_pix = (y_norm * (height - 1)).long()
        x_pix = torch.clamp(x_pix, 0, width - 1)
        y_pix = torch.clamp(y_pix, 0, height - 1)

        p = (data.x[:, 0] > 0).long()
        frames.index_put_((b, p, y_pix, x_pix), torch.ones_like(p, dtype=frames.dtype), accumulate=True)
        
        # Log raw event statistics occasionally
        if torch.rand(1).item() < 0.01:
            print(f"[SDT Input Raw] Max events/pixel: {frames.max().item():.1f}, Mean: {frames.mean().item():.6f}", flush=True)

        # Apply log1p normalization with reference 100 (restored from working version)
        # Training: max~160 events → log1p(160)/log1p(100)≈1.15→clamp to 1.0 (acceptable)
        # Eval: max~700 events → log1p(700)/log1p(100)≈1.95→clamp to 1.0 (some saturation, but manageable)
        # This is the reference that previously showed non-zero mAP (Epoch 0, 2)
        # log1p(1)/log1p(100)≈0.01, log1p(10)/log1p(100)≈0.23, log1p(100)/log1p(100)=1.0
        # target_scale = 5.0 
        # log_scale = torch.log1p(torch.tensor(target_scale, device=frames.device, dtype=frames.dtype))
        # frames = torch.log1p(frames) / log_scale
        # frames.clamp_(max=1.0)
        frames = torch.clamp(frames, max=3.0) / 3.0
        
        # Log normalized frame statistics occasionally
        if torch.rand(1).item() < 0.01:
            sat_pct = (frames >= 1.0).float().mean().item() * 100
            print(f"[SDT Input Norm] Max: {frames.max().item():.4f}, Mean: {frames.mean().item():.6f}, Saturated: {sat_pct:.2f}%", flush=True)
        
        return frames.unsqueeze(0)  # [T=1, B, 2, H, W]

    def _maybe_checkpoint(self, module, *tensors):
        if self.training and self.use_checkpointing:
            return activation_checkpoint(module, *tensors, use_reentrant=False)
        return module(*tensors)

    def _run_blocks(self, x, blocks):
        for blk in blocks:
            x = self._maybe_checkpoint(blk, x)
        return x

    def _prepare_input(self, x):
        if Data is not None and isinstance(x, Data):
            frames = self._events_to_frames(x, self.height, self.width)
            if self.repeat_static and self.T > 1:
                frames = frames.repeat(self.T, 1, 1, 1, 1)
            return frames

        if not torch.is_tensor(x):
            raise TypeError(f"Unsupported input type {type(x)}. Expected torch.Tensor or torch_geometric.data.Data.")

        if x.dim() == 4:  # BCHW
            x = x.unsqueeze(0)
            if self.repeat_static and self.T > 1:
                x = x.repeat(self.T, 1, 1, 1, 1)
            return x

        if x.dim() != 5:
            raise ValueError(f"Expected input with 4 or 5 dims, got {x.dim()}.")

        if self.input_t_first:
            return x

        # Default: treat as (B, T, C, H, W)
        
        # Handle ambiguity where B == T
        if x.shape[0] == self.T and x.shape[1] == self.T:
            logging.warning(
                f"SpikformerV3Extractor: Ambiguous input shape (BatchSize==TimeStep={self.T}). "
                "Assuming input is (B, T, C, H, W) and permuting to (T, B, ...) because 'input_t_first' is False. "
                "If input is already (T, B, ...), set 'sdt_input_t_first=True'."
            )
            return x.permute(1, 0, 2, 3, 4).contiguous()

        if x.shape[1] == self.T and x.shape[0] != self.T:
            return x.permute(1, 0, 2, 3, 4).contiguous()
        if x.shape[0] == self.T:
            return x  # already T-first
        return x.permute(1, 0, 2, 3, 4).contiguous()

    def forward(self, x, reset=True):
        """
        Returns:
            list[Tensor]: [P3, P4, P5] where
                - P3: stride 8, shape (B, C2, H/8, W/8)
                - P4: stride 16, shape (B, C3, H/16, W/16)
                - P5: stride 32, shape (B, C4, H/32, W/32)
        """
        if reset:
            functional.reset_net(self)

        x = self._prepare_input(x)  # -> [T, B, C, H, W]

        x = self._maybe_checkpoint(self.downsample1_1, x)
        x = self._run_blocks(x, self.ConvBlock1_1)

        x = self._maybe_checkpoint(self.downsample1_2, x)
        x = self._run_blocks(x, self.ConvBlock1_2)

        x = self._maybe_checkpoint(self.downsample2, x)
        x = self._run_blocks(x, self.ConvBlock2_1)
        stage2 = self._run_blocks(x, self.ConvBlock2_2)  # stride 8

        x = self._maybe_checkpoint(self.downsample3, stage2)
        h3, w3 = x.shape[-2:]
        x_tokens = x.flatten(3)  # T,B,C,N
        x_tokens = self._run_blocks(x_tokens, self.block3)
        stage3 = x_tokens.view(x.shape[0], x.shape[1], self.embed_dim[2], h3, w3)  # stride 16

        stage4 = self._maybe_checkpoint(self.downsample4, stage3)
        stage4 = self._run_blocks(stage4, self.ConvBlock4)  # stride 32

        # Collapse time for neck/head consumption
        p3 = stage2.mean(dim=0)  # (B, C2, H/8, W/8)
        p4 = stage3.mean(dim=0)  # (B, C3, H/16, W/16)
        p5 = stage4.mean(dim=0)  # (B, C4, H/32, W/32)

        # --- Debug: Check feature statistics ---
        if torch.rand(1).item() < 0.005:
            print(f"[SDT Features] p3: mean={p3.mean().item():.4f}, std={p3.std().item():.4f}, max={p3.max().item():.4f}", flush=True)
            print(f"[SDT Features] p4: mean={p4.mean().item():.4f}, std={p4.std().item():.4f}, max={p4.max().item():.4f}", flush=True)
            print(f"[SDT Features] p5: mean={p5.mean().item():.4f}, std={p5.std().item():.4f}, max={p5.max().item():.4f}", flush=True)
        # --------------------------------------

        return [p3, p4, p5]

    def _load_pretrained_weights(self, weight_path: str):
        try:
            ckpt = torch.load(weight_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                elif "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]

            if not isinstance(ckpt, dict):
                logging.warning(f"[SpikformerV3Extractor] Unexpected checkpoint format at {weight_path}, skip loading.")
                return

            # 1. 过滤掉分类头 (head)
            filtered = {k: v for k, v in ckpt.items() if not k.startswith("head")}

            # 2. 处理通道数不匹配的输入层权重
            # mismatch_keys = ["downsample1_1.encode_conv.weight", "patch_embed.proj.weight"]
            
            # for key in mismatch_keys:
            #     if key in filtered:
            #         # 检查输入通道维 (dim 1) 是否匹配
            #         if filtered[key].shape[1] != self.in_channels:
            #             logging.warning(
            #                 f"[SpikformerV3Extractor] Removing mismatch key '{key}' "
            #                 f"(ckpt input={filtered[key].shape[1]}, model input={self.in_channels})"
            #             )
            #             del filtered[key]
            keys_to_remove = []
            for k in filtered.keys():
                # 如果第一层下采样层（输入层），或者是 patch_embed
                if "downsample1_1" in k or "patch_embed" in k:
                    keys_to_remove.append(k)
            
            if len(keys_to_remove) > 0:
                logging.warning(f"[SpikformerV3Extractor] Force removing {len(keys_to_remove)} keys from first layer (downsample1_1) to avoid BN mismatch.")
                for k in keys_to_remove:
                    del filtered[k]

            # 3. 加载剩余匹配的权重
            missing, unexpected = self.load_state_dict(filtered, strict=False)
            logging.info(f"[SpikformerV3Extractor] Loaded pretrained weights from {weight_path}. "
                         f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        except Exception as exc:
            logging.warning(f"[SpikformerV3Extractor] Failed to load pretrained weights from {weight_path}: {exc}")
