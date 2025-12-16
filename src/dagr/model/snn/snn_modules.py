import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSpike4(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)


class MultiSpike8(nn.Module):
    class quant8(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=8))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 8] = 0
            return grad_input

    def forward(self, x):
        return self.quant8.apply(x)


class MultiSpike2(nn.Module):
    class quant2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=2))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 2] = 0
            return grad_input

    def forward(self, x):
        return self.quant2.apply(x)


class MultiSpike1(nn.Module):
    class quant1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=1))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 1] = 0
            return grad_input

    def forward(self, x):
        return self.quant1.apply(x)


class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        self.act = act
        self.qtrick = MultiSpike4()

    def forward(self, x):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * 0.25 + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SpikeConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeConvWithoutBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)
        self.lif = mem_update()
        self.s = s

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x


class BNAndPadLayer(nn.Module):
    def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepRepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=False, group=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=group, bias=False),
        )
        self.body = nn.Sequential(bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2, act2_layer=nn.Identity, bias=False, kernel_size=3, padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.dwconv2 = nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias)
        self.pwconv3 = SepRepConv(med_channels, dim)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)
        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.bn2(self.dwconv2(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.bn3(self.pwconv3(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        return x


class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(self.lif(x).flatten(0, 1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x


class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = mem_update()

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()
        return x


class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return x


class VoxelGrid:
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert_CHW(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            value = 2 * events['p'] - 1

            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    for tlim in [t0, t0 + 1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim - events['x']).abs()) * (1 - (ylim - events['y']).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid

    def convert_CHW_polarities(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid_pos = self.voxel_grid.clone()
            voxel_grid_neg = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            mask_pos = (events['p'] == 1).bool()
            mask_neg = (events['p'] == 0).bool()
            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    for tlim in [t0, t0 + 1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = (1 - (xlim - events['x']).abs()) * (1 - (ylim - events['y']).abs()) * (1 - (tlim - t_norm).abs())

                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()

                        voxel_grid_pos.put_(index[mask * mask_pos], interp_weights[mask * mask_pos], accumulate=True)
                        voxel_grid_neg.put_(index[mask * mask_neg], interp_weights[mask * mask_neg], accumulate=True)
            voxel_grid = torch.cat((voxel_grid_pos.unsqueeze(dim=1), voxel_grid_neg.unsqueeze(dim=1)), dim=1)

        return voxel_grid


class MS_GetT_Voxel(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 2, T: int = 4, height: int = None, width: int = None):
        super().__init__()
        assert in_channels == 2 and out_channels == 2, "MS_GetT_Voxel expects in/out channels = 2 for event polarities"
        self.T = int(T)
        self.height = height
        self.width = width

    def _voxelize_sample(self, events_dict: dict, H: int, W: int) -> torch.Tensor:
        vg = VoxelGrid((self.T, H, W))
        return vg.convert_CHW_polarities(events_dict)  # [T,2,H,W]

    def forward(self, data_or_tensor):
        # If input is already a BCHW tensor, expand to TBCHW for compatibility. usually this won't happen
        if isinstance(data_or_tensor, torch.Tensor) and data_or_tensor.dim() == 4:
            B, C, H, W = data_or_tensor.shape
            return data_or_tensor.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        data = data_or_tensor

        H_meta = getattr(data, 'meta_height', None)
        W_meta = getattr(data, 'meta_width', None)
        
        H = H_meta if H_meta is not None else self.height
        W = W_meta if W_meta is not None else self.width

        
        assert H is not None and W is not None, "Height/Width must be provided to MS_GetT_Voxel via constructor or data.meta_*"
        H = int(H)
        W = int(W)
        device = data.x.device
        if hasattr(data, 'batch') and data.batch is not None:
            b = data.batch.long()
            B = int(b.max().item()) + 1 if b.numel() > 0 else 1
        else:
            b = torch.zeros((data.pos.shape[0],), dtype=torch.long, device=device)
            B = 1

        out = torch.zeros((self.T, B, 2, H, W), dtype=torch.float32, device=device)

        for bi in range(B):
            mask = (b == bi) if (B > 1 or hasattr(data, 'batch')) else torch.ones((data.pos.shape[0],), dtype=torch.bool, device=device)
            if int(mask.sum().item()) == 0:
                continue

            # Prepare events dict matching VoxelGrid expectations
            x_float = data.pos[mask, 0].to(torch.float32) * (W - 1)
            y_float = data.pos[mask, 1].to(torch.float32) * (H - 1)
            t_float = data.pos[mask, 2].to(torch.float32)
            if t_float.numel() >= 2 and float(t_float[-1].item()) == float(t_float[0].item()):
                t_float = t_float.clone()
                t_float[-1] = t_float[-1] + 1.0
            p_val = (data.x[mask, 0] > 0).to(torch.int64)

            events_dict = {
                'p': p_val,
                't': t_float,
                'x': x_float.to(torch.float32),
                'y': y_float.to(torch.float32),
            }

            voxel = self._voxelize_sample(events_dict, H, W)  # [T,2,H,W]
            out[:, bi] = voxel

        return out

class MS_CancelT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def forward(self, x):
        x = x.mean(0)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, full=False):
        super().__init__()
        self.full = full
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.mlp_ratio = mlp_ratio
        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))
        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))
        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, int(self.mlp_ratio * C), H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x
        return x


class MS_AllConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):
        super().__init__()
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x
        return x


class SpikeSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))


