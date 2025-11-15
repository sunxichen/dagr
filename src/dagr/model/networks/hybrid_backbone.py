import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from dagr.model.networks.image_backbone import ImageBackbone
from dagr.model.networks.snn_backbone_yaml import SNNBackboneYAMLWrapper
from dagr.model.layers.fusion import SpikeCAFR

from dagr.model.mad_imports.flow_models.model import EVFlowNet
from dagr.model.mad_imports.utils.iwe import compute_pol_iwe, get_interpolation, interpolate, purge_unfeasible
from dagr.model.mad_imports.utils.utils_encoding import events_to_voxel, events_to_image
# from dagr.model.mad_imports.dataloader.basedataset import (
#     create_list_encoding, 
#     create_polarity_mask
# )
from dagr.model.mad_imports.dataloader.basedataset import BaseDataLoader
from dagr.model.networks.mad_aux_modules import MADBackbone
from pathlib import Path


class HybridBackbone(nn.Module):
    """
    RGB backbone with progressive fusion from SNN temporal features via SpikeCAFR.

    Exposes:
      - out_channels = [256, 512]
      - strides = [16, 32]
      - get_output_sizes(height,width) compatible with YOLOXHead
    """

    def __init__(self, args, height: int, width: int):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.num_bins_mad = 5 # 来自 mad_representation config

        # RGB backbone using minimal ImageBackbone; will run image path and expose 4 stages
        args_local = args
        args_local.use_image = True
        self.rgb = ImageBackbone(args_local, height=height, width=width)

        # SNN backbone (temporal features)
        yaml_path = getattr(args, 'snn_yaml_path', 'dagr/src/dagr/cfg/snn_yolov8.yaml')
        scale = getattr(args, 'snn_scale', 's')
        self.snn = SNNBackboneYAMLWrapper(args, height=height, width=width, yaml_path=yaml_path, scale=scale)

        # derive channel dimensions from ImageBackbone exposed specs
        c2_ch = self.rgb.feature_channels[0]
        c3_ch = self.rgb.feature_channels[1]
        c4_ch = self.rgb.output_channels[0]
        c5_ch = self.rgb.output_channels[1]

        self.fuse_p2 = SpikeCAFR(rgb_in_channels=c2_ch, evt_in_channels=64, out_channels=c2_ch)
        self.fuse_p3 = SpikeCAFR(rgb_in_channels=c3_ch, evt_in_channels=128, out_channels=c3_ch)
        self.fuse_p4 = SpikeCAFR(rgb_in_channels=c4_ch, evt_in_channels=256, out_channels=c4_ch)
        self.fuse_p5 = SpikeCAFR(rgb_in_channels=c5_ch, evt_in_channels=512, out_channels=c5_ch)

        # --- MAD branch ---
        # 1. EVFlowNet (用于 T_m)
        mad_flow_config = {
            "base_num_channels": 32,
            "kernel_size": 3,
            "mask_output": True,
        }
        self.mad_flow = EVFlowNet(mad_flow_config, num_bins=self.num_bins_mad)
        # --- Modified for 3-branch hybrid backbone (Fused, RGB, MAD) ---
        # 检查是否设置了 --no_load_mad_flow 标志
        if getattr(args, 'no_load_mad_flow', False):
            print("\033[93mWARNING: --no_load_mad_flow is set. EVFlowNet will use random weights.\033[0m")
        else:
            # 这是原有的加载逻辑，现在它只在没有 --no_load_mad_flow 时执行
            if hasattr(args, 'mad_flow_checkpoint') and args.mad_flow_checkpoint:
                if Path(args.mad_flow_checkpoint).exists():
                    print(f"Loading MAD EVFlowNet checkpoint from: {args.mad_flow_checkpoint}")
                    checkpoint = torch.load(args.mad_flow_checkpoint, map_location='cpu')
                    self.mad_flow.load_state_dict(checkpoint)
                else:
                    print(f"\033[91mERROR: --mad_flow_checkpoint path specified but NOT FOUND:\033[0m")
                    print(f"  {args.mad_flow_checkpoint}")
                    print(f"\033[91mPlease provide a valid path or use --no_load_mad_flow to proceed with random weights.\033[0m")
                    # 可以在这里选择是继续还是抛出异常
                    # raise FileNotFoundError(f"EVFlowNet checkpoint not found at {args.mad_flow_checkpoint}")
            else:
                print(f"\033[93mWARNING: --mad_flow_checkpoint not specified.\033[0m")
                print(f"\033[93mMAD EVFlowNet will use random weights. This is not recommended for final training.\033[0m")
        
        self.mad_flow.eval()
        self.mad_flow.requires_grad_(False) # 冻结
        # --- Modified end ---

        # 2. MADBackbone (用于 T_a, T_m -> 特征)
        self.mad_backbone = MADBackbone(t_a_channels=2, t_m_channels=2)

        self.use_checkpointing = getattr(args, 'use_checkpointing', False)

        self.out_channels = [c2_ch, c3_ch, c4_ch, c5_ch]
        self.strides = [4, 8, 16, 32]
        self.num_scales = 4
        self.num_classes = self.snn.num_classes
        self.use_image = True

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes
    
    # --- MAD输入转换辅助函数 ---
    # def _convert_dagr_to_mad_inputs(self, data):
    #     """ 从 dagr Data 对象创建 MAD 分支所需的输入张量 """
    #     batch_size = data.num_graphs
    #     device = data.x.device
    #     H, W = int(data.height[0]), int(data.width[0])
        
    #     voxel_inputs, cnt_inputs, list_inputs, pol_mask_inputs = [], [], [], []
        
    #     for sample_data in data.to_data_list():
    #         H, W = int(sample_data.height[0]), int(sample_data.width[0])

    #         ts = sample_data.pos[:, 2]
            
    #         ps = sample_data.x[:, 0]

    #         xs = torch.clamp(sample_data.pos[:, 0] * (W - 1), 0, W - 1)
    #         ys = torch.clamp(sample_data.pos[:, 1] * (H - 1), 0, H - 1)

    #         # 1. inp_voxel
    #         inp_voxel = events_to_voxel(xs, ys, ts, ps, self.num_bins_mad, sensor_size=(H, W))
    #         voxel_inputs.append(inp_voxel)

    #         # 2. inp_cnt
    #         inp_cnt = self._create_cnt_encoding(xs, ys, ps, sensor_size=(H, W))
    #         cnt_inputs.append(inp_cnt)  

    #         # 3. inp_list
    #         inp_list = BaseDataLoader.create_list_encoding(xs, ys, ts, ps).transpose(1, 0) # [N, 4]
    #         list_inputs.append(inp_list)

    #         # 4. inp_pol_mask
    #         inp_pol_mask = BaseDataLoader.create_polarity_mask(ps).transpose(1, 0) # [N, 2]
    #         pol_mask_inputs.append(inp_pol_mask)
        
    #     # 批处理和填充
    #     batch_inp_voxel = torch.stack(voxel_inputs)
    #     batch_inp_cnt = torch.stack(cnt_inputs) 
    #     batch_inp_list = torch.nn.utils.rnn.pad_sequence(list_inputs, batch_first=True)
    #     batch_inp_pol_mask = torch.nn.utils.rnn.pad_sequence(pol_mask_inputs, batch_first=True)

    #     return batch_inp_voxel, batch_inp_cnt, batch_inp_list, batch_inp_pol_mask
    def _convert_dagr_to_mad_inputs(self, data):
        """ 从 dagr Data 对象创建 MAD 分支所需的输入张量 """
        batch_size = data.num_graphs
        device = data.x.device
        
        voxel_inputs, cnt_inputs, list_inputs, pol_mask_inputs = [], [], [], []

        for sample_data in data.to_data_list():
            H, W = int(sample_data.height[0]), int(sample_data.width[0]) # H=215, W=320
            
            # --- 修正：计算 EVFlowNet 所需的填充尺寸 (16 的倍数) ---
            # EVFlowNet 有 4 个下采样层
            self.mad_H_pad = int(math.ceil(H / 16.0) * 16) # 215 -> 224
            self.mad_W_pad = int(math.ceil(W / 16.0) * 16) # 320 -> 320
            # --- 修正结束 ---
            
            ts = sample_data.pos[:, 2]
            ps = sample_data.x[:, 0]
            xs = torch.clamp(sample_data.pos[:, 0] * (W - 1), 0, W - 1)
            ys = torch.clamp(sample_data.pos[:, 1] * (H - 1), 0, H - 1)

            # 1. inp_voxel (使用 *padding后* 的尺寸)
            inp_voxel = events_to_voxel(xs, ys, ts, ps, self.num_bins_mad, sensor_size=(self.mad_H_pad, self.mad_W_pad))
            voxel_inputs.append(inp_voxel)
            
            # 2. inp_cnt (使用 *padding后* 的尺寸)
            inp_cnt = self._create_cnt_encoding(xs, ys, ps, sensor_size=(self.mad_H_pad, self.mad_W_pad))
            cnt_inputs.append(inp_cnt)
            
            # 3. inp_list (用于 iwe) - iwe.py 不需要 padding
            inp_list = BaseDataLoader.create_list_encoding(xs, ys, ts, ps).transpose(1, 0) # [N, 4]
            list_inputs.append(inp_list)

            # 4. inp_pol_mask
            inp_pol_mask = BaseDataLoader.create_polarity_mask(ps).transpose(1, 0) # [N, 2]
            pol_mask_inputs.append(inp_pol_mask)

        # 批处理和填充
        batch_inp_voxel = torch.stack(voxel_inputs)
        batch_inp_cnt = torch.stack(cnt_inputs)
        batch_inp_list = torch.nn.utils.rnn.pad_sequence(list_inputs, batch_first=True)
        batch_inp_pol_mask = torch.nn.utils.rnn.pad_sequence(pol_mask_inputs, batch_first=True)

        return batch_inp_voxel, batch_inp_cnt, batch_inp_list, batch_inp_pol_mask

    def _compute_mad_appearance(self, T_m, inp_list, inp_pol_mask, H_orig, W_orig):
        """ 包装 iwe.py 中的 compute_pol_iwe 逻辑 """
        inp_pol_mask0 = inp_pol_mask[:, :, 0:1]
        inp_pol_mask1 = inp_pol_mask[:, :, 1:2]
        
        resize_shape_pad = (self.mad_H_pad, self.mad_W_pad)

        # T_a = compute_pol_iwe(
        #     T_m,
        #     inp_list,
        #     resize_shape,
        #     inp_pol_mask0,
        #     inp_pol_mask1,
        #     flow_scaling=resize_shape[1], # 使用宽度 640
        #     round_idx=True,
        # )
        T_a_padded = compute_pol_iwe(
            T_m, # T_m 已经是 [B, 2, 224, 320]
            inp_list,
            resize_shape_pad, # 告诉 iwe.py 使用 padding 后的尺寸
            inp_pol_mask0,
            inp_pol_mask1,
            flow_scaling=resize_shape_pad[1], # 320
            round_idx=True,
        )
        return T_a_padded[:, :, :H_orig, :W_orig] # 裁剪 [:, :, :215, :320]

    def _create_cnt_encoding(self, xs, ys, ps, sensor_size):
        return self._events_to_channels(xs, ys, ps, sensor_size)
    
    def _events_to_channels(self, xs, ys, ps, sensor_size):
        mask_pos = ps.clone()
        mask_neg = ps.clone()
        mask_pos[ps < 0] = 0
        mask_neg[ps > 0] = 0
        
        pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
        neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)
        return torch.stack([pos_cnt, neg_cnt])

    def forward(self, data):
        H, W = int(data.height[0]), int(data.width[0])
        device = data.x.device

        # try:
        #     print(f"[HybridDebug] rgb_module={self.rgb.net.module.__class__.__name__} feature_layers={self.rgb.net.feature_layers} output_layers={self.rgb.net.output_layers}")
        # except Exception as e:
        #     print(f"[HybridDebug] rgb_module/info unavailable: {repr(e)}")
        # try:
        #     print(f"[HybridDebug] image: shape={tuple(data.image.shape)}, dtype={data.image.dtype}, device={data.image.device}")
        # except Exception as e:
        #     print(f"[HybridDebug] image: unavailable ({repr(e)})")

        # -- checkpointing --
        if self.training and self.use_checkpointing:
            features, image_outs = activation_checkpoint(self.rgb, data.image, use_reentrant=False)
        else:
            features, image_outs = self.rgb(data.image)
        # print(f"[HybridDebug] HookModule -> features={len(features)}, outputs={len(image_outs)}")
        # for i, f in enumerate(features):
        #     try:
        #         print(f"[HybridDebug] features[{i}]: {tuple(f.shape)}")
        #     except Exception as e:
        #         print(f"[HybridDebug] features[{i}]: shape unavailable ({repr(e)})")
        # for i, o in enumerate(image_outs):
        #     try:
        #         print(f"[HybridDebug] outputs[{i}]: {tuple(o.shape)}")
        #     except Exception as e:
        #         print(f"[HybridDebug] outputs[{i}]: shape unavailable ({repr(e)})")
        # if len(image_outs) < 2:
        #     print("[HybridDebug][WARN] image_outs fewer than 2 items; check output_layers and img_net")

        rgb_c2 = features[1] if len(features) > 1 else None
        rgb_c3 = features[2] if len(features) > 2 else None
        rgb_c4 = image_outs[0]
        rgb_c5 = image_outs[1]

        # SNN temporal features
        if self.training and self.use_checkpointing:
            snn_feats = activation_checkpoint(self.snn.forward_time, data, use_reentrant=False)
        else:
            snn_feats = self.snn.forward_time(data)
        # print(f"[HybridDebug] snn taps: {list(snn_feats.keys())}")
        # for k, v in snn_feats.items():
        #     try:
        #         print(f"[HybridDebug] snn[{k}] shape={tuple(v.shape)}")
        #     except Exception as e:
        #         print(f"[HybridDebug] snn[{k}] shape unavailable ({repr(e)})")
        p2_t = snn_feats.get("p2")
        p3_t = snn_feats.get("p3")
        p4_t = snn_feats.get("p4")
        p5_t = snn_feats.get("p5")

        if self.training and self.use_checkpointing:
            fused_p2 = activation_checkpoint(self.fuse_p2, rgb_c2, p2_t, use_reentrant=False) if (rgb_c2 is not None and p2_t is not None) else None
            fused_p3 = activation_checkpoint(self.fuse_p3, rgb_c3, p3_t, use_reentrant=False) if (rgb_c3 is not None and p3_t is not None) else None
            fused_p4 = activation_checkpoint(self.fuse_p4, rgb_c4, p4_t, use_reentrant=False) if (rgb_c4 is not None and p4_t is not None) else None
            fused_p5 = activation_checkpoint(self.fuse_p5, rgb_c5, p5_t, use_reentrant=False) if (rgb_c5 is not None and p5_t is not None) else None
        else:
            fused_p2 = self.fuse_p2(rgb_c2, p2_t) if (rgb_c2 is not None and p2_t is not None) else None
            fused_p3 = self.fuse_p3(rgb_c3, p3_t) if (rgb_c3 is not None and p3_t is not None) else None
            fused_p4 = self.fuse_p4(rgb_c4, p4_t) if (rgb_c4 is not None and p4_t is not None) else None
            fused_p5 = self.fuse_p5(rgb_c5, p5_t) if (rgb_c5 is not None and p5_t is not None) else None

        fused = [x for x in [fused_p2, fused_p3, fused_p4, fused_p5] if x is not None]
        rgb_only = [x for x in [rgb_c2, rgb_c3, rgb_c4, rgb_c5] if x is not None]
        
        # --- MAD branch ---
        mad_feats = None
        if self.training:
            with torch.no_grad(): # T_m 和 T_a 的生成不应计算梯度
                T_m, T_a = torch.zeros(1), torch.zeros(1) # 占位符
                try:
                    # 1. 准备 MAD 输入
                    mad_inputs = self._convert_dagr_to_mad_inputs(data)
                    b_vox, b_cnt, b_list, b_pol = mad_inputs
                    
                    # 2. 获取 T_m (运动)
                    self.mad_flow.to(device)
                    flow_output = self.mad_flow(b_vox.to(device), b_cnt.to(device))
                    T_m = flow_output["flow"][0].detach() # [B, 2, H, W]

                    # 3. 获取 T_a (外观)
                    T_a = self._compute_mad_appearance(T_m, b_list, b_pol, H, W).detach() # [B, 2, H, W]
                
                except Exception as e:
                    print(f"[HybridDebug] MAD T_a/T_m 生成失败: {e}")
                    # 创建B,C,H,W零张量以允许训练继续
                    T_a = torch.zeros(data.num_graphs, 2, H, W, device=device)
                    T_m = torch.zeros(data.num_graphs, 2, H, W, device=device)

            # --- 在送入 MADBackbone 之前，将 T_a 和 T_m 填充到 (224, 320) ---
            # MADBackbone 也需要 16 的倍数
            H_pad = self.mad_H_pad # 224
            W_pad = self.mad_W_pad # 320
            
            # (top, bottom, left, right)
            padding = (0, W_pad - W, 0, H_pad - H) # (0, 0, 0, 9)
            
            T_a_padded_for_backbone = F.pad(T_a, padding, "constant", 0)
            T_m_padded_for_backbone = T_m # [B, 2, H, W]
            # T_m_padded_for_backbone = F.pad(T_m, padding, "constant", 0)
            # --- 结束 ---

            # 4. 运行 MADBackbone (这部分需要计算梯度)
            if self.use_checkpointing:
                mad_feats = activation_checkpoint(self.mad_backbone, T_a_padded_for_backbone, T_m_padded_for_backbone, use_reentrant=False)
            else:
                mad_feats = self.mad_backbone(T_a_padded_for_backbone, T_m_padded_for_backbone)
        # --- MAD branch end ---

        # mad_feats=None in inference
        return fused, rgb_only, mad_feats


