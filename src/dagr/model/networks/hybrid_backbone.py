import torch
import torch.nn as nn

from dagr.model.networks.image_backbone import ImageBackbone
from dagr.model.networks.snn_backbone_yaml import SNNBackboneYAMLWrapper
from dagr.model.layers.fusion import SpikeCAFR


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

    def forward(self, data):
        # try:
        #     print(f"[HybridDebug] rgb_module={self.rgb.net.module.__class__.__name__} feature_layers={self.rgb.net.feature_layers} output_layers={self.rgb.net.output_layers}")
        # except Exception as e:
        #     print(f"[HybridDebug] rgb_module/info unavailable: {repr(e)}")
        # try:
        #     print(f"[HybridDebug] image: shape={tuple(data.image.shape)}, dtype={data.image.dtype}, device={data.image.device}")
        # except Exception as e:
        #     print(f"[HybridDebug] image: unavailable ({repr(e)})")

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

        fused_p2 = self.fuse_p2(rgb_c2, p2_t) if (rgb_c2 is not None and p2_t is not None) else None
        fused_p3 = self.fuse_p3(rgb_c3, p3_t) if (rgb_c3 is not None and p3_t is not None) else None
        fused_p4 = self.fuse_p4(rgb_c4, p4_t) if (rgb_c4 is not None and p4_t is not None) else None
        fused_p5 = self.fuse_p5(rgb_c5, p5_t) if (rgb_c5 is not None and p5_t is not None) else None

        fused = [x for x in [fused_p2, fused_p3, fused_p4, fused_p5] if x is not None]
        rgb_only = [x for x in [rgb_c2, rgb_c3, rgb_c4, rgb_c5] if x is not None]
        return fused, rgb_only


