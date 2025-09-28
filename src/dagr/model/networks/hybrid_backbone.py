import torch
import torch.nn as nn

from dagr.model.networks.net_img import HookModule
from torchvision.models import resnet18, resnet34, resnet50
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

        # RGB backbone (pretrained ResNet via HookModule)
        img_net = eval(args.img_net)
        self.rgb = HookModule(img_net(pretrained=True),
                              input_channels=3,
                              height=height, width=width,
                              feature_layers=["layer2", "layer3", "layer4"],
                              output_layers=["layer3", "layer4"],
                              feature_channels=None,
                              output_channels=[256, 512])

        # SNN backbone (temporal features)
        yaml_path = getattr(args, 'snn_yaml_path', 'dagr/src/dagr/cfg/snn_yolov8.yaml')
        scale = getattr(args, 'snn_scale', 's')
        self.snn = SNNBackboneYAMLWrapper(args, height=height, width=width, yaml_path=yaml_path, scale=scale)

        # fusion blocks aligned to output_layers: we fuse at two stages (stride 16, 32)
        self.fuse_p4 = SpikeCAFR(in_channels=256, out_channels=256)
        self.fuse_p5 = SpikeCAFR(in_channels=512, out_channels=512)

        self.out_channels = [256, 512]
        self.strides = [16, 32]
        self.num_scales = 2
        self.num_classes = self.snn.num_classes
        self.use_image = True

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return sizes

    def forward(self, data):
        # RGB features
        rgb_feats, rgb_outs = self.rgb(data.image)
        # rgb_outs correspond to [C4,C5] with channels [256,512]

        # SNN temporal features
        snn_feats = self.snn.forward_time(data)
        p4_t = snn_feats["p4"]  # [T,B,256,H/16,W/16]
        p5_t = snn_feats["p5"]  # [T,B,512,H/32,W/32]

        # Fuse at two stages, following REFusion-like progressive design
        fused_p4 = self.fuse_p4(rgb_outs[0], p4_t)
        fused_p5 = self.fuse_p5(rgb_outs[1], p5_t)

        # return fused and image-only features for dual supervision in HybridHead
        return [fused_p4, fused_p5], [rgb_outs[0], rgb_outs[1]]


