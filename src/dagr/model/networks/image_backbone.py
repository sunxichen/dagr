import torch
import torch.nn as nn

from dagr.model.networks.net_img import HookModule
from torchvision.models import resnet18, resnet34, resnet50  # ensure eval(args.img_net) resolves


class ImageBackbone(nn.Module):
    """
    RGB backbone for hybrid fusion.

    - Wraps HookModule to extract image features/outputs
    - Applies 1x1 projections ONLY on the taps actually consumed by HybridBackbone
      features: layer1 (idx=1), layer2 (idx=2)
      outputs:  layer3 (idx=0), layer4 (idx=1)
    - Does NOT instantiate any graph related modules
    - Preserves submodule name `net` for checkpoint compatibility ("backbone.net.")
    """

    def __init__(self, args, height: int, width: int):
        super().__init__()

        # Channel configuration aligned with Net
        base_width = getattr(args, 'base_width', 1.0)
        after_pool_width = getattr(args, 'after_pool_width', 1.0)
        net_stem_width = getattr(args, 'net_stem_width', 1.0)

        channels = [
            1,
            int(base_width * 32),
            int(after_pool_width * 64),
            int(net_stem_width * 128),
            int(net_stem_width * 128),
            int(net_stem_width * 128),
        ]

        img_net_ctor = eval(getattr(args, 'img_net', 'resnet18'))
        self.net = HookModule(
            img_net_ctor(pretrained=True),
            input_channels=3,
            height=height,
            width=width,
            feature_layers=["conv1", "layer1", "layer2", "layer3", "layer4"],
            output_layers=["layer3", "layer4"],
            feature_channels=None,   # we project only selected taps below
            output_channels=None,
        )

        # Remove classification head parameters from underlying image backbone (e.g., ResNet.fc)
        if hasattr(self.net.module, 'fc'):
            self.net.module.fc = nn.Identity()

        # Exposed mapped channel specs for consumers (HybridBackbone)
        self.feature_channels = [channels[1], channels[2]]  # for layer1, layer2
        self.output_channels = [256, 256]                   # for layer3, layer4 outputs

        # 1x1 projections only for consumed taps
        # feature taps
        self.proj_feat_l1 = nn.Conv2d(self.net.feature_channels[1], self.feature_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_feat_l2 = nn.Conv2d(self.net.feature_channels[2], self.feature_channels[1], kernel_size=1, stride=1, padding=0, bias=False)

        # output taps
        self.proj_out_l3 = nn.Conv2d(self.net.output_channels[0], self.output_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_out_l4 = nn.Conv2d(self.net.output_channels[1], self.output_channels[1], kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, image: torch.Tensor):
        features, outputs = self.net(image)

        # Clone lists to avoid in-place side-effects
        features_mapped = list(features)
        outputs_mapped = list(outputs)

        # Map only consumed indices
        if len(features_mapped) > 2:
            features_mapped[1] = self.proj_feat_l1(features_mapped[1])
            features_mapped[2] = self.proj_feat_l2(features_mapped[2])

        if len(outputs_mapped) > 1:
            outputs_mapped[0] = self.proj_out_l3(outputs_mapped[0])
            outputs_mapped[1] = self.proj_out_l4(outputs_mapped[1])

        return features_mapped, outputs_mapped


