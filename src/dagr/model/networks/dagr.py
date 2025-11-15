import torch

import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss

from dagr.model.networks.net import Net
try:
    from dagr.model.networks.snn_backbone_yaml import SNNBackboneYAMLWrapper
except Exception:
    SNNBackboneYAMLWrapper = None
# try:
#     from dagr.model.networks.hybrid_backbone import HybridBackbone
# except Exception:
#     HybridBackbone = None
from dagr.model.networks.hybrid_backbone import HybridBackbone
from dagr.model.layers.spline_conv import SplineConvToDense
from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_subnetwork, voxel_size_to_params, postprocess_network_output, convert_to_evaluation_format, init_grid_and_stride, convert_to_training_format

# --- 新增: HybridHeadV2 (用于三分支训练) ---
class HybridHeadV2(YOLOXHead):
    def __init__(self, num_classes, strides, 
                 in_channels_fused, 
                 in_channels_image, 
                 in_channels_mad, 
                 act="silu", depthwise=False, width=1.0, args=None):
        
        # 注意：YOLOXHead 的 'width' 参数会缩放 'in_channels'
        # 我们在这里假设 'width' 已经应用在 backbone 的输出通道上
        # 或者我们在这里为每个 head 设置 width=1.0
        
        # 我们需要一个 YOLOXHead 来处理 'super' 调用，但我们实际上将使用 3 个
        super().__init__(num_classes, width=1.0, strides=strides, in_channels=in_channels_fused, act=act, depthwise=depthwise)
        
        self.strides = strides
        self.num_scales = len(in_channels_fused)

        # 1. Fused Head (用于推理和训练)
        self.fused_head = YOLOXHead(num_classes, width=1.0, strides=strides, in_channels=in_channels_fused, act=act, depthwise=depthwise)
        
        # 2. Image Head (仅用于训练)
        self.image_head = YOLOXHead(num_classes, width=1.0, strides=strides, in_channels=in_channels_image, act=act, depthwise=depthwise)
        
        # 3. MAD Head (仅用于训练)
        self.mad_head = YOLOXHead(num_classes, width=1.0, strides=strides, in_channels=in_channels_mad, act=act, depthwise=depthwise)

        self.use_checkpointing = getattr(args, 'use_checkpointing', False) if args else False

    def forward(self, xin, labels=None, imgs=None):
        
        if self.training:
            # 1. 解包特征和标签
            fused_feats, image_feats, mad_feats = xin
            # 假设标签元组为 (labels_fused, labels_image, labels_mad)
            labels_fused, labels_image, labels_mad = labels

            if self.use_checkpointing:
                losses_fused = activation_checkpoint(self.fused_head, fused_feats, labels_fused, imgs, use_reentrant=False)
                losses_image = activation_checkpoint(self.image_head, image_feats, labels_image, imgs, use_reentrant=False)
                if mad_feats is not None:
                    losses_mad = activation_checkpoint(self.mad_head, mad_feats, labels_mad, imgs, use_reentrant=False)
                else:
                    # losses_mad = {
                    #     'total_loss': 0.0, 'loss_cls': 0.0, 'loss_conf': 0.0, 'loss_reg': 0.0, 'num_fg': 0
                    # }
                    zero_loss = torch.tensor(0.0, device=fused_feats[0].device, dtype=fused_feats[0].dtype)
                    losses_mad = (zero_loss, zero_loss, zero_loss, zero_loss, zero_loss, 0.0)
            else:

                # 2. 计算 Fused 分支损失
                losses_fused = self.fused_head(fused_feats, labels_fused, imgs)
                
                # 3. 计算 Image 分支损失
                losses_image = self.image_head(image_feats, labels_image, imgs)

                # 4. 计算 MAD 分支损失
                if mad_feats is not None:
                    losses_mad = self.mad_head(mad_feats, labels_mad, imgs)
                else:
                    # 如果 MAD 分支失败 (例如，在推理或数据转换失败时)
                    # losses_mad = {
                    #     'total_loss': 0.0, 'loss_cls': 0.0, 'loss_conf': 0.0, 'loss_reg': 0.0, 'num_fg': 0
                    # }
                    zero_loss = torch.tensor(0.0, device=fused_feats[0].device, dtype=fused_feats[0].dtype)
                    losses_mad = (zero_loss, zero_loss, zero_loss, zero_loss, zero_loss, 0.0)

            # 5. 合并损失
            # total_loss = losses_fused['total_loss'] + losses_image['total_loss'] + losses_mad['total_loss']
            total_loss = losses_fused[0] + losses_image[0] + losses_mad[0]
            iou_loss = losses_fused[1] + losses_image[1] + losses_mad[1]
            obj_loss = losses_fused[2] + losses_image[2] + losses_mad[2]
            cls_loss = losses_fused[3] + losses_image[3] + losses_mad[3]
            l1_loss = losses_fused[4] + losses_image[4] + losses_mad[4]
            num_fg = losses_fused[5] + losses_image[5] + losses_mad[5]
            
            # 返回 YOLOX.forward所期望的 *元组*
            return (total_loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg)

        else: # --- 推理 ---
            # 仅使用 Fused 分支
            fused_feats, _, _ = xin
            return self.fused_head(fused_feats)
# --- HybridHeadV2 结束 ---

class DAGR(YOLOX):
    def __init__(self, args, height, width):
        self.conf_threshold = 0.001
        self.nms_threshold = 0.65

        self.height = height
        self.width = width

        use_snn = hasattr(args, 'use_snn_backbone') and getattr(args, 'use_snn_backbone') and SNNBackboneYAMLWrapper is not None
        print(f"Debug: use_snn: {use_snn}")

        use_image = hasattr(args, 'use_image') and getattr(args, 'use_image')
        print(f"Debug: use_image: {use_image}")

        if use_snn and getattr(args, 'use_image', False) and HybridBackbone is not None:
            # --- Moddified for 3-branch hybrid backbone (Fused, RGB, MAD) ---
            # print(f"Debug: running with hybrid backbone")
            # backbone = HybridBackbone(args, height=height, width=width)
            # head = HybridHead(num_classes=backbone.num_classes,
            #                    strides=backbone.strides,
            #                    in_channels=backbone.out_channels,
            #                    args=args)
            # --- Modified end ---
            print(f"Debug: running with 3-branch hybrid backbone (Fused, RGB, MAD)")
            backbone = HybridBackbone(args, height=height, width=width)

            rgb_all_channels = backbone.rgb.feature_channels + backbone.rgb.output_channels
            head = HybridHeadV2(
                num_classes=backbone.num_classes,
                strides=backbone.strides,
                in_channels_fused=backbone.out_channels,     # SNN-fused
                in_channels_image=rgb_all_channels, # RGB-only
                in_channels_mad=backbone.mad_backbone.out_channels, # MAD-only
                args=args
            )
        elif use_snn:
            yaml_path = getattr(args, 'snn_yaml_path', 'dagr/src/dagr/cfg/snn_yolov8.yaml')
            scale = getattr(args, 'snn_scale', 's')
            backbone = SNNBackboneYAMLWrapper(args, height=height, width=width, yaml_path=yaml_path, scale=scale)
            head = YOLOXHead(num_classes=backbone.num_classes,
                             width=1.0,
                             strides=backbone.strides,
                             in_channels=backbone.out_channels)
        else:
            backbone = Net(args, height=height, width=width)
            head = GNNHead(num_classes=backbone.num_classes,
                           in_channels=backbone.out_channels,
                           in_channels_cnn=backbone.out_channels_cnn,
                           strides=backbone.strides,
                           pretrain_cnn=args.pretrain_cnn,
                           args=args)

        super().__init__(backbone=backbone, head=head)

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            init_subnetwork(self, state_dict['ema'], "head.cnn_head.")

    def cache_luts(self, width, height, radius):
        # LUTs are specific to graph-based spline convs; skip when using SNN backbone.
        if isinstance(self.backbone, Net):
            M = 2 * float(int(radius * width + 2) / width)
            r = int(radius * width+1)
            self.backbone.conv_block1.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=r)
            self.backbone.conv_block1.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=r)

            rx, ry, M = voxel_size_to_params(self.backbone.pool1, height, width)
            self.backbone.layer2.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.backbone.layer2.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

            rx, ry, M = voxel_size_to_params(self.backbone.pool2, height, width)
            self.backbone.layer3.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.backbone.layer3.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

            rx, ry, M = voxel_size_to_params(self.backbone.pool3, height, width)
            self.backbone.layer4.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.backbone.layer4.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

            self.head.stem1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.obj_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

            rx, ry, M = voxel_size_to_params(self.backbone.pool4, height, width)
            self.backbone.layer5.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.backbone.layer5.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

            if self.head.num_scales > 1:
                self.head.stem2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
                self.head.cls_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
                self.head.reg_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
                self.head.cls_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
                self.head.reg_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
                self.head.obj_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        # --- Modified for 3-branch hybrid backbone (Fused, RGB, MAD) ---
        # if not hasattr(self.head, "output_sizes"):
        #     self.head.output_sizes = self.backbone.get_output_sizes()
        # --- Modified end ---

        if not hasattr(self.head, "output_sizes") and hasattr(self.backbone, "get_output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)

            if self.backbone.use_image:
                targets0 = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
                # --- Modified for 3-branch hybrid backbone (Fused, RGB, MAD) ---
                # 假设MAD分支也应该预测当前帧 (targets)，而不是前一帧 (targets0)
                targets_tuple = (targets, targets0, targets)
                outputs = YOLOX.forward(self, x, targets_tuple)
                # targets = (targets, targets0)
                # --- Modified end ---
            else:
                outputs = YOLOX.forward(self, x, targets)

            return outputs

            # gt_target inputs need to be [l cx cy w h] in pixels
            # outputs = YOLOX.forward(self, x, targets)

            # return outputs

        x.reset = reset

        outputs = YOLOX.forward(self, x)

        detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)

        ret = [detections]

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)

        return ret


class CNNHead(YOLOXHead):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False):
        super().__init__(num_classes, width, strides, in_channels, act, depthwise)
    
    def forward(self, xin):
        outputs = dict(cls_output=[], reg_output=[], obj_output=[])

        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)

            outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))

        return outputs


class HybridHead(YOLOXHead):
    def __init__(self, num_classes, strides=[16, 32], in_channels=[256, 512], act="silu", depthwise=False, args=None):
        # Use width=1.0 to match fused feature channels exactly, not scaled by yolo_stem_width
        YOLOXHead.__init__(self, num_classes, 1.0, strides, in_channels, act, depthwise)
        self.strides = strides
        self.num_scales = len(in_channels)
        # Image-only head for auxiliary supervision
        self.image_head = CNNHead(num_classes=num_classes, width=1.0, strides=strides, in_channels=in_channels, act=act, depthwise=depthwise)

    def _forward_single(self, xin):
        outputs = dict(cls_output=[], reg_output=[], obj_output=[])
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)
            outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))
        return outputs
    
    def collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        """Collect and process outputs - key: distinguish between training and inference"""
        if self.training:
            # Training: decode bbox coordinates for loss calculation
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, output.type())
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(
                torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output)
            )
        else:
            # Inference: keep raw predictions, only add sigmoid
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
        
        ret['outputs'].append(output)

    def forward(self, xin, labels=None, imgs=None):
        fused_feats, image_feats = xin  # both are [BCHW] lists

        # Compute raw outputs once
        out_fused = self._forward_single(fused_feats)
        out_image = self.image_head(image_feats)

        if self.training:
            # Collect and process outputs for training
            fused_ret = dict(outputs=[], x_shifts=[], y_shifts=[], expanded_strides=[])
            image_ret = dict(outputs=[], x_shifts=[], y_shifts=[], expanded_strides=[])

            for k in range(self.num_scales):
                self.collect_outputs(out_fused["cls_output"][k], out_fused["reg_output"][k], 
                                   out_fused["obj_output"][k], k, self.strides[k], ret=fused_ret)
                self.collect_outputs(out_image["cls_output"][k], out_image["reg_output"][k], 
                                   out_image["obj_output"][k], k, self.strides[k], ret=image_ret)

            if isinstance(labels, tuple) and len(labels) == 2:
                labels_fused, labels_image = labels
            else:
                labels_fused, labels_image = labels, labels

            # Outputs are already [B, H*W, C] after get_output_and_grid, directly concatenate
            losses_image = self.get_losses(
                imgs,
                image_ret['x_shifts'],
                image_ret['y_shifts'],
                image_ret['expanded_strides'],
                labels_image,
                torch.cat(image_ret['outputs'], 1),  # [B, total_anchors, C]
                [],  # origin_preds not needed when use_l1=False
                dtype=image_feats[0].dtype,
            )

            losses_fused = self.get_losses(
                imgs,
                fused_ret['x_shifts'],
                fused_ret['y_shifts'],
                fused_ret['expanded_strides'],
                labels_fused,
                torch.cat(fused_ret['outputs'], 1),  # [B, total_anchors, C]
                [],  # origin_preds not needed when use_l1=False
                dtype=fused_feats[0].dtype,
            )

            # Sum losses element-wise
            return tuple(l_img + l_fused for l_img, l_fused in zip(losses_image, losses_fused))

        else:  # Inference
            # Collect outputs for inference (no decoding in collect_outputs)
            fused_ret = dict(outputs=[])
            for k in range(self.num_scales):
                self.collect_outputs(out_fused["cls_output"][k], out_fused["reg_output"][k],
                                   out_fused["obj_output"][k], k, self.strides[k], ret=fused_ret)

            self.hw = [x.shape[-2:] for x in fused_ret['outputs']]
            # Flatten and concatenate [B, C, H, W] -> [B, total_anchors, C]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in fused_ret['outputs']], dim=2
            ).permute(0, 2, 1)

            return self.decode_outputs(outputs, dtype=fused_feats[0].type())

class GNNHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        in_channels_cnn=[256, 512, 1024],
        act="silu",
        depthwise=False,
        pretrain_cnn=False,
        args=None
    ):
        YOLOXHead.__init__(self, num_classes, args.yolo_stem_width, strides, in_channels, act, depthwise)

        self.pretrain_cnn = pretrain_cnn
        self.num_scales = args.num_scales
        self.use_image = args.use_image
        self.batch_size = args.batch_size
        self.no_events = args.no_events

        self.in_channels = in_channels
        self.n_anchors = 1
        self.num_classes = num_classes

        n_reg = max(in_channels)
        self.stem1 = ConvBlock(in_channels=in_channels[0], out_channels=n_reg, args=args)
        self.cls_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.cls_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
        self.reg_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.reg_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
        self.obj_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.num_scales > 1:
            self.stem2 = ConvBlock(in_channels=in_channels[1], out_channels=n_reg, args=args)
            self.cls_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.cls_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
            self.reg_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.reg_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
            self.obj_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.use_image:
            self.cnn_head = CNNHead(num_classes=num_classes, strides=strides, in_channels=in_channels_cnn)

        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self.grid_cache = None
        self.stride_cache = None
        self.cache = []

    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred, batch_size, cache):
        x = stem(x)

        cls_feat = cls_conv(shallow_copy(x))
        reg_feat = reg_conv(x)

        # we need to provide the batchsize, since sometimes it cannot be foudn from the data, especially when nodes=0
        cls_output = cls_pred(cls_feat, batch_size=batch_size)
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        obj_output = obj_pred(reg_feat, batch_size=batch_size)

        return cls_output, reg_output, obj_output

    def forward(self, xin: Data, labels=None, imgs=None):
        # for events + image outputs
        hybrid_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])
        image_out = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])

        if self.use_image:
            xin, image_feat = xin

            if labels is not None:
                if self.use_image:
                    labels, image_labels = labels

            # resize image, and process with CNN
            image_feat = [torch.nn.functional.interpolate(f, o) for f, o in zip(image_feat, self.output_sizes)]
            out_cnn = self.cnn_head(image_feat)

            # collect outputs from image alone, so the image network also learns to detect on its own.
            for k in [0, 1]:
                self.collect_outputs(out_cnn["cls_output"][k],
                                     out_cnn["reg_output"][k],
                                     out_cnn["obj_output"][k],
                                     k, self.strides[k], ret=image_out)

        batch_size = len(out_cnn["cls_output"][0]) if self.use_image else self.batch_size
        cls_output, reg_output, obj_output = self.process_feature(xin[0], self.stem1, self.cls_conv1, self.reg_conv1,
                                                        self.cls_pred1, self.reg_pred1, self.obj_pred1, batch_size=batch_size, cache=self.cache)

        if self.use_image:
            cls_output[:batch_size] += out_cnn["cls_output"][0].detach()
            reg_output[:batch_size] += out_cnn["reg_output"][0].detach()
            obj_output[:batch_size] += out_cnn["obj_output"][0].detach()

        self.collect_outputs(cls_output, reg_output, obj_output, 0, self.strides[0], ret=hybrid_out)

        if self.num_scales > 1:
            cls_output, reg_output, obj_output = self.process_feature(xin[1], self.stem2, self.cls_conv2,
                                                                      self.reg_conv2, self.cls_pred2, self.reg_pred2,
                                                                      self.obj_pred2, batch_size=batch_size, cache=self.cache)
            if self.use_image:
                batch_size = out_cnn["cls_output"][0].shape[0]
                cls_output[:batch_size] += out_cnn["cls_output"][1].detach()
                reg_output[:batch_size] += out_cnn["reg_output"][1].detach()
                obj_output[:batch_size] += out_cnn["obj_output"][1].detach()

            self.collect_outputs(cls_output, reg_output, obj_output, 1, self.strides[1], ret=hybrid_out)

        if self.training:
            # if we are only training the image detectors (pretraining),
            # we only need to minimize the loss at detections from the image branch.
            if self.use_image:
                losses_image = self.get_losses(
                    imgs,
                    image_out['x_shifts'],
                    image_out['y_shifts'],
                    image_out['expanded_strides'],
                    image_labels,
                    torch.cat(image_out['outputs'], 1),
                    image_out['origin_preds'],
                    dtype=image_out['x_shifts'][0].dtype,
                )

                if not self.pretrain_cnn:
                    losses_events  = self.get_losses(
                    imgs,
                    hybrid_out['x_shifts'],
                    hybrid_out['y_shifts'],
                    hybrid_out['expanded_strides'],
                    labels,
                    torch.cat(hybrid_out['outputs'], 1),
                    hybrid_out['origin_preds'],
                    dtype=xin[0].x.dtype,
                )

                    losses_image = list(losses_image)
                    losses_events = list(losses_events)

                    for i in range(5):
                        losses_image[i] = losses_image[i] + losses_events[i]

                return losses_image
            else:
                return self.get_losses(
                    imgs,
                    hybrid_out['x_shifts'],
                    hybrid_out['y_shifts'],
                    hybrid_out['expanded_strides'],
                    labels,
                    torch.cat(hybrid_out['outputs'], 1),
                    hybrid_out['origin_preds'],
                    dtype=xin[0].x.dtype,
                )
        else:
            out = image_out['outputs'] if self.no_events else hybrid_out['outputs']

            self.hw = [x.shape[-2:] for x in out]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)

            return self.decode_outputs(outputs, dtype=out[0].type())

    def collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, output.type())
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))
        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

        ret['outputs'].append(output)

    def decode_outputs(self, outputs, dtype):
        if self.grid_cache is None:
            self.grid_cache, self.stride_cache = init_grid_and_stride(self.hw, self.strides, dtype)

        outputs[..., :2] = (outputs[..., :2] + self.grid_cache) * self.stride_cache
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.stride_cache
        return outputs

