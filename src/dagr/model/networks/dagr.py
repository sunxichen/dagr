import torch

import torch.nn.functional as F

from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss

from dagr.model.networks.net import Net
try:
    from dagr.model.networks.snn_backbone_yaml import SNNBackboneYAMLWrapper
except Exception:
    SNNBackboneYAMLWrapper = None
try:
    from dagr.model.networks.hybrid_backbone import HybridBackbone
except Exception:
    HybridBackbone = None
from dagr.model.layers.spline_conv import SplineConvToDense
from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_subnetwork, voxel_size_to_params, postprocess_network_output, convert_to_evaluation_format, init_grid_and_stride, convert_to_training_format


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
            print(f"Debug: running with hybrid backbone")
            backbone = HybridBackbone(args, height=height, width=width)
            head = HybridHead(num_classes=backbone.num_classes,
                               strides=backbone.strides,
                               in_channels=backbone.out_channels,
                               args=args)
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

        # propagate image loss alpha to head if provided
        try:
            if hasattr(args, 'hybrid_image_loss_alpha') and hasattr(self.head, '__setattr__'):
                setattr(self.head, 'image_loss_alpha', float(getattr(args, 'hybrid_image_loss_alpha', 0.0)))
        except Exception:
            pass

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
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)

            if self.backbone.use_image:
                targets0 = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
                targets = (targets, targets0)

            # gt_target inputs need to be [l cx cy w h] in pixels
            outputs = YOLOX.forward(self, x, targets)

            return outputs

        x.reset = reset

        outputs = YOLOX.forward(self, x)

        # Optional diagnostic: before/after filtering counts
        if hasattr(self.head, 'debug_eval') and getattr(self.head, 'debug_eval', False):
            try:
                det_no_filter = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=False,
                                                           height=self.height, width=self.width)
                total_before = sum(int(d['boxes'].shape[0]) for d in det_no_filter)
                print(f"[EvalDebug] conf_th={self.conf_threshold}, nms_th={self.nms_threshold}, candidates_before_filter={total_before}")
            except Exception as e:
                print(f"[EvalDebug][WARN] stats failed: {repr(e)}")

        detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)
        if hasattr(self.head, 'debug_eval') and getattr(self.head, 'debug_eval', False):
            try:
                total_after = sum(int(d['boxes'].shape[0]) for d in detections)
                print(f"[EvalDebug] candidates_after_filter={total_after}")
            except Exception as e:
                print(f"[EvalDebug][WARN] post stats failed: {repr(e)}")

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
    
    def _collect_outputs(self, cls_output, reg_output, obj_output, k, stride, ret):
        """Helper method to collect outputs; mirrors GNNHead semantics.
        - Training: keep logits and also compute grids/strides for loss.
        - Eval: apply sigmoid to obj/cls and only append outputs (no grids needed).
        """
        reg_output = reg_output.contiguous()
        obj_output = obj_output.contiguous()
        cls_output = cls_output.contiguous()

        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            ret["outputs"].append(output)
            ret["origin_preds"].append(torch.cat([reg_output, obj_output, cls_output], 1))

            hsize, wsize = cls_output.shape[-2:]
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type_as(cls_output)
            grid = grid.view(1, -1, 2)

            ret["x_shifts"].append(grid[..., 0])
            ret["y_shifts"].append(grid[..., 1])
            ret["expanded_strides"].append(torch.full((1, grid.shape[1]), stride, dtype=cls_output.dtype, device=cls_output.device))
        else:
            # Eval path: probabilities required by decode/postprocess
            output = torch.cat([
                reg_output,
                obj_output.sigmoid(),
                cls_output.sigmoid()
            ], 1)
            ret["outputs"].append(output)

    def forward(self, xin, labels=None, imgs=None):
        fused_feats, image_feats = xin  # both are [BCHW] lists

        # compute outputs
        out_fused = self._forward_single(fused_feats)
        out_image = self.image_head(image_feats)

        # collect feature maps
        fused_ret = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])
        image_ret = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])

        for k in range(self.num_scales):
            self._collect_outputs(out_fused["cls_output"][k], out_fused["reg_output"][k], out_fused["obj_output"][k], k, self.strides[k], ret=fused_ret)
            self._collect_outputs(out_image["cls_output"][k], out_image["reg_output"][k], out_image["obj_output"][k], k, self.strides[k], ret=image_ret)

        if self.training:
            # Expect labels to be (labels_fused, labels_image)
            if isinstance(labels, tuple) and len(labels) == 2:
                labels_fused, labels_image = labels
            else:
                labels_fused, labels_image = labels, labels

            # Labels diagnostics
            if getattr(self, 'debug_eval', False):
                try:
                    is_tuple = isinstance(labels, tuple)
                    same_obj = (labels_fused is labels_image)
                    print(f"[HybridTrainDebug][labels_detail] is_tuple={is_tuple}, same_object={same_obj}", flush=True)
                    def _label_bbox_stats(lbl):
                        try:
                            # lbl shape: [B, M, 5] => [cls, cx, cy, w, h]
                            mask = (lbl.sum(dim=2) > 0)
                            if mask.numel() == 0 or mask.sum() == 0:
                                return None
                            coords = lbl[:, :, 1:5]
                            coords = coords[mask]
                            return dict(
                                shape=tuple(lbl.shape),
                                x_min=float(coords[:, 0].min().detach().cpu()), x_max=float(coords[:, 0].max().detach().cpu()),
                                y_min=float(coords[:, 1].min().detach().cpu()), y_max=float(coords[:, 1].max().detach().cpu()),
                                w_min=float(coords[:, 2].min().detach().cpu()), w_max=float(coords[:, 2].max().detach().cpu()),
                                h_min=float(coords[:, 3].min().detach().cpu()), h_max=float(coords[:, 3].max().detach().cpu()),
                                count=int(coords.shape[0]),
                            )
                        except Exception:
                            return None
                    stats_img = _label_bbox_stats(labels_image)
                    stats_fus = _label_bbox_stats(labels_fused)
                    if stats_img is not None:
                        print(
                            f"[HybridTrainDebug][labels_image] shape={stats_img['shape']}, x=[{stats_img['x_min']:.1f},{stats_img['x_max']:.1f}], y=[{stats_img['y_min']:.1f},{stats_img['y_max']:.1f}], w=[{stats_img['w_min']:.1f},{stats_img['w_max']:.1f}], h=[{stats_img['h_min']:.1f},{stats_img['h_max']:.1f}], count={stats_img['count']}",
                            flush=True,
                        )
                    if stats_fus is not None:
                        print(
                            f"[HybridTrainDebug][labels_fused] shape={stats_fus['shape']}, x=[{stats_fus['x_min']:.1f},{stats_fus['x_max']:.1f}], y=[{stats_fus['y_min']:.1f},{stats_fus['y_max']:.1f}], w=[{stats_fus['w_min']:.1f},{stats_fus['w_max']:.1f}], h=[{stats_fus['h_min']:.1f},{stats_fus['h_max']:.1f}], count={stats_fus['count']}",
                            flush=True,
                        )
                except Exception as _e_lbl:
                    print(f"[HybridTrainDebug][labels_detail][WARN] {repr(_e_lbl)}", flush=True)

            # Flatten outputs along spatial dimension before concatenating across scales
            # Ensure contiguous tensors for proper view/reshape operations
            image_outputs_flat = torch.cat([x.contiguous().flatten(start_dim=2) for x in image_ret['outputs']], dim=2).permute(0, 2, 1).contiguous()
            fused_outputs_flat = torch.cat([x.contiguous().flatten(start_dim=2) for x in fused_ret['outputs']], dim=2).permute(0, 2, 1).contiguous()

            # BBox prediction diagnostics before loss
            if getattr(self, 'debug_eval', False):
                try:
                    def _stats(t):
                        t = t.detach()
                        return dict(
                            shape=tuple(t.shape),
                            min=float(t.min().cpu()), max=float(t.max().cpu()),
                            mean=float(t.mean().cpu()), std=float(t.std().cpu()),
                        )
                    img_bbox = image_outputs_flat[..., :4].reshape(-1, 4)
                    fus_bbox = fused_outputs_flat[..., :4].reshape(-1, 4)
                    s_img = _stats(img_bbox)
                    s_fus = _stats(fus_bbox)
                    print(f"[HybridTrainDebug][image][bbox_pred] shape={s_img['shape']}, min={s_img['min']:.3f}, max={s_img['max']:.3f}, mean={s_img['mean']:.3f}, std={s_img['std']:.3f}", flush=True)
                    print(f"[HybridTrainDebug][fused][bbox_pred] shape={s_fus['shape']}, min={s_fus['min']:.3f}, max={s_fus['max']:.3f}, mean={s_fus['mean']:.3f}, std={s_fus['std']:.3f}", flush=True)
                except Exception as _e_bbox:
                    print(f"[HybridTrainDebug][bbox_pred][WARN] {repr(_e_bbox)}", flush=True)
            
            losses_image = self.get_losses(
                imgs,
                image_ret['x_shifts'],
                image_ret['y_shifts'],
                image_ret['expanded_strides'],
                labels_image,
                image_outputs_flat,
                image_ret['origin_preds'],
                dtype=image_ret['x_shifts'][0].dtype,
            )

            losses_fused = self.get_losses(
                imgs,
                fused_ret['x_shifts'],
                fused_ret['y_shifts'],
                fused_ret['expanded_strides'],
                labels_fused,
                fused_outputs_flat,
                fused_ret['origin_preds'],
                dtype=fused_feats[0].dtype,
            )

            # Debug print for branch-wise losses and GT stats
            if getattr(self, 'debug_eval', False):
                try:
                    # Unpack losses (each returns a 6-tuple)
                    def _to_float(x):
                        try:
                            return float(x.detach().item())
                        except Exception:
                            try:
                                return float(x)
                            except Exception:
                                return 0.0

                    li = [
                        _to_float(losses_image[0]),
                        _to_float(losses_image[1]),
                        _to_float(losses_image[2]),
                        _to_float(losses_image[3]),
                        _to_float(losses_image[4]),
                        _to_float(losses_image[5]),
                    ]
                    lf = [
                        _to_float(losses_fused[0]),
                        _to_float(losses_fused[1]),
                        _to_float(losses_fused[2]),
                        _to_float(losses_fused[3]),
                        _to_float(losses_fused[4]),
                        _to_float(losses_fused[5]),
                    ]

                    # Count GTs per sample for each label set
                    num_gt_img = None
                    num_gt_fus = None
                    try:
                        nlabel_img = (labels_image.sum(dim=2) > 0).sum(dim=1)
                        num_gt_img = [int(x) for x in nlabel_img.detach().cpu().tolist()]
                    except Exception:
                        pass
                    try:
                        nlabel_fus = (labels_fused.sum(dim=2) > 0).sum(dim=1)
                        num_gt_fus = [int(x) for x in nlabel_fus.detach().cpu().tolist()]
                    except Exception:
                        pass

                    print(
                        f"[HybridTrainDebug][image] total={li[0]:.4f}, iou={li[1]:.4f}, obj={li[2]:.4f}, cls={li[3]:.4f}, l1={li[4]:.4f}, num_fg={li[5]:.2f}",
                        flush=True,
                    )
                    print(
                        f"[HybridTrainDebug][fused] total={lf[0]:.4f}, iou={lf[1]:.4f}, obj={lf[2]:.4f}, cls={lf[3]:.4f}, l1={lf[4]:.4f}, num_fg={lf[5]:.2f}",
                        flush=True,
                    )
                    if (num_gt_img is not None) or (num_gt_fus is not None):
                        print(
                            f"[HybridTrainDebug][labels] image_num_gt={num_gt_img}, fused_num_gt={num_gt_fus}",
                            flush=True,
                        )
                except Exception as _e:
                    print(f"[HybridTrainDebug][WARN] failed to print debug losses: {repr(_e)}", flush=True)

            # alpha = float(getattr(self, 'image_loss_alpha', 0.0))
            # if alpha != 0.0:
            #     losses_image = list(losses_image)
            #     losses_fused = list(losses_fused)
            #     for i in range(len(losses_fused)):
            #         losses_fused[i] = losses_fused[i] + alpha * losses_image[i]
            #     return losses_fused
            # else:
            #     losses_fused = list(losses_fused)
            #     return losses_fused
            losses_fused = list(losses_fused)
            return losses_fused

        # eval: use fused outputs
        out = fused_ret['outputs']

        # Optional debug: shapes/strides and activation ranges
        if getattr(self, 'debug_eval', False):
            try:
                num_scales = len(out)
                print(f"[HybridEvalDebug] num_scales={num_scales}, strides={self.strides}")
                for k in range(min(num_scales, 2)):
                    # pre-activation stats from raw heads
                    cls_raw = out_fused["cls_output"][k]
                    obj_raw = out_fused["obj_output"][k]
                    cls_raw_min = float(cls_raw.min().detach().cpu()) if cls_raw.numel() > 0 else 0.0
                    cls_raw_max = float(cls_raw.max().detach().cpu()) if cls_raw.numel() > 0 else 0.0
                    obj_raw_min = float(obj_raw.min().detach().cpu()) if obj_raw.numel() > 0 else 0.0
                    obj_raw_max = float(obj_raw.max().detach().cpu()) if obj_raw.numel() > 0 else 0.0
                    # post-activation stats from collected outputs
                    # layout: [reg4, obj1, clsC]
                    reg_ch = 4
                    obj_ch = 1
                    cls_ch = out[k].shape[1] - reg_ch - obj_ch
                    obj_act = out[k][:, reg_ch:reg_ch+obj_ch]
                    cls_act = out[k][:, reg_ch+obj_ch:reg_ch+obj_ch+cls_ch]
                    obj_act_min = float(obj_act.min().detach().cpu()) if obj_act.numel() > 0 else 0.0
                    obj_act_max = float(obj_act.max().detach().cpu()) if obj_act.numel() > 0 else 0.0
                    cls_act_min = float(cls_act.min().detach().cpu()) if cls_act.numel() > 0 else 0.0
                    cls_act_max = float(cls_act.max().detach().cpu()) if cls_act.numel() > 0 else 0.0
                    hsize, wsize = out[k].shape[-2:]
                    print(f"[HybridEvalDebug][s{k}] hw=({hsize},{wsize}), stride={self.strides[k]}, obj_raw=[{obj_raw_min:.3f},{obj_raw_max:.3f}], cls_raw=[{cls_raw_min:.3f},{cls_raw_max:.3f}], obj_act=[{obj_act_min:.3f},{obj_act_max:.3f}], cls_act=[{cls_act_min:.3f},{cls_act_max:.3f}]")

                # Also print image-branch activation stats for comparison
                try:
                    img_out_list = image_ret['outputs']
                    num_scales_img = len(img_out_list)
                    for k in range(min(num_scales_img, 2)):
                        cls_raw_i = out_image["cls_output"][k]
                        obj_raw_i = out_image["obj_output"][k]
                        cls_raw_min_i = float(cls_raw_i.min().detach().cpu()) if cls_raw_i.numel() > 0 else 0.0
                        cls_raw_max_i = float(cls_raw_i.max().detach().cpu()) if cls_raw_i.numel() > 0 else 0.0
                        obj_raw_min_i = float(obj_raw_i.min().detach().cpu()) if obj_raw_i.numel() > 0 else 0.0
                        obj_raw_max_i = float(obj_raw_i.max().detach().cpu()) if obj_raw_i.numel() > 0 else 0.0

                        reg_ch = 4
                        obj_ch = 1
                        cls_ch_i = img_out_list[k].shape[1] - reg_ch - obj_ch
                        obj_act_i = img_out_list[k][:, reg_ch:reg_ch+obj_ch]
                        cls_act_i = img_out_list[k][:, reg_ch+obj_ch:reg_ch+obj_ch+cls_ch_i]
                        obj_act_min_i = float(obj_act_i.min().detach().cpu()) if obj_act_i.numel() > 0 else 0.0
                        obj_act_max_i = float(obj_act_i.max().detach().cpu()) if obj_act_i.numel() > 0 else 0.0
                        cls_act_min_i = float(cls_act_i.min().detach().cpu()) if cls_act_i.numel() > 0 else 0.0
                        cls_act_max_i = float(cls_act_i.max().detach().cpu()) if cls_act_i.numel() > 0 else 0.0
                        hsize_i, wsize_i = img_out_list[k].shape[-2:]
                        print(f"[HybridEvalDebug][image][s{k}] hw=({hsize_i},{wsize_i}), stride={self.strides[k]}, obj_raw=[{obj_raw_min_i:.3f},{obj_raw_max_i:.3f}], cls_raw=[{cls_raw_min_i:.3f},{cls_raw_max_i:.3f}], obj_act=[{obj_act_min_i:.3f},{obj_act_max_i:.3f}], cls_act=[{cls_act_min_i:.3f},{cls_act_max_i:.3f}]")
                except Exception as _e2:
                    print(f"[HybridEvalDebug][image][WARN] failed to compute image-branch stats: {repr(_e2)}")
            except Exception as e:
                print(f"[HybridEvalDebug][WARN] failed to compute stats: {repr(e)}")

        self.hw = [x.shape[-2:] for x in out]
        outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)
        return self.decode_outputs(outputs, dtype=out[0].type())

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

