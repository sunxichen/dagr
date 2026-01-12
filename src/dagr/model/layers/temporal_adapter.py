import torch
import torch.nn as nn


class TemporalAdapter(nn.Module):
    """
    时序适配层：将 [T, B, C, H, W] 转换为 [B, C, H, W]
    
    策略：Conv1d(kernel_size=1) + mean(dim=0)
    
    对每个尺度的特征：
    1. 应用 Conv1d(kernel_size=1) 在通道维度上进行时序特征变换
    2. 使用 mean(dim=0) 进行时序池化
    """
    
    def __init__(self, in_channels_list):
        """
        Args:
            in_channels_list: List[int], 每个尺度的通道数，如 [128, 256, 512]
        """
        super().__init__()
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(ch, ch, kernel_size=1, bias=False)
            for ch in in_channels_list
        ])
    
    def forward(self, feats):
        """
        Args:
            feats: List[Tensor] 或 Tensor
                - 如果是 List: 每个元素是 [T, B, C, H, W]
                - 如果是 Tensor: 单个 [T, B, C, H, W]
        Returns:
            List[Tensor] 或 Tensor: 每个元素是 [B, C, H, W]
        """
        if isinstance(feats, (list, tuple)):
            # 处理多尺度特征列表
            return [self._process_single_feat(feat, idx) for idx, feat in enumerate(feats)]
        else:
            # 处理单个特征
            return self._process_single_feat(feats, 0)
    
    def _process_single_feat(self, feat, layer_idx):
        """
        处理单个时序特征：Conv1d + mean
        
        Args:
            feat: Tensor of shape [T, B, C, H, W]
            layer_idx: int, 使用的 Conv1d 层索引
        Returns:
            Tensor of shape [B, C, H, W]
        """
        T, B, C, H, W = feat.shape
        
        # 应用 Conv1d: 在时序维度上处理
        # Conv1d 输入格式: [batch, channels, length]
        # 将 [T, B, C, H, W] reshape 为 [B*H*W, C, T]
        # 这样每个空间位置和 batch 独立处理，在时序维度 T 上应用 Conv1d
        feat_reshaped = feat.permute(1, 3, 4, 2, 0).contiguous()  # [B, H, W, C, T]
        feat_reshaped = feat_reshaped.view(B * H * W, C, T)  # [B*H*W, C, T]
        
        # Conv1d 处理：在时序维度 T 上应用
        conv1d = self.conv1d_layers[layer_idx]
        feat_conv = conv1d(feat_reshaped)  # [B*H*W, C, T]
        
        # Reshape 回原始形状
        feat_conv = feat_conv.view(B, H, W, C, T)  # [B, H, W, C, T]
        feat_conv = feat_conv.permute(4, 0, 3, 1, 2).contiguous()  # [T, B, C, H, W]
        
        # 时序池化：mean over time dimension
        feat_spatial = feat_conv.mean(dim=0)  # [B, C, H, W]
        
        return feat_spatial

