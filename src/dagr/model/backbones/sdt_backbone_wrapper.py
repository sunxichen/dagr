import torch.nn as nn
from dagr.model.layers.temporal_adapter import TemporalAdapter


class SDTBackboneWrapper(nn.Module):
    """
    包装 SpikformerV3Extractor，在 forward 中应用时序适配层
    
    这样 YOLOX 看到的仍然是 [B, C, H, W] 格式，不需要修改 YOLOX 代码
    
    当不使用融合层时（use_image=False），使用此包装器：
    - Backbone 返回 [T, B, C, H, W] 格式的时序特征
    - 包装器应用适配层转换为 [B, C, H, W] 格式
    - YOLOX Head 正常接收和处理
    """
    
    def __init__(self, backbone):
        """
        Args:
            backbone: SpikformerV3Extractor 实例，返回 [T, B, C, H, W] 格式特征
        """
        super().__init__()
        self.backbone = backbone
        
        # 创建适配层，使用 backbone 的输出通道数
        self.temporal_adapter = TemporalAdapter(backbone.out_channels)
        
        # 暴露 backbone 的所有属性，保持接口兼容
        self.out_channels = backbone.out_channels
        self.strides = backbone.strides
        self.num_scales = backbone.num_scales
        self.num_classes = backbone.num_classes
        self.use_image = backbone.use_image
        self.is_snn = backbone.is_snn
    
    def get_output_sizes(self):
        """获取输出尺寸，与原始 backbone 相同"""
        return self.backbone.get_output_sizes()
    
    def forward(self, x, reset=True):
        """
        Forward pass: 调用原始 backbone，然后应用适配层
        
        Args:
            x: 输入数据（Data 对象或 Tensor）
            reset: bool, 是否重置 SNN 状态
        
        Returns:
            List[Tensor]: [p3, p4, p5]，每个是 [B, C, H, W] 格式
        """
        # 调用原始 backbone，返回 [T, B, C, H, W] 格式的时序特征
        temporal_feats = self.backbone(x, reset=reset)
        
        # 应用适配层，转换为 [B, C, H, W] 格式
        spatial_feats = self.temporal_adapter(temporal_feats)
        
        return spatial_feats

