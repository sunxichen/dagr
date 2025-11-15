import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 dagr 项目中导入所需的 ResNet-like 模块
# 如果 dagr.model.layers.conv.Layer 不适用，
# 我们可以使用标准的 Conv+BN+ReLU 块。
# 为简单起见，我们定义一个简单的 ConvBlock。
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class EMGA(nn.Module):
    """
    Event Motion Guided Attention (EMGA) 模块
    基于 MAD 论文 中的图 4 和公式 (4)。
    """
    def __init__(self, channels):
        super().__init__()
        # 假设 F_a 和 F_m 具有相同的通道数 'channels'
        
        # 空间注意力路径
        self.spatial_attn_conv = nn.Conv2d(channels * 2, 1, kernel_size=1)
        
        # 通道注意力路径
        self.channel_attn_gap = nn.AdaptiveAvgPool2d(1)
        self.channel_attn_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, F_a, F_m):
        """
        F_a: 外观特征 [B, C, H, W]
        F_m: 运动特征 [B, C, H, W]
        """
        # 空间注意力 (Spatial Attention)
        concat = torch.cat([F_a, F_m], dim=1)
        psi_s = torch.sigmoid(self.spatial_attn_conv(concat)) # [B, 1, H, W]
        F_as = F_a * psi_s
        
        # 通道注意力 (Channel Attention)
        gap = self.channel_attn_gap(F_as)
        psi_c = torch.softmax(self.channel_attn_conv(gap), dim=1) # [B, C, 1, 1]
        
        # 融合与残差连接
        F_am = F_a + (F_as * psi_c)
        return F_am

class MADBackbone(nn.Module):
    """
    MAD 骨干网络
    基于 MAD 论文 中的图 3(b)。
    此实现确保了与 dagr HybridBackbone 兼容的 4 个尺度输出。
    """
    def __init__(self, t_a_channels=2, t_m_channels=2):
        super().__init__()
        
        # 定义每个阶段的输出通道
        self.channels_out = [64, 128, 256, 512] # P2, P3, P4, P5
        
        # --- 外观分支 (Appearance Branch) ---
        # 对应图 3(b) 中的 "Res Block"
        # 阶段 0 (P2, s=4)
        self.app_block_0 = conv_bn_relu(t_a_channels, self.channels_out[0], kernel_size=7, stride=4, padding=3)
        # 阶段 1 (P3, s=8)
        self.app_block_1 = conv_bn_relu(self.channels_out[0], self.channels_out[1], kernel_size=3, stride=2, padding=1)
        # 阶段 2 (P4, s=16)
        self.app_block_2 = conv_bn_relu(self.channels_out[1], self.channels_out[2], kernel_size=3, stride=2, padding=1)
        # 阶段 3 (P5, s=32)
        self.app_block_3 = conv_bn_relu(self.channels_out[2], self.channels_out[3], kernel_size=3, stride=2, padding=1)
        
        # --- 运动分支 (Motion Branch) ---
        # 对应图 3(b) 中的 "Conv"
        # 阶段 0 (P2, s=4)
        self.mot_block_0 = conv_bn_relu(t_m_channels, self.channels_out[0], kernel_size=7, stride=4, padding=3)
        # 阶段 1 (P3, s=8)
        self.mot_block_1 = conv_bn_relu(self.channels_out[0], self.channels_out[1], kernel_size=3, stride=2, padding=1)
        # 阶段 2 (P4, s=16)
        self.mot_block_2 = conv_bn_relu(self.channels_out[1], self.channels_out[2], kernel_size=3, stride=2, padding=1)
        # 阶段 3 (P5, s=32)
        self.mot_block_3 = conv_bn_relu(self.channels_out[2], self.channels_out[3], kernel_size=3, stride=2, padding=1)

        # --- EMGA 融合模块 ---
        self.emga_0 = EMGA(self.channels_out[0])
        self.emga_1 = EMGA(self.channels_out[1])
        self.emga_2 = EMGA(self.channels_out[2])
        self.emga_3 = EMGA(self.channels_out[3])

    def forward(self, T_a, T_m):
        # T_a: [B, 2, H, W]
        # T_m: [B, 2, H, W]
        
        mad_feats = []
        
        # 阶段 0 (P2, s=4)
        F_a_0 = self.app_block_0(T_a) # Res Block-0
        F_m_0 = self.mot_block_0(T_m) # Conv-0
        F_am_0 = self.emga_0(F_a_0, F_m_0) # EMGA-0
        mad_feats.append(F_am_0)
        
        # 阶段 1 (P3, s=8)
        # 论文中 F_am_0 馈入 Res Block-1
        F_a_1 = self.app_block_1(F_am_0) # Res Block-1
        F_m_1 = self.mot_block_1(F_m_0) # Conv-1
        F_am_1 = self.emga_1(F_a_1, F_m_1) # EMGA-1
        mad_feats.append(F_am_1)
        
        # 阶段 2 (P4, s=16)
        F_a_2 = self.app_block_2(F_am_1) # Res Block-2
        F_m_2 = self.mot_block_2(F_m_1) # Conv-2
        F_am_2 = self.emga_2(F_a_2, F_m_2) # EMGA-2
        mad_feats.append(F_am_2)
        
        # 阶段 3 (P5, s=32)
        F_a_3 = self.app_block_3(F_am_2) # Res Block-3
        F_m_3 = self.mot_block_3(F_m_2) # Conv-3
        F_am_3 = self.emga_3(F_a_3, F_m_3) # EMGA-3
        mad_feats.append(F_am_3)
        
        # 返回 [mad_p2, mad_p3, mad_p4, mad_p5]
        # 形状: [B, 64, H/4, W/4], [B, 128, H/8, W/8], [B, 256, H/16, W/16], [B, 512, H/32, W/32]
        return mad_feats

    @property
    def out_channels(self):
        return self.channels_out

    @property
    def strides(self):
        return [4, 8, 16, 32]