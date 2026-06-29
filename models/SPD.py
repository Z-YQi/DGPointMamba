# -*- coding: utf-8 -*-
# @Author: Peng Xiang
import torch
import torch.nn as nn
from models.utils import MLP_Res, MLP_CONV
from models.linear_skip_transformer import LinearSkipTransformer


class SPD(nn.Module):
    """
    Snowflake Point Deconvolution (SPD) 模块
    
    用于点云上采样的模块，通过点级分裂（point-wise splitting）和特征细化来生成更高分辨率的点云。
    这是Snowflake点云补全方法中的核心上采样模块。
    """
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1, bounding=True, global_feat=True):
        """
        初始化SPD模块
        
        Args:
            dim_feat (int, optional): 全局特征的维度。默认为512
            up_factor (int, optional): 上采样因子，即每个点生成的新点数。默认为2
            i (int, optional): 当前上采样层的索引，用于控制位移的缩放。默认为0
            radius (float, optional): 半径参数，用于控制位移的边界。默认为1
            bounding (bool, optional): 是否对位移进行边界限制（使用tanh）。
                                        如果为True，位移会被限制在[-1/radius^i, 1/radius^i]范围内。
                                        默认为True
            global_feat (bool, optional): 是否使用全局特征。如果为True，会使用全局特征增强局部特征。
                                          默认为True
        """
        super(SPD, self).__init__()
        self.i = i  # 层索引
        self.up_factor = up_factor  # 上采样因子

        self.bounding = bounding  # 是否限制位移边界
        self.radius = radius  # 半径参数

        self.global_feat = global_feat  # 是否使用全局特征
        self.ps_dim = 32 if global_feat else 64  # 点分裂特征的维度

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat if self.global_feat else 128, layer_dims=[256, 128])

        self.skip_transformer = LinearSkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, self.ps_dim])
        self.ps = nn.ConvTranspose1d(self.ps_dim, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global=None, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1) if self.global_feat else feat_1
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = self.mlp_delta(torch.relu(K_curr))
        if self.bounding:
            delta = torch.tanh(delta) / self.radius**self.i  # (B, 3, N_prev * up_factor)

        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, K_curr