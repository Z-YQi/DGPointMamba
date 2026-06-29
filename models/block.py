# Copyright (c) 2023, Tri Dao, Albert Gu.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from timm.models.layers import DropPath



class Block(nn.Module):
    """
    基础Transformer块，包装了mixer（如Mamba）与LayerNorm/RMSNorm和残差连接
    
    该Block的结构与标准的prenorm Transformer块略有不同。
    标准块结构：LN -> MHA/MLP -> Add
    [参考: https://arxiv.org/abs/2002.04745]
    这里使用：Add -> LN -> Mixer，同时返回hidden_states（mixer的输出）和residual。
    这主要是为了性能优化，可以融合add和LayerNorm操作。
    残差连接需要被提供（除了第一个块）。
    """
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        初始化Block
        
        Args:
            dim (int): 特征维度
            mixer_cls: Mixer类的构造函数（如Mamba），用于创建mixer实例
            norm_cls: 归一化类，默认为nn.LayerNorm，也可以是RMSNorm
            fused_add_norm (bool): 是否融合add和norm操作以提升性能。默认为False
            residual_in_fp32 (bool): 是否在fp32精度下计算残差连接。默认为False
            drop_path (float): DropPath的丢弃率，用于正则化。默认为0.0
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)  # 创建mixer实例（如Mamba）
        self.norm = norm_cls(dim)  # 创建归一化层
        
        # DropPath：随机深度正则化，训练时随机跳过某些路径
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        """
        前向传播，通过编码器层处理输入
        
        Args:
            hidden_states (Tensor): 输入序列，形状为 (B, L, D)，其中B为批次大小，
                                    L为序列长度，D为特征维度
            residual (Optional[Tensor]): 残差连接，用于与当前输出相加。
                                         如果为None，则使用hidden_states作为残差。
                                         默认为None
            inference_params: 推理参数，用于优化推理时的性能。默认为None
        
        Returns:
            hidden_states (Tensor): mixer的输出，形状为 (B, L, D)
            residual (Tensor): 更新后的残差，形状为 (B, L, D)
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
