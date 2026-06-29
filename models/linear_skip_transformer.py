# models/linear_skip_transformer.py
import torch, torch.nn as nn
from .utils import MLP_Res, grouping_operation, query_knn

class LinearSkipTransformer(nn.Module):
    """
    线性跳跃Transformer模块
    
    基于KNN的局部注意力机制，用于点云特征提取。通过位置编码和局部特征聚合
    来增强点云特征表示。相比标准Transformer，该模块计算效率更高。
    """

    def __init__(self, in_channel, dim=256, n_knn=16,
                 pos_hidden_dim=64, agg='mean'):
        """
        初始化线性跳跃Transformer
        
        Args:
            in_channel (int): 输入特征通道数
            dim (int, optional): 内部特征维度，用于位置编码和注意力计算。默认为256
            n_knn (int, optional): K近邻的数量，用于构建局部邻域。默认为16
            pos_hidden_dim (int, optional): 位置编码的隐藏层维度。默认为64
            agg (str, optional): 特征聚合方式，可选 'mean'（平均）或 'max'（最大）。
                                 默认为 'mean'
        """
        super().__init__()
        self.n_knn = n_knn  # K近邻数量
        self.agg = agg  # 聚合方式     
    
        self.mlp_v = MLP_Res(in_dim=in_channel*2,
                             hidden_dim=in_channel,
                             out_dim=in_channel)
        self.lin_proj = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )
        self.local_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    @torch.cuda.amp.autocast(False)  # 禁用自动混合精度，确保计算精度
    def forward(self, pos, key, query, include_self=True):
        """
        前向传播
        
        Args:
            pos (torch.Tensor): 点云坐标，形状为 (B, 3, N)，其中B为批次大小，N为点数
            key (torch.Tensor): 键特征，形状为 (B, in_channel, N)
            query (torch.Tensor): 查询特征，形状为 (B, in_channel, N)
            include_self (bool, optional): 是否在KNN中包含点自身。默认为True
        
        Returns:
            torch.Tensor: 增强后的特征，形状为 (B, in_channel, N)
        """

        value = self.mlp_v(torch.cat([key, query], dim=1))  
        identity = value                                     

        v = self.lin_proj(value)                            


        B, dim, N = v.shape
        xyz = pos.permute(0,2,1).contiguous()              
        idx_knn = query_knn(self.n_knn, xyz, xyz,
                            include_self=include_self)       # (B,N,n_knn)

        v_group = grouping_operation(v, idx_knn)             # (B,dim,N,n_knn)

        pos_rel = pos.view(B, 3, N, 1) - grouping_operation(pos, idx_knn)
        pos_enc = self.pos_mlp(pos_rel)                      # (B,dim,N,n_knn)

        feat = self.local_mlp(v_group + pos_enc)             # (B,dim,N,n_knn)

        if self.agg == 'mean':
            agg = feat.mean(dim=-1)                          # (B,dim,N)
        else:
            agg = feat.max(dim=-1)[0]

        out = self.conv_end(agg)                             # (B,C,N)
        return out + identity
