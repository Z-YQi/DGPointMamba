import torch
from .default import encode as ptv3_encode  


class PatchGroup(torch.nn.Module):
    """
    点云补丁分组模块
    
    将点云按照空间序列化（Z-order或Hilbert曲线）的方式分组为多个补丁（patches），
    每个补丁包含固定数量的点。主要用于将点云转换为适合Transformer处理的序列格式。
    """

    def __init__(self, group_size, use_serialization=True, scale=10.0,
                 serialization_depth=16, order_mode="z", enable_patch_shuffle=False,
                 depth=None):
        """
        初始化补丁分组模块
        
        Args:
            group_size (int): 每个补丁组的大小，即每个组包含的点数
            use_serialization (bool, optional): 是否使用序列化方法对点进行排序。默认为True
            scale (float, optional): 坐标缩放因子，用于将点云坐标转换为网格坐标。
                                     在Z-order或Hilbert编码前，坐标会乘以该因子并取整。
                                     默认为10.0
            depth (int, optional): 空间编码的深度（位数），用于Z-order或Hilbert曲线编码。
                                   决定了编码的精度，通常为16位。默认为16
            enable_patch_shuffle (bool, optional): 是否在训练时启用补丁打乱，用于数据增强。
                                                    如果为True，训练时会随机打乱补丁的顺序。
                                                    默认为False
        """
        super().__init__()
        if depth is not None:
            serialization_depth = depth
        if order_mode != "z":
            raise ValueError(f"PatchGroup currently supports only order_mode='z', got {order_mode!r}")
        self.group_size = group_size
        self.use_serialization = use_serialization
        self.scale = scale
        self.depth = serialization_depth
        self.order_mode = order_mode
        self.enable_patch_shuffle = enable_patch_shuffle

    def forward(self, xyz_s, xyz_t=None):
        """
        前向传播，对点云进行分组
        
        Args:
            xyz_s (torch.Tensor): 源点云坐标，形状为 (B, N, 3)，其中B为批次大小，N为点数
            xyz_t (torch.Tensor, optional): 目标点云坐标，形状为 (B, N, 3)。
                                            如果提供，会使用统一的坐标最小值进行对齐。
                                            默认为None
        
        Returns:
            如果 xyz_t 为 None:
                grouped (torch.Tensor): 分组后的点云，形状为 (B, G, group_size, 3)，
                                        其中G为组数
                center (torch.Tensor): 每个组的中心点，形状为 (B, G, 3)
            
            如果 xyz_t 不为 None:
                grouped_s (torch.Tensor): 源点云分组结果，形状为 (B, G, group_size, 3)
                center_s (torch.Tensor): 源点云每组中心点，形状为 (B, G, 3)
                grouped_t (torch.Tensor): 目标点云分组结果，形状为 (B, G, group_size, 3)
                center_t (torch.Tensor): 目标点云每组中心点，形状为 (B, G, 3)
        """
        if xyz_t is None:
            grouped, center = self._serialize_and_group(xyz_s)
            return grouped, center
        else:
            # 使用统一的坐标最小值，确保源和目标点云使用相同的编码基准
            unified_min = torch.min(xyz_s.amin(dim=1, keepdim=True), xyz_t.amin(dim=1, keepdim=True))
            order = self.order_mode  # 使用Z-order编码
            grouped_s, center_s = self._serialize_and_group(xyz_s, order, coord_min=unified_min)
            grouped_t, center_t = self._serialize_and_group(xyz_t, order, coord_min=unified_min)
            return grouped_s, center_s, grouped_t, center_t

    def _serialize_and_group(self, xyz, order=None, coord_min=None):
        """
        对点云进行序列化和分组
        
        该方法首先将点云坐标转换为网格坐标，然后使用Z-order或Hilbert曲线进行空间编码，
        根据编码值对点进行排序，最后将排序后的点分组为固定大小的补丁。
        
        Args:
            xyz (torch.Tensor): 点云坐标，形状为 (B, N, 3)
            order (str, optional): 空间编码方式，可选 "z"、"z-trans"、"hilbert"、"hilbert-trans"。
                                   如果为None，则使用默认的Z-order编码。默认为None
            coord_min (torch.Tensor, optional): 坐标的最小值，用于坐标归一化。
                                                 如果为None，则从输入点云计算。默认为None
        
        Returns:
            grouped (torch.Tensor): 分组后的点云，形状为 (B, G, group_size, 3)
            center (torch.Tensor): 每个组的中心点坐标，形状为 (B, G, 3)
        """
        B, N, _ = xyz.shape
        G = N // self.group_size  # 计算组数
        if coord_min is None:
            coord_min = xyz.amin(dim=1, keepdim=True)  # 计算每个批次的最小坐标值
        
        '''
        数据示例
        # 假设有一个批次大小为2，每个批次3个点的点云
        xyz = torch.tensor([
            # 批次0: 3个点
            [[1.0, 2.0, 3.0],   # 点1
            [4.0, 5.0, 6.0],   # 点2
            [2.0, 1.0, 4.0]],  # 点3
            
            # 批次1: 3个点
            [[5.0, 6.0, 7.0],   # 点1
            [3.0, 2.0, 1.0],   # 点2
            [4.0, 5.0, 6.0]]   # 点3
        ])
        # 形状: (2, 3, 3) = (B, N, 3)

        # 计算每个批次的最小坐标值
        coord_min = xyz.amin(dim=1, keepdim=True)
        print(coord_min)
        # 输出:
        # tensor([[[1.0, 1.0, 3.0]],  # 批次0: x_min=1.0, y_min=1.0, z_min=3.0
        #         [[3.0, 2.0, 1.0]]])  # 批次1: x_min=3.0, y_min=2.0, z_min=1.0
        # 形状: (2, 1, 3) = (B, 1, 3)
        '''
        # 将坐标转换为网格坐标：先减去最小值归一化，再乘以缩放因子并取整
        #这里的xyz是整个数据集的，不是一个批次。coord_min是每个批次的最小值
        grid_coord = ((xyz - coord_min) * self.scale).int().reshape(-1, 3)#（B,N,3)->(B*N,3)
        
        # 对每个批次进行空间编码
        code_all = []
        for b in range(B):#把(B*N,3)分成B个(N,3)
            grid_b = grid_coord[b * N : (b + 1) * N]
            use_order = order if order is not None else self.order_mode
            if use_order != "z":
                raise ValueError(f"PatchGroup currently supports only order_mode='z', got {use_order!r}")
            # 使用Z-order或Hilbert曲线编码，生成每个点的编码值
            code_b = ptv3_encode(grid_b, batch=None, depth=self.depth, order=use_order)#传一个批次进来(N,3)
            #code_b 形状：(N,) 含义：一个批次中 N 个点的 Z-order 编码值
            # code_b.shape = (5,)  # 形状：5个元素（假设这里N=5）
            # 编码值存储在 code_b 的值中：
            # code_b = tensor([12345678, 23456789, 34567890, 45678901, 56789012])
            #            ↑         ↑         ↑         ↑         ↑
            #          点1编码   点2编码   点3编码   点4编码   点5编码
            code_all.append(code_b)
        
        code = torch.stack(code_all, dim=0)  # [B, N] - 堆叠所有批次的编码
        
        # 根据编码值对点进行排序，实现空间序列化
        sort_idx = code.argsort(dim=1)  # [B, N] - 排序索引
        xyz_sorted = torch.gather(xyz, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))
        '''
        # === 输入 ===（这个例子中B=1）
        xyz = [[1.0, 2.0, 3.0],   # 点0
            [5.0, 4.0, 6.0],   # 点1
            [2.0, 1.0, 4.0],   # 点2
            [4.0, 5.0, 7.0],   # 点3
            [3.0, 3.0, 5.0]]   # 点4

        code = [12345, 56789, 23456, 45678, 34567]
        #       点0    点1    点2    点3    点4

        # === 排序过程 ===
        sort_idx = code.argsort(dim=1)
        # sort_idx = [0, 2, 4, 3, 1]
        #             ↑  ↑  ↑  ↑  ↑
        #            按编码值从小到大排序后的原始索引

        # === 输出 ===
        xyz_sorted = [[1.0, 2.0, 3.0],   # 点0（编码值最小）
                    [2.0, 1.0, 4.0],   # 点2
                    [3.0, 3.0, 5.0],   # 点4
                    [4.0, 5.0, 7.0],   # 点3
                    [5.0, 4.0, 6.0]]   # 点1（编码值最大）
        '''
        
        # 将排序后的点分组为固定大小的补丁
        grouped = xyz_sorted[:, :G * self.group_size, :].reshape(B, G, self.group_size, 3)
        
        # 训练时可选：随机打乱补丁顺序，用于数据增强
        if self.training and self.enable_patch_shuffle:
            perm = torch.randperm(G, device=xyz.device)
            grouped = grouped[:, perm, :]
        
        # 计算每个组的中心点
        center = grouped.mean(dim=2)
        return grouped, center



