import math
import random
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from mamba_ssm.modules.mamba_simple import Mamba
from knn_cuda import KNN
from .block import Block
from .build import MODELS
from .utils import MLP_Res, fps_subsample
from .SPD import SPD
from .patch_group import PatchGroup

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# --- Spatial SSM ---
class SpatialSSM(nn.Module):
    """
    空间状态空间模型（Spatial State Space Model）
    
    通过深度卷积和余弦相似度计算源点云和目标点云之间的空间对齐权重，
    用于域适应任务中的空间特征对齐。
    """
    def __init__(self, dim):
        """
        初始化空间SSM模块
        
        Args:
            dim (int): 特征维度
        """
        super().__init__()
        # 深度可分离卷积，用于提取空间特征
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # 余弦相似度，用于计算对齐权重
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def forward(self, x_s, x_t):
        """
        前向传播，计算空间对齐权重并应用到特征上
        
        Args:
            x_s (torch.Tensor): 源点云特征，形状为 (B, D, G)，其中B为批次大小，
                                D为特征维度，G为组数
            x_t (torch.Tensor): 目标点云特征，形状为 (B, D, G)
        
        Returns:
            x_s * w (torch.Tensor): 加权后的源特征，形状为 (B, D, G)
            x_t * w (torch.Tensor): 加权后的目标特征，形状为 (B, D, G)
        """
        ds = self.dw(x_s)  # 源特征的空间卷积
        dt = self.dw(x_t)  # 目标特征的空间卷积
        w  = self.cos(ds, dt).unsqueeze(1)   # (B,1,G) - 计算对齐权重
        return x_s * w, x_t * w
    
def feature_perturbation(x, epsilon=0.1):
    """
    特征扰动函数，用于数据增强
    
    Args:
        x (torch.Tensor): 输入特征
        epsilon (float, optional): 噪声强度。默认为0.1
    
    Returns:
        torch.Tensor: 扰动后的特征（50%概率）或原始特征
    """
    if random.random() < 0.5:  
        noise = torch.randn_like(x) * epsilon
        return x + noise
    return x
   
class ChannelSSM(nn.Module):
    """
    通道状态空间模型（Channel State Space Model）
    
    通过通道级别的特征混合和对齐，实现源域和目标域之间的通道对齐。
    使用自适应对齐强度来动态调整对齐程度。
    """
    def __init__(self, dim, segments=4):
        """
        初始化通道SSM模块
        
        Args:
            dim (int): 特征维度
            segments (int, optional): 通道分段数量，用于特征混合。默认为4
        """
        super().__init__()
        self.seg = segments  # 通道分段数
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)  # 余弦相似度
        
        self.alignment_strength = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_s, x_t):
        B, D, G = x_s.shape
        

        global_s = x_s.mean(dim=2)  # [B, D]
        global_t = x_t.mean(dim=2)
        strength_input = torch.cat([global_s, global_t], dim=1)
        alignment_strength = self.alignment_strength(strength_input)  # [B, 1]

        #切成四份
        s_chunks = x_s.chunk(self.seg, 1)
        t_chunks = x_t.chunk(self.seg, 1)
        
        #对应论文中的Segment Cross-mixing
        x_mix = torch.cat([s_chunks[0], t_chunks[1], s_chunks[2], t_chunks[3]], 1)
        x_t_mix = torch.cat([t_chunks[0], s_chunks[1], t_chunks[2], s_chunks[3]], 1)

        w = self.cos(x_mix, x_t_mix).unsqueeze(-1)  # [B, G, 1]

        adaptive_w = w * alignment_strength.unsqueeze(-1)  # [B, G, 1]
        
        return x_s * adaptive_w, x_t * adaptive_w

    
class SeedGenerator(nn.Module):
    """
    种子点生成器
    
    从全局特征生成初始点云种子，用于点云补全的粗粒度生成阶段。
    通过转置卷积和多个MLP层逐步细化生成点云坐标。
    """
    def __init__(self, dim_feat=256, num_pc=128):
        """
        初始化种子生成器
        
        Args:
            dim_feat (int, optional): 输入特征维度。默认为256
            num_pc (int, optional): 生成的初始点云数量。默认为128
        """
        super(SeedGenerator, self).__init__()
        # 转置卷积：从1个特征点扩展到num_pc个点
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        # 残差MLP层，用于特征细化
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        # 最终输出层：生成3D坐标
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        前向传播，从全局特征生成点云种子
        
        Args:
            feat (torch.Tensor): 全局特征，形状为 (b, dim_feat, 1)
        
        Returns:
            torch.Tensor: 生成的点云坐标，形状为 (b, 3, num_pc)
        """
        x1 = self.ps(feat)  
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  
        completion = self.mlp_4(x3)  
        return completion

class Encoder(nn.Module):
    """
    点云组编码器
    
    将点云组编码为特征向量。首先对每个点进行特征提取，然后通过最大池化
    和全局特征融合，最终输出每个组的全局特征表示。
    """
    def __init__(self, encoder_channel):
        """
        初始化编码器
        
        Args:
            encoder_channel (int): 输出特征通道数
        """
        super().__init__()
        self.encoder_channel = encoder_channel
        # 第一层卷积：从3D坐标提取特征
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256,512,1),
        )
        # 第二层卷积：融合全局和局部特征
        self.second_conv = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
        前向传播，将点云组编码为特征
        
        Args:
            point_groups (torch.Tensor): 点云组，形状为 (B, G, N, 3)，
                                        其中B为批次大小，G为组数，N为每组点数
        
        Returns:
            feature_global (torch.Tensor): 每个组的全局特征，形状为 (B, G, encoder_channel)
        """
        bs, g, n, _ = point_groups.shape 
        point_groups = point_groups.reshape(bs * g, n, 3) 
        feature = self.first_conv(point_groups.transpose(2, 1))  
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  
        feature = self.second_conv(feature) 
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  
        return feature_global.reshape(bs, g, self.encoder_channel)



def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, 
                residual_in_fp32=False, fused_add_norm=False, layer_idx=None, 
                drop_path=0., device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    """
    Mixer模型（基于Mamba的序列模型）
    
    由多个Block层堆叠而成，用于处理序列化的点云特征。每个Block包含
    Mamba mixer、归一化层和残差连接。支持RMSNorm和LayerNorm两种归一化方式。
    """
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        """
        初始化MixerModel
        
        Args:
            d_model (int): 模型特征维度
            n_layer (int): Transformer层数（Block数量）
            ssm_cfg (dict, optional): Mamba SSM的配置参数。默认为None
            norm_epsilon (float, optional): 归一化层的epsilon值。默认为1e-5
            rms_norm (bool, optional): 是否使用RMSNorm替代LayerNorm。默认为False
            initializer_cfg (dict, optional): 权重初始化配置。默认为None
            fused_add_norm (bool, optional): 是否融合add和norm操作。默认为False
            residual_in_fp32 (bool, optional): 是否在fp32精度下计算残差。默认为False
            drop_out_in_block (float, optional): Block内的dropout率。默认为0.0
            drop_path (float, optional): DropPath的丢弃率。默认为0.1
            device: 设备类型
            dtype: 数据类型
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32 

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm #False
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        ) 

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        ) 

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        ) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        } 

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states) 
        if not self.fused_add_norm: 
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else: 
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class MambaDecoder(nn.Module):
    """
    Mamba解码器
    
    基于Mamba的序列解码器，用于处理编码后的点云特征序列。
    通过多层MixerModel来提取和细化特征表示。
    """
    def __init__(self, embed_dim=384, depth=4, norm_layer=nn.LayerNorm, config=None):
        """
        初始化Mamba解码器
        
        Args:
            embed_dim (int, optional): 嵌入维度。默认为384
            depth (int, optional): 解码器层数。默认为4
            norm_layer: 归一化层类型（未使用，保留用于兼容性）
            config: 配置对象，包含rms_norm、drop_path等参数
        """
        super().__init__()
        if hasattr(config, "use_external_dwconv_at_last"):
            self.use_external_dwconv_at_last = config.use_external_dwconv_at_last
        else: #False
            self.use_external_dwconv_at_last = False
        # 创建MixerModel作为解码器主体
        self.blocks = MixerModel(d_model=embed_dim,
                                 n_layer=depth,
                                 rms_norm=config.rms_norm,
                                 drop_path=config.drop_path) 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x
    

@MODELS.register_module()
class DGPointMamba(nn.Module):
    """
    DGPointMamba single-input completion backbone.
    """
    def __init__(self, config):
        """
        Initialize the source-only DGPointMamba completion backbone.
        """
        super().__init__()
        print_log(f'[DGPointMamba] ', logger='DGPointMamba')
        self.config = config
        self.trans_dim = config.trans_dim  # Transformer特征维度
        self.points = config.points  # 点云数量
        num_p0 = config.num_p0  # 初始点云数量
        dim_feat = config.num_p0  # 特征维度
        num_pc = config.num_pc  # 种子点数量
        radius = config.radius  # 半径参数
        bounding = config.bounding  # 是否限制边界
        up_factors = config.up_factors  # 上采样因子列表
        self.order_mode = config.order_mode  # 序列化模式


        self.encoder_dims = config.encoder_dims
        self.mamba_depth = config.mamba_depth
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

        self.norm = nn.LayerNorm(self.trans_dim)
        ##
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.MAE_decoder = MambaDecoder(
            embed_dim=self.trans_dim,
            depth=self.mamba_depth,
            config=config,
        )

        print_log(f'[DGPointMamba] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='DGPointMamba')


        # 点云分组模块：将点云分组为patches
        self.group_divider = PatchGroup(
            group_size=self.group_size,  # 每组点数
            use_serialization=True,  # 使用序列化
            scale=15.0,  # 坐标缩放因子
            order_mode=self.order_mode,
            enable_patch_shuffle=False  # 不启用patch打乱
        )

        self.decoder = Decoder(dim_feat=self.trans_dim, num_pc=num_pc, num_p0=num_p0, radius=radius, 
                               bounding=bounding, up_factors=up_factors)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    

    def forward(self, partial, **kwargs):
        neighborhoods, centers = self.group_divider(partial)
        group_input_tokens = self.encoder(neighborhoods)
        pos = self.pos_embed(centers)
        tokens = self.drop_out(group_input_tokens)

        out = self.MAE_decoder(tokens, pos)
        out = self.norm(out)  # [B, G, D]

        global_feat = torch.max(out, dim=1, keepdim=True)[0].transpose(1, 2)
        rebuild_points = self.decoder(global_feat, partial)
        aux = {
            "mamba_depth": self.mamba_depth,
            "order_mode": self.order_mode,
        }
        return rebuild_points, aux
    
class Decoder(nn.Module):
    """
    点云解码器
    
    从全局特征生成完整的点云。首先通过SeedGenerator生成粗粒度点云，
    然后通过多个SPD模块逐步上采样到目标分辨率。
    """
    def __init__(self, dim_feat=256, num_pc=128, num_p0=256,
                 radius=1, bounding=True, up_factors=None):
        """
        初始化解码器
        
        Args:
            dim_feat (int, optional): 特征维度。默认为256
            num_pc (int, optional): 种子点数量。默认为128
            num_p0 (int, optional): 初始点云数量。默认为256
            radius (float, optional): 半径参数。默认为1
            bounding (bool, optional): 是否限制边界。默认为True
            up_factors (list, optional): 上采样因子列表。默认为None（使用[1]）
        """
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        # 粗粒度生成器：从特征生成初始点云
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors): 
            uppers.append(SPD
            (dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)
        self.ln = nn.LayerNorm(dim_feat)


    def forward(self, feat, partial):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        feat = self.ln(feat.transpose(1, 2)).transpose(1, 2) 
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)  
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()#[32,3,512]
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd
                          
