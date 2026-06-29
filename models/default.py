import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order=["z","z-trans","hilbert","hilbert-trans"]):
    #assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    assert order in {"z","z-trans","hilbert","hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    """
    Z-order曲线编码函数
    
    将3D网格坐标编码为Z-order曲线的一维编码值。Z-order曲线是一种空间填充曲线，
    能够将多维空间中的点映射到一维空间，同时保持空间局部性。
    
    Args:
        grid_coord (torch.Tensor): 网格坐标，形状为 (N, 3)，其中N为点的数量
        depth (int, optional): 编码深度（位数），决定了编码的精度和范围。
                               默认为16，表示每个坐标轴使用16位编码
    
    Returns:
        torch.Tensor: Z-order编码值，形状为 (N,)，每个值对应一个点的编码
    """
    # 提取x、y、z坐标并转换为长整型（整数坐标）
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # 注意：这里不支持批次维度，批次编码在Point类中维护
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    """
    Z-order曲线解码函数
    
    将Z-order编码值解码回3D网格坐标。这是z_order_encode的逆操作。
    
    Args:
        code (torch.Tensor): Z-order编码值，形状为 (N,)，其中N为点的数量
        depth (int): 编码深度（位数），必须与编码时使用的depth相同
    
    Returns:
        torch.Tensor: 解码后的网格坐标，形状为 (N, 3)，其中N为点的数量
    """
    # 解码得到x、y、z坐标
    x, y, z = z_order_decode_(code, depth=depth)
    # 将x、y、z坐标堆叠成 (N, 3) 的形状
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
