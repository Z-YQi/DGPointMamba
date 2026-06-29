# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union

#LUT = Look-Up Table（查找表）
class KeyLUT:
    """
    Z-order编码查找表类
    
    预计算编码和解码查找表（Look-Up Table, LUT），用于加速Z-order编码/解码过程。
    通过查表替代循环计算，大幅提升性能。
    """
    def __init__(self):
        """
        初始化查找表
        
        预计算0-255的x、y、z坐标编码表，以及0-511的键值解码表。
        这些表存储在CPU设备上，需要时可以转移到其他设备（如GPU）。
        """
        r256 = torch.arange(256, dtype=torch.int64)  # 0-255的序列，用于生成8位编码表
        r512 = torch.arange(512, dtype=torch.int64)  # 0-511的序列，用于生成9位解码表
        zero = torch.zeros(256, dtype=torch.int64)  # 全零张量，作为占位符
        device = torch.device("cpu")

        # 预计算编码查找表：分别计算x、y、z坐标的8位编码表
        # EX: x坐标编码表，EY: y坐标编码表，EZ: z坐标编码表
        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),  # x坐标编码表（depth=8）
                self.xyz2key(zero, r256, zero, 8),  # y坐标编码表（depth=8）
                self.xyz2key(zero, zero, r256, 8),  # z坐标编码表（depth=8）
            )
        }
        # 预计算解码查找表：9位键值解码表（可解码3个坐标轴）
        self._decode = {device: self.key2xyz(r512, 9)}

    #编码查找表
    def encode_lut(self, device=torch.device("cpu")):
        """
        获取指定设备的编码查找表
        
        Args:
            device: 目标设备，默认为CPU
        
        Returns:
            tuple: (EX, EY, EZ) 编码查找表元组，分别对应x、y、z坐标的编码表
        """
        if device not in self._encode:
            # 如果目标设备没有查找表，从CPU复制过去
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        """
        获取指定设备的解码查找表
        
        Args:
            device: 目标设备，默认为CPU
        
        Returns:
            tuple: (DX, DY, DZ) 解码查找表元组，分别对应x、y、z坐标的解码表
        """
        if device not in self._decode:
            # 如果目标设备没有查找表，从CPU复制过去
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        """
        将xyz坐标编码为Z-order键值（基础实现，用于生成查找表）
        
        Args:
            x (torch.Tensor): x坐标
            y (torch.Tensor): y坐标
            z (torch.Tensor): z坐标
            depth (int): 编码深度（位数）
        
        Returns:
            torch.Tensor: Z-order编码键值
        """
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i  # 第i位的掩码
            # 交错排列：x的第i位放在键的第(3*i+2)位，y的第i位放在第(3*i+1)位，z的第i位放在第(3*i+0)位
            key = (
                key
                | ((x & mask) << (2 * i + 2))  # x坐标位左移(2*i+2)位
                | ((y & mask) << (2 * i + 1))  # y坐标位左移(2*i+1)位
                | ((z & mask) << (2 * i + 0))  # z坐标位左移(2*i+0)位
            )
        return key

    def key2xyz(self, key, depth):
        """
        将Z-order键值解码为xyz坐标（基础实现，用于生成查找表）
        
        这是xyz2key的逆操作，从交错的键值中提取x、y、z坐标的各个位。
        
        Args:
            key (torch.Tensor): Z-order编码键值
            depth (int): 编码深度（位数）
        
        Returns:
            tuple: (x, y, z) 解码后的坐标
        """
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            # 从键值中提取x、y、z的第i位
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))  # 提取x的第i位
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))  # 提取y的第i位
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))  # 提取z的第i位
        return x, y, z


# 全局查找表实例，用于加速编码/解码
_key_lut = KeyLUT()


def xyz2key(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    b: Optional[Union[torch.Tensor, int]] = None,
    depth: int = 16,
):
    """
    将xyz坐标编码为Z-order键值（高效版本，使用查找表）
    
    基于预计算的查找表进行编码，速度远快于基于循环的方法。
    支持深度最大为16位，如果depth>8，则分两段处理（低8位和高位）。
    
    Args:
        x (torch.Tensor): x坐标张量
        y (torch.Tensor): y坐标张量
        z (torch.Tensor): z坐标张量
        b (torch.Tensor or int, optional): 批次索引，应小于32768。
                                           如果为torch.Tensor，其大小必须与x、y、z相同。
                                           默认为None
        depth (int, optional): Z-order键的深度（位数），必须小于17。默认为16
    
    Returns:
        torch.Tensor: Z-order编码键值，形状与输入坐标相同
    """
    # 获取编码查找表（根据输入设备自动选择）
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()  # 转换为长整型

    # 处理低8位（depth <= 8）或低depth位（depth < 8）
    mask = 255 if depth > 8 else (1 << depth) - 1  # 掩码：低8位或低depth位
    # 使用查找表进行编码：分别查表后按位或合并
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    
    # 如果depth > 8，需要处理高位部分
    if depth > 8:
        mask = (1 << (depth - 8)) - 1  # 高位掩码
        # 处理高8位：右移8位后查表编码
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        # 将高8位编码左移24位后与低8位合并
        key = key16 << 24 | key

    # 如果提供了批次索引，将其编码到键值的高位（第48位及以上）
    if b is not None:
        b = b.long()
        key = b << 48 | key  # 批次索引左移48位后与键值合并

    return key


def key2xyz(key: torch.Tensor, depth: int = 16):
    """
    将Z-order键值解码为xyz坐标和批次索引（高效版本，使用查找表）
    
    基于预计算的查找表进行解码，这是xyz2key的逆操作。
    支持深度最大为16位，分块处理以提高效率。
    
    Args:
        key (torch.Tensor): Z-order编码键值
        depth (int, optional): Z-order键的深度（位数），必须小于17。默认为16
    
    Returns:
        tuple: (x, y, z, b) 解码后的坐标和批次索引
            - x (torch.Tensor): x坐标
            - y (torch.Tensor): y坐标
            - z (torch.Tensor): z坐标
            - b (torch.Tensor): 批次索引（从键值的高48位提取）
    """
    # 获取解码查找表（根据输入设备自动选择）
    DX, DY, DZ = _key_lut.decode_lut(key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    # 提取批次索引（键值的高48位）
    b = key >> 48
    # 清除批次索引位，保留低48位的坐标编码
    key = key & ((1 << 48) - 1)

    # 计算需要处理的块数：每9位可以编码3个坐标轴的3位
    # 例如depth=16时，需要处理 (16+2)//3 = 6 块
    n = (depth + 2) // 3
    for i in range(n):
        # 提取第i块的9位键值（可编码3个坐标轴各3位）
        k = key >> (i * 9) & 511  # 511 = 2^9 - 1，用于提取9位
        # 使用查找表解码，并将结果左移到正确位置
        x = x | (DX[k] << (i * 3))  # x坐标的第i块（3位）
        y = y | (DY[k] << (i * 3))  # y坐标的第i块（3位）
        z = z | (DZ[k] << (i * 3))  # z坐标的第i块（3位）

    return x, y, z, b
