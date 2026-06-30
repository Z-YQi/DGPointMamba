"""
SinPoint Data Augmentation for Point Clouds
Adapted from: https://github.com/dhh1995/SinPoint

This module provides sinusoidal-based point cloud augmentation
for domain generalization in 3D point cloud tasks.
"""

import torch
import numpy as np


class SinPoint:
    """
    SinPoint augmentation class for point cloud data.
    
    Uses sinusoidal transformations to generate diverse augmented samples
    while maintaining the geometric structure of point clouds.
    
    Args:
        args: Configuration object with the following attributes:
            - A (float): Amplitude parameter for sinusoidal transformation
            - w (float): Frequency parameter for sinusoidal transformation
            - rand_center_num (int): Number of random centers for local augmentation
            - sample (str): Sampling method, 'RPS' (Random Point Sampling) or 'FPS' (Farthest Point Sampling)
            - isCat (bool): Whether to concatenate original and augmented data
            - shuffle (bool): Whether to shuffle concatenated data
    """

    def __init__(self, args):
        self.rand_center_num = args.rand_center_num
        self.w = args.w
        self.A = args.A
        self.sample = args.sample
        self.isCat = args.isCat
        self.shuffle = args.shuffle

    def Local(self, data):
        """
        Local sinusoidal augmentation based on multiple random centers.
        
        Args:
            data (torch.Tensor): Input point cloud, shape [B, N, 3]
        
        Returns:
            torch.Tensor: Augmented point cloud, shape [B, N, 3]
        """
        device = data.device
        B, N, C = data.shape
        if self.sample == "RPS":
            # Random Point Sampling: B * k
            idxs = self.generate_random_permutations_batch(B, N, self.rand_center_num, device=device)
        elif self.sample == "FPS":
            # Farthest Point Sampling: B * k
            idxs = self.farthest_point_sample(data, self.rand_center_num)
        else:
            raise ValueError(f"Unsupported SinPoint sampling method: {self.sample}")
        dist = torch.zeros_like(data).to(device)
        for i in range(self.rand_center_num):
            center = self.index_points(data, idxs[:, i]).unsqueeze(1)
            dist = dist + data - center
        dist = dist / self.rand_center_num
        w = -self.w + (self.w + self.w) * torch.rand([1, 1, C])
        A = -self.A + (self.A + self.A) * torch.rand([1, 1, C])
        move = A.to(device) * torch.sin(w.to(device) * dist)
        newdata = data + move
        return newdata

    def Global(self, data):
        """
        Global sinusoidal augmentation applied to entire point cloud.
        
        Args:
            data (torch.Tensor): Input point cloud, shape [B, N, 3]
        
        Returns:
            torch.Tensor: Augmented point cloud, shape [B, N, 3]
        """
        device = data.device
        B, N, C = data.shape
        newdata = torch.zeros_like(data)
        w = -self.w + (self.w + self.w) * torch.rand([1, 1, C])
        A = -self.A + (self.A + self.A) * torch.rand([1, 1, C])
        move = A.to(device) * torch.sin(w.to(device) * data)
        newdata = data + move
        return newdata

    def Sin(self, data, label=[]):
        """
        Main augmentation method with normalization.
        
        Args:
            data (torch.Tensor): Input point cloud, shape [B, N, 3]
            label (torch.Tensor, optional): Labels, shape [B]
        
        Returns:
            tuple: (augmented_data, labels)
                - augmented_data (torch.Tensor): Augmented point cloud, shape [B, N, 3] or [2B, N, 3] if isCat=True
                - labels (torch.Tensor): Labels, shape [B] or [2B] if isCat=True
        """
        B, _, _ = data.shape
        newdata, shift, scale = self.normalize_point_clouds(data)
        if self.rand_center_num == 0:
            newdata = self.Global(newdata)
        else:
            newdata = self.Local(newdata)
        newdata = newdata * scale + shift
        
        # Handle label processing
        if len(label) == 0:
            label = torch.zeros(B, dtype=torch.long, device=data.device)
        label = label.unsqueeze(1)
        
        if self.isCat:
            newdata = torch.cat([data, newdata], dim=0)
            label = torch.cat([label, label], dim=0)
            if self.shuffle:
                idxs = torch.randperm(B * 2, device=data.device)
                newdata = newdata[idxs, :, :]
                label = label[idxs, :]
        return newdata, label.squeeze(1)

    def index_points(self, points, idx):
        """
        Index points based on given indices.
        
        Args:
            points (torch.Tensor): Input points, shape [B, N, C]
            idx (torch.Tensor): Sample indices, shape [B, S]
        
        Returns:
            torch.Tensor: Indexed points, shape [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def farthest_point_sample(self, xyz, npoint):
        """
        Farthest Point Sampling (FPS).
        
        Args:
            xyz (torch.Tensor): Point cloud data, shape [B, N, 3]
            npoint (int): Number of samples
        
        Returns:
            torch.Tensor: Sampled point indices, shape [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = self.square_distance(xyz, centroid).squeeze(2)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids

    def generate_random_permutations_batch(self, B, N, npoint, device=None):
        """
        Generate random permutations for batch sampling.
        
        Args:
            B (int): Batch size
            N (int): Number of points
            npoint (int): Number of samples
        
        Returns:
            torch.Tensor: Sampled point indices, shape [B, npoint]
        """
        all_permutations = torch.stack([torch.randperm(N, device=device) for _ in range(B)])
        centroids = all_permutations[:, :npoint]
        return centroids

    def square_distance(self, src, dst):
        """
        Calculate Euclidean distance between each two points.
        
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1) + sum(dst**2,dim=-1) - 2*src^T*dst
        
        Args:
            src (torch.Tensor): Source points, shape [B, N, C]
            dst (torch.Tensor): Target points, shape [B, M, C]
        
        Returns:
            torch.Tensor: Per-point square distance, shape [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def normalize_point_clouds(self, pcs):
        """
        Normalize point clouds by centering and scaling.
        
        Args:
            pcs (torch.Tensor): Point clouds, shape [B, N, C]
        
        Returns:
            tuple: (normalized_pcs, shift, scale)
                - normalized_pcs (torch.Tensor): Normalized point clouds
                - shift (torch.Tensor): Mean values for each batch
                - scale (torch.Tensor): Standard deviation for each batch
        """
        B, N, C = pcs.shape
        shift = torch.mean(pcs, dim=1).unsqueeze(1)
        scale = torch.std(pcs.view(B, N * C), dim=1).unsqueeze(1).unsqueeze(1)
        newpcs = (pcs - shift) / scale
        return newpcs, shift, scale
