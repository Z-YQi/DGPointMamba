"""
Test script for SinPoint augmentation visualization.

This script loads data from DAPointMamba's Source Dataloader, applies SinPoint augmentation,
and saves both original and augmented point clouds as PLY files for visual comparison.

Usage:
    python test_augmentation_vis.py
    python test_augmentation_vis.py --config cfgs/ModelNet_models/DAPointMamba.yaml --batch_size 8
"""

import torch
import numpy as np
import open3d as o3d
import argparse
import os
from tools import builder
from utils.config import cfg_from_yaml_file
from utils.sinpoint_augmentation import SinPoint


def save_point_cloud(points, output_dir, filename):
    """
    Save a single point cloud as PLY file.
    
    Args:
        points (np.ndarray): Point cloud data, shape [N, 3]
        output_dir (str): Output directory
        filename (str): Output filename
    
    Returns:
        str: Full path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape [N, 3], got {points.shape}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    filepath = os.path.join(output_dir, filename)
    o3d.io.write_point_cloud(filepath, pcd)
    
    return filepath


def save_batch_point_clouds(batch_points, output_dir, prefix):
    """
    Save a batch of point clouds as PLY files.
    
    Args:
        batch_points (torch.Tensor or np.ndarray): Batch of point clouds, shape [B, N, 3]
        output_dir (str): Output directory
        prefix (str): Filename prefix ('original' or 'augmented')
    
    Returns:
        list: List of saved file paths
    """
    # Convert to numpy if needed
    if isinstance(batch_points, torch.Tensor):
        batch_points = batch_points.detach().cpu().double().numpy()
    elif isinstance(batch_points, np.ndarray):
        batch_points = batch_points.astype(np.float64).copy()
    else:
        raise TypeError(f"Unsupported type: {type(batch_points)}")
    
    if batch_points.ndim != 3 or batch_points.shape[2] != 3:
        raise ValueError(f"Batch must have shape [B, N, 3], got {batch_points.shape}")
    
    saved_files = []
    for i, points in enumerate(batch_points):
        filename = f"point_cloud_{i}_{prefix}.ply"
        filepath = save_point_cloud(points, output_dir, filename)
        saved_files.append(filepath)
    
    return saved_files


def main():
    """Main function for testing SinPoint augmentation visualization."""
    
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description='Test SinPoint Augmentation Visualization')
    parser.add_argument('--config', type=str, 
                       default='cfgs/ModelNet_models/DAPointMamba.yaml',
                       help='Config file path')
    parser.add_argument('--output_dir', type=str, 
                       default='augmentation_visualization',
                       help='Output directory for PLY files')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Number of samples to visualize')
    args = parser.parse_args()
    
    print("="*80)
    print("SinPoint Augmentation Visualization Test")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)
    
    # 2. Load configuration file
    print("\n[1/7] Loading configuration file...")
    config = cfg_from_yaml_file(args.config)
    print(f"✓ Configuration loaded successfully")
    print(f"  - Dataset: {config.dataset.train._base_.NAME}")
    print(f"  - Class choice: {config.dataset.train._base_.CLASS_CHOICE}")
    print(f"  - Virtual dataset: {config.dataset.train.virtual_dataset}")
    
    # 3. Create temporary args object (simulate main.py's args)
    print("\n[2/7] Creating dataloader arguments...")
    class TempArgs:
        def __init__(self):
            self.local_rank = 0
            self.num_workers = 0  # Set to 0 to avoid multiprocessing issues
            self.distributed = False
    
    temp_args = TempArgs()
    
    # Add batch size to config if not present
    if not hasattr(config.dataset.train, 'others'):
        config.dataset.train.others = {}
    if not hasattr(config.dataset.train.others, 'bs'):
        config.dataset.train.others.bs = args.batch_size
    
    print("✓ Temporary args created")
    
    # 4. Build Source Dataloader
    print("\n[3/7] Building Source Dataloader...")
    _, train_dataloader = builder.virtual_dataset_builder(temp_args, config.dataset.train)
    print(f"✓ Dataloader created successfully")
    print(f"  - Total batches: {len(train_dataloader)}")
    
    # 5. Initialize SinPoint
    print("\n[4/7] Initializing SinPoint augmentation...")
    class SinPointArgs:
        def __init__(self):
            self.A = 0.8              # Amplitude parameter
            self.w = 3.0              # Frequency parameter
            self.rand_center_num = 4  # Random center point count
            self.sample = "RPS"       # Random Point Sampling
            self.isCat = False        # Don't concatenate original data
            self.shuffle = False      # Don't shuffle
    
    sinpoint_aug = SinPoint(SinPointArgs())
    print("✓ SinPoint initialized successfully")
    print(f"  - A (amplitude): {sinpoint_aug.A}")
    print(f"  - w (frequency): {sinpoint_aug.w}")
    print(f"  - Random centers: {sinpoint_aug.rand_center_num}")
    print(f"  - Sampling method: {sinpoint_aug.sample}")
    
    # 6. Get one batch of data
    print("\n[5/7] Loading one batch from dataloader...")
    data_iter = iter(train_dataloader)
    source_data = next(data_iter)
    
    # 7. Extract source_partial (based on data format)
    # CRNShapeNet returns: (gt, partial, index)
    source_gt, source_partial, source_index = source_data
    print(f"✓ Data loaded successfully")
    print(f"  - Ground truth shape: {source_gt.shape}")
    print(f"  - Partial shape: {source_partial.shape}")
    
    # Handle potential extra dimensions and limit batch size
    source_partial = source_partial.squeeze()
    if source_partial.ndim == 2:
        # Single sample, add batch dimension
        source_partial = source_partial.unsqueeze(0)
    
    # Limit batch size
    if source_partial.shape[0] > args.batch_size:
        source_partial = source_partial[:args.batch_size]
        print(f"  - Limited to {args.batch_size} samples")
    
    # 8. Apply SinPoint augmentation
    print("\n[6/7] Applying SinPoint augmentation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_partial = source_partial.to(device)
    
    print(f"  - Source partial shape: {source_partial.shape}")
    print(f"  - Device: {source_partial.device}")
    
    aug_partial, _ = sinpoint_aug.Sin(source_partial)
    
    print(f"✓ Augmentation completed")
    print(f"  - Augmented partial shape: {aug_partial.shape}")
    print(f"  - Device: {aug_partial.device}")
    
    # Verify shape consistency
    assert source_partial.shape == aug_partial.shape, \
        f"Shape mismatch: {source_partial.shape} vs {aug_partial.shape}"
    assert source_partial.device == aug_partial.device, \
        f"Device mismatch: {source_partial.device} vs {aug_partial.device}"
    print("  - Shape and device consistency verified ✓")
    
    # 9. Save original and augmented point clouds
    print(f"\n[7/7] Saving point clouds to: {args.output_dir}")
    
    original_files = save_batch_point_clouds(source_partial, args.output_dir, "original")
    augmented_files = save_batch_point_clouds(aug_partial, args.output_dir, "augmented")
    
    # 10. Print save information
    print(f"\n✓ Saved {len(original_files)} original point clouds:")
    for f in original_files:
        print(f"    - {f}")
    
    print(f"\n✓ Saved {len(augmented_files)} augmented point clouds:")
    for f in augmented_files:
        print(f"    - {f}")
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Files saved to: {os.path.abspath(args.output_dir)}")
    print("="*80)


if __name__ == "__main__":
    main()
