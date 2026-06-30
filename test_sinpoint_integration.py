"""
Test script to verify SinPoint integration into the DGPointMamba project.
"""

import torch
from utils.sinpoint_augmentation import SinPoint


def test_sinpoint_integration():
    """Test if SinPoint can be imported and used correctly."""
    
    print("="*60)
    print("Testing SinPoint Integration")
    print("="*60)
    
    # Create configuration
    class SinPointArgs:
        def __init__(self):
            self.A = 0.8
            self.w = 3.0
            self.rand_center_num = 4
            self.sample = "RPS"
            self.isCat = False
            self.shuffle = False
    
    # Initialize SinPoint
    try:
        sinpoint_aug = SinPoint(SinPointArgs())
        print("✓ SinPoint initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize SinPoint: {e}")
        return False
    
    # Create dummy data
    batch_size = 4
    num_points = 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dummy_data = torch.randn(batch_size, num_points, 3).to(device)
    print(f"✓ Created dummy point cloud: shape={dummy_data.shape}, device={dummy_data.device}")
    
    # Test augmentation
    try:
        aug_data, _ = sinpoint_aug.Sin(dummy_data)
        print(f"✓ Augmentation successful: shape={aug_data.shape}, device={aug_data.device}")
    except Exception as e:
        print(f"✗ Augmentation failed: {e}")
        return False
    
    # Verify shape and device consistency
    if aug_data.shape == dummy_data.shape:
        print("✓ Shape consistency verified")
    else:
        print(f"✗ Shape mismatch: {aug_data.shape} vs {dummy_data.shape}")
        return False
    
    if aug_data.device == dummy_data.device:
        print("✓ Device consistency verified")
    else:
        print(f"✗ Device mismatch: {aug_data.device} vs {dummy_data.device}")
        return False
    
    # Test Local augmentation
    try:
        local_aug_data = sinpoint_aug.Local(dummy_data)
        print(f"✓ Local augmentation successful: shape={local_aug_data.shape}")
    except Exception as e:
        print(f"✗ Local augmentation failed: {e}")
        return False
    
    # Test Global augmentation
    try:
        global_aug_data = sinpoint_aug.Global(dummy_data)
        print(f"✓ Global augmentation successful: shape={global_aug_data.shape}")
    except Exception as e:
        print(f"✗ Global augmentation failed: {e}")
        return False
    
    print("="*60)
    print("All tests passed! SinPoint integration successful.")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_sinpoint_integration()
    exit(0 if success else 1)
