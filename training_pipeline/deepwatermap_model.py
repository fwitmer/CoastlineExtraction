"""
4-Channel DeepWaterMap Architecture, Loss Functions, and Dataset Loader for Planet Labs Data

Implements the DeepWaterMap architecture adapted for 4-band Planet Labs satellite imagery (Blue, Green, Red, NIR).
Includes BCEDiceLoss and comprehensive online spatial & photometric data augmentations.
"""

import os
import glob
import re
import numpy as np
import rasterio as rio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for coastal boundary segmentation.
    Penalizes both pixel-level classification errors and overall mask boundary dissimilarity.
    """
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()
        
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

class ConvBlockDWM(nn.Module):
    """Convolution block with BatchNorm and optional ReLU activation."""
    def __init__(self, in_c, out_c, k_size, stride=1, use_relu=True):
        super().__init__()
        padding = k_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.01)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = F.relu(x)
        return x

class DownscalingUnitDWM(nn.Module):
    """Downscaling unit with residual addition."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = ConvBlockDWM(in_c, out_c, k_size=5, stride=2, use_relu=True)
        self.c2 = ConvBlockDWM(out_c, out_c, k_size=3, stride=1, use_relu=True)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        return x1 + x2

class UpscalingUnitDWM(nn.Module):
    """Upscaling unit using PixelShuffle (sub-pixel convolution)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.c1 = ConvBlockDWM(in_c // 4, out_c, k_size=3, stride=1, use_relu=True)
        self.c2 = ConvBlockDWM(out_c, out_c, k_size=3, stride=1, use_relu=True)

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        return x1 + x2

class BottleneckUnitDWM(nn.Module):
    """Bottleneck unit with residual connection."""
    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBlockDWM(c, c, k_size=3, stride=1, use_relu=True)
        self.c2 = ConvBlockDWM(c, c, k_size=3, stride=1, use_relu=True)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        return x1 + x2

class DeepWaterMap4Chan(nn.Module):
    """
    4-Channel PyTorch DeepWaterMap model for Planet Labs satellite data (RGB + NIR).
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.first_layer = ConvBlockDWM(in_channels, 4, k_size=1, stride=1, use_relu=False)
        self.down1 = DownscalingUnitDWM(4, 16)
        self.down2 = DownscalingUnitDWM(16, 64)
        self.down3 = DownscalingUnitDWM(64, 256)
        self.down4 = DownscalingUnitDWM(256, 1024)
        
        self.bottleneck = BottleneckUnitDWM(1024)
        
        self.up1 = UpscalingUnitDWM(1024, 256)
        self.up2 = UpscalingUnitDWM(256, 64)
        self.up3 = UpscalingUnitDWM(64, 16)
        self.up4 = UpscalingUnitDWM(16, 4)
        
        self.last_layer = ConvBlockDWM(4, 1, k_size=1, stride=1, use_relu=False)

    def forward(self, x):
        skips = []
        x0 = self.first_layer(x)
        skips.append(x0)
        
        x1 = self.down1(x0)
        skips.append(x1)
        
        x2 = self.down2(x1)
        skips.append(x2)
        
        x3 = self.down3(x2)
        skips.append(x3)
        
        x4 = self.down4(x3)
        skips.append(x4)
        
        b = self.bottleneck(x4)
        
        d1 = b + skips.pop()
        u1 = self.up1(d1)
        
        d2 = u1 + skips.pop()
        u2 = self.up2(d2)
        
        d3 = u2 + skips.pop()
        u3 = self.up3(d3)
        
        d4 = u3 + skips.pop()
        u4 = self.up4(d4)
        
        d5 = u4 + skips.pop()
        out = self.last_layer(d5)
        return out

class SegmentationDataset4Chan(Dataset):
    """
    Dataset class for 4-channel (RGB + NIR) Planet Labs satellite imagery and binary NDWI water masks.
    Supports comprehensive spatial (flips, rotations, random scale/crop) and photometric (brightness jitter, noise) augmentations.
    """
    def __init__(self, data_dir, image_size=(256, 256), is_train=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        self.image_mask_pairs = self._find_pairs()

    def _find_pairs(self):
        pairs = []
        image_files = glob.glob(os.path.join(self.data_dir, "*.tif"))
        image_files = [f for f in image_files if "_concatenated_ndwi_mask_" not in os.path.basename(f)]
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            
            # Pattern 1: standard clip replace
            mask_name = img_name.replace("_clip_", "_concatenated_ndwi_mask_clip_")
            mask_path = os.path.join(self.data_dir, mask_name)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                continue

            # Pattern 2: clip replace without trailing clip
            mask_name_alt1 = img_name.replace("_clip_", "_concatenated_ndwi_mask_")
            mask_path_alt1 = os.path.join(self.data_dir, mask_name_alt1)
            if os.path.exists(mask_path_alt1):
                pairs.append((img_path, mask_path_alt1))
                continue
                
            # Pattern 3: regex base match
            base_match = re.match(r'(.+)_\d+-of-\d+(_[^_]+)?\.tif$', img_name)
            if base_match:
                base_name = base_match.group(1)
                suffix = base_match.group(2) if base_match.group(2) else ""
                parts = img_name.replace('.tif', '').split('_')
                tile_str = [p for p in parts if "-of-" in p]
                if tile_str:
                    mask_name_alt2 = f"{base_name}_concatenated_ndwi_mask_{tile_str[0]}{suffix}.tif"
                    mask_path_alt2 = os.path.join(self.data_dir, mask_name_alt2)
                    if os.path.exists(mask_path_alt2):
                        pairs.append((img_path, mask_path_alt2))
                        continue
        print(f"Found {len(pairs)} 4-channel image-mask pairs in {self.data_dir}")
        return pairs

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        with rio.open(img_path) as src:
            # Read all 4 bands (Blue, Green, Red, NIR)
            image_data = src.read()  # (4, H, W)
            if image_data.shape[0] < 4:
                pad_band = image_data[2:3]
                image_data = np.concatenate([image_data, pad_band], axis=0)
            elif image_data.shape[0] > 4:
                image_data = image_data[:4]

        image_data = image_data.astype(np.float32)
        image_data = np.nan_to_num(image_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        # DeepWaterMap style per-tile min-max normalization
        img_min = np.min(image_data)
        image_data = image_data - img_min
        img_max = np.max(image_data)
        if img_max > 0:
            image_data = image_data / img_max
            
        with rio.open(mask_path) as src:
            mask_data = src.read(1)
            mask_data = (mask_data > 0).astype(np.float32)

        # Convert to torch tensors
        img_tensor = torch.from_numpy(image_data)
        mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)
        
        # Resize to target input size
        if img_tensor.shape[1:] != self.image_size:
            img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=self.image_size, mode='nearest').squeeze(0)

        # Apply comprehensive online data augmentations during training
        if self.is_train:
            # 1. Random Horizontal Flip (50% probability)
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[2])

            # 2. Random Vertical Flip (50% probability)
            if torch.rand(1).item() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[1])
                mask_tensor = torch.flip(mask_tensor, dims=[1])

            # 3. Random 90°/180°/270° Rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k=k, dims=[1, 2])
                mask_tensor = torch.rot90(mask_tensor, k=k, dims=[1, 2])

            # 4. Random Brightness / Contrast Multiplicative Jitter (50% probability)
            if torch.rand(1).item() > 0.5:
                scale_factor = torch.empty(1).uniform_(0.85, 1.15).item()
                img_tensor = torch.clamp(img_tensor * scale_factor, 0.0, 1.0)

            # 5. Random Additive Zero-Mean Gaussian Noise (50% probability)
            if torch.rand(1).item() > 0.5:
                noise_std = torch.empty(1).uniform_(0.005, 0.02).item()
                noise = torch.randn_like(img_tensor) * noise_std
                img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)

            # 6. Random Scale & Crop (50% probability)
            if torch.rand(1).item() > 0.5:
                crop_ratio = torch.empty(1).uniform_(0.85, 1.0).item()
                crop_h = int(self.image_size[0] * crop_ratio)
                crop_w = int(self.image_size[1] * crop_ratio)
                
                top = torch.randint(0, self.image_size[0] - crop_h + 1, (1,)).item()
                left = torch.randint(0, self.image_size[1] - crop_w + 1, (1,)).item()
                
                img_cropped = img_tensor[:, top:top+crop_h, left:left+crop_w]
                mask_cropped = mask_tensor[:, top:top+crop_h, left:left+crop_w]
                
                img_tensor = F.interpolate(img_cropped.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)
                mask_tensor = F.interpolate(mask_cropped.unsqueeze(0), size=self.image_size, mode='nearest').squeeze(0)

        return img_tensor, mask_tensor
