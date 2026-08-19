"""
U-Net & Attention U-Net Training & Model Definitions for Coastline Segmentation

This module provides dataset utilities, model definitions, checkpoint management, 
and training routines for coastline extraction.

Supported Model Architectures:
- ClassicUNet: Standard 4-stage U-Net with BatchNorm and 2-conv blocks.
- UNet / DeepUNet: Deeper 5-stage U-Net with GroupNorm, 3-conv residual blocks.
- AttentionUNet: 5-stage U-Net with Attention Gates for spatial feature filtering.

Usage:
    python train_unet.py
"""

import rasterio as rio
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
import re
import json
import pickle
from datetime import datetime

# Add parent directory to path to import load_config
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from load_config import load_config, get_augment_tiles_output_folder, get_training_config, get_model_save_path

# ----------------------------
# Dataset Class
# ----------------------------
class SegmentationDataset(Dataset):
    """
    Dataset for coastline segmentation training.
    Automatically pairs GeoTIFF satellite image tiles with corresponding binary NDWI water masks.

    Args:
        data_dir (str): Path to directory containing tiled images and masks.
        transform (callable, optional): PyTorch transform to apply to images and masks.
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_mask_pairs = self._find_image_mask_pairs()

    def _find_image_mask_pairs(self):
        """
        Scans data directory for matching image and mask files.

        Returns:
            list of tuple: List of (image_path, mask_path) file pairs.
        """
        pairs = []
        image_files = glob.glob(os.path.join(self.data_dir, "*.tif"))
        image_files = [f for f in image_files if "_concatenated_ndwi_mask_" not in os.path.basename(f)]
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            mask_name = img_name.replace("_clip_", "_concatenated_ndwi_mask_clip_")
            mask_path = os.path.join(self.data_dir, mask_name)
            
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
                continue
                
            base_match = re.match(r'(.+)_\d+-of-\d+(_[^_]+)?\.tif$', img_name)
            if base_match:
                base_name = base_match.group(1)
                mask_name_alt = f"{base_name}_concatenated_ndwi_mask_{img_name.split('_')[-2]}_{img_name.split('_')[-1]}"
                mask_path_alt = os.path.join(self.data_dir, mask_name_alt)
                if os.path.exists(mask_path_alt):
                    pairs.append((img_path, mask_path_alt))
                else:
                    print(f"Warning: Mask not found for {img_name}")
        
        print(f"Found {len(pairs)} image-mask pairs")
        return pairs

    def __len__(self):
        """Returns total number of paired samples in dataset."""
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        """
        Loads and preprocesses image and mask at given index.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (transformed_image_tensor, binary_mask_tensor)
        """
        img_path, mask_path = self.image_mask_pairs[idx]
        with rio.open(img_path) as src:
            image_data = src.read([3, 2, 1])
            image_data = (np.clip(image_data.astype(np.float32) / 10000.0, 0.0, 1.0) * 255.0).astype(np.uint8)
            image_data = np.transpose(image_data, (1, 2, 0))
            image = Image.fromarray(image_data)
            
        with rio.open(mask_path) as src:
            mask_data = src.read(1)
            mask_data = (mask_data > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask_data, mode="L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask

# ----------------------------
# Model Components
# ----------------------------
def _init_weights(m):
    """
    Initializes module parameters using Kaiming Normal initialization 
    for Conv layers and constant initialization for Normalization layers.

    Args:
        m (nn.Module): PyTorch module layer.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ClassicDoubleConv(nn.Module):
    """
    Classic double convolution block consisting of two 3x3 Conv2d layers,
    each followed by BatchNorm2d and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class ClassicUNet(nn.Module):
    """
    Classic 4-stage U-Net architecture for semantic segmentation.

    Args:
        n_channels (int): Number of input channels (default: 3 for RGB).
        n_classes (int): Number of output classes (default: 1 for binary water segmentation).
    """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = ClassicDoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ClassicDoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ClassicDoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = ClassicDoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = ClassicDoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = ClassicDoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ClassicDoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ClassicDoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = ClassicDoubleConv(128, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for Classic U-Net.

        Returns:
            torch.Tensor: Raw logits tensor of shape (batch_size, n_classes, H, W).
        """
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        mid = self.middle(self.pool4(d4))

        u4 = self.conv4(torch.cat([self.up4(mid), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))

        return self.final(u1)  # Return raw logits

class DoubleConv(nn.Module):
    """
    Deeper residual convolution block featuring 3 convolutional layers, 
    GroupNorm normalization, and residual shortcut connections.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = min(32, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class UNet(nn.Module):
    """
    Deep 5-stage residual U-Net architecture with GroupNorm and residual blocks.

    Args:
        n_channels (int): Input image channels (default: 3).
        n_classes (int): Number of output segmentation classes (default: 1).
    """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.down5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

        self.middle = DoubleConv(1024, 2048)

        self.up5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(2048, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        """
        Forward pass for Deep U-Net.

        Returns:
            torch.Tensor: Raw logits tensor of shape (batch_size, n_classes, H, W).
        """
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        d5 = self.down5(self.pool4(d4))
        mid = self.middle(self.pool5(d5))

        u5_out = self.up5(mid)
        if u5_out.size()[2:] != d5.size()[2:]:
            u5_out = nn.functional.interpolate(u5_out, size=d5.size()[2:], mode='bilinear', align_corners=True)
        u5 = self.conv5(torch.cat([u5_out, d5], dim=1))

        u4_out = self.up4(u5)
        if u4_out.size()[2:] != d4.size()[2:]:
            u4_out = nn.functional.interpolate(u4_out, size=d4.size()[2:], mode='bilinear', align_corners=True)
        u4 = self.conv4(torch.cat([u4_out, d4], dim=1))

        u3_out = self.up3(u4)
        if u3_out.size()[2:] != d3.size()[2:]:
            u3_out = nn.functional.interpolate(u3_out, size=d3.size()[2:], mode='bilinear', align_corners=True)
        u3 = self.conv3(torch.cat([u3_out, d3], dim=1))

        u2_out = self.up2(u3)
        if u2_out.size()[2:] != d2.size()[2:]:
            u2_out = nn.functional.interpolate(u2_out, size=d2.size()[2:], mode='bilinear', align_corners=True)
        u2 = self.conv2(torch.cat([u2_out, d2], dim=1))

        u1_out = self.up1(u2)
        if u1_out.size()[2:] != d1.size()[2:]:
            u1_out = nn.functional.interpolate(u1_out, size=d1.size()[2:], mode='bilinear', align_corners=True)
        u1 = self.conv1(torch.cat([u1_out, d1], dim=1))

        return self.final(u1)  # Return raw logits for BCEWithLogitsLoss

# Alias DeepUNet
DeepUNet = UNet

class AttentionGate(nn.Module):
    """
    Attention Gate module to filter skip connection feature maps based on 
    gating signals from deeper network layers.

    Args:
        F_g (int): Number of feature maps in gating signal tensor.
        F_l (int): Number of feature maps in skip connection tensor.
        F_int (int): Intermediate channel reduction dimension.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        g_groups = min(32, F_int)
        x_groups = min(32, F_int)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(g_groups, F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(x_groups, F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Forward pass for Attention Gate.

        Args:
            g (torch.Tensor): Gating signal tensor from deeper layer.
            x (torch.Tensor): Skip connection feature map tensor.

        Returns:
            torch.Tensor: Attention-weighted feature map.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        out = self.relu(g1 + x1)
        out = self.psi(out)
        return x * out

class AttentionUNet(nn.Module):
    """
    Deep 5-stage residual Attention U-Net architecture.
    Uses Attention Gates on skip connections for enhanced spatial feature selection.

    Args:
        n_channels (int): Input image channels (default: 3).
        n_classes (int): Output segmentation classes (default: 1).
    """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.down5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

        self.middle = DoubleConv(1024, 2048)

        self.up5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.attn5 = AttentionGate(F_g=1024, F_l=1024, F_int=512)
        self.conv5 = DoubleConv(2048, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x):
        """
        Forward pass for Attention U-Net.

        Returns:
            torch.Tensor: Raw logits tensor of shape (batch_size, n_classes, H, W).
        """
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        d5 = self.down5(self.pool4(d4))
        mid = self.middle(self.pool5(d5))

        up5_out = self.up5(mid)
        if up5_out.size()[2:] != d5.size()[2:]:
            up5_out = nn.functional.interpolate(up5_out, size=d5.size()[2:], mode='bilinear', align_corners=True)
        attn5_out = self.attn5(g=up5_out, x=d5)
        u5 = self.conv5(torch.cat([up5_out, attn5_out], dim=1))

        up4_out = self.up4(u5)
        if up4_out.size()[2:] != d4.size()[2:]:
            up4_out = nn.functional.interpolate(up4_out, size=d4.size()[2:], mode='bilinear', align_corners=True)
        attn4_out = self.attn4(g=up4_out, x=d4)
        u4 = self.conv4(torch.cat([up4_out, attn4_out], dim=1))
        
        up3_out = self.up3(u4)
        if up3_out.size()[2:] != d3.size()[2:]:
            up3_out = nn.functional.interpolate(up3_out, size=d3.size()[2:], mode='bilinear', align_corners=True)
        attn3_out = self.attn3(g=up3_out, x=d3)
        u3 = self.conv3(torch.cat([up3_out, attn3_out], dim=1))
        
        up2_out = self.up2(u3)
        if up2_out.size()[2:] != d2.size()[2:]:
            up2_out = nn.functional.interpolate(up2_out, size=d2.size()[2:], mode='bilinear', align_corners=True)
        attn2_out = self.attn2(g=up2_out, x=d2)
        u2 = self.conv2(torch.cat([up2_out, attn2_out], dim=1))
        
        up1_out = self.up1(u2)
        if up1_out.size()[2:] != d1.size()[2:]:
            up1_out = nn.functional.interpolate(up1_out, size=d1.size()[2:], mode='bilinear', align_corners=True)
        attn1_out = self.attn1(g=up1_out, x=d1)
        u1 = self.conv1(torch.cat([up1_out, attn1_out], dim=1))

        return self.final(u1)  # Return raw logits for BCEWithLogitsLoss

# ----------------------------
# Checkpoint Functions
# ----------------------------
def save_training_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                           best_val_loss, config, checkpoint_path):
    """
    Saves full training checkpoint dictionary to file.

    Args:
        model (nn.Module): Current model state.
        optimizer (torch.optim.Optimizer): Current optimizer state.
        epoch (int): Current training epoch index.
        train_loss (float): Average training loss for current epoch.
        val_loss (float): Average validation loss for current epoch.
        best_val_loss (float): Historical best validation loss.
        config (dict): Active configuration settings.
        checkpoint_path (str): File path to write checkpoint file.
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

def load_training_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads saved checkpoint weights and state into model and optimizer.

    Args:
        checkpoint_path (str): Path to checkpoint file.
        model (nn.Module): Target model instance.
        optimizer (torch.optim.Optimizer, optional): Target optimizer instance.

    Returns:
        tuple or None: (epoch, train_loss, val_loss, best_val_loss, config) if loaded, else None.
    """
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return (checkpoint['epoch'], 
                checkpoint['train_loss'], 
                checkpoint['val_loss'], 
                checkpoint['best_val_loss'], 
                checkpoint['config'])
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def get_checkpoint_path(model_save_path, epoch=None):
    """
    Constructs standardized checkpoint file path based on model path and epoch.

    Args:
        model_save_path (str): Base model output save path.
        epoch (int, optional): Epoch index.

    Returns:
        str: Absolute or relative checkpoint filepath.
    """
    base_dir = os.path.dirname(model_save_path)
    base_name = os.path.splitext(os.path.basename(model_save_path))[0]
    if epoch is not None:
        return os.path.join(base_dir, f"{base_name}_checkpoint_epoch_{epoch}.pth")
    else:
        return os.path.join(base_dir, f"{base_name}_checkpoint_latest.pth")

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, train_loader, val_loader, config, model_save_path, resume_from_checkpoint=True):
    """
    Main training loop for U-Net & Attention U-Net models.

    Handles forward pass, BCEWithLogits loss computation, backpropagation,
    validation evaluation, checkpointing, and saving the best performing model.

    Args:
        model (nn.Module): U-Net model instance.
        train_loader (DataLoader): PyTorch training DataLoader.
        val_loader (DataLoader): PyTorch validation DataLoader.
        config (dict): Configuration parameters dictionary.
        model_save_path (str): Filepath to save the best model weights.
        resume_from_checkpoint (bool): Whether to resume training from existing checkpoint.
    """
    training_config = get_training_config(config)
    epochs = training_config.get('epochs', 30)
    lr = training_config.get('learning_rate', 1e-4)
    device = training_config.get('device', 'auto')
    save_every_n_epochs = training_config.get('save_every_n_epochs', 5)
    
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    print(f"Training on device: {device}")
    
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    checkpoint_path = get_checkpoint_path(model_save_path)
    if resume_from_checkpoint:
        checkpoint_data = load_training_checkpoint(checkpoint_path, model, optimizer)
        if checkpoint_data is not None:
            start_epoch, _, _, best_val_loss, _ = checkpoint_data
            start_epoch += 1
            print(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if (epoch + 1) % save_every_n_epochs == 0:
            save_training_checkpoint(model, optimizer, epoch, avg_train_loss, 
                                   avg_val_loss, best_val_loss, config, checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved! Val Loss: {avg_val_loss:.4f}")

    save_training_checkpoint(model, optimizer, epochs-1, avg_train_loss, 
                           avg_val_loss, best_val_loss, config, checkpoint_path)
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    config = load_config()
    training_config = get_training_config(config)
    
    data_dir = get_augment_tiles_output_folder(config)
    model_save_path = get_model_save_path(config)
    
    image_size = training_config.get('image_size', [256, 256])
    batch_size = training_config.get('batch_size', 8)
    train_split = training_config.get('train_split', 0.8)
    model_type = training_config.get('model_type', 'attention_unet')
    
    print(f"Data directory: {data_dir}")
    print(f"Model save path: {model_save_path}")
    print(f"Model architecture type: {model_type}")
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(data_dir, transform=transform)
    
    if len(dataset) == 0:
        print("No image-mask pairs found! Please check your data directory and file naming.")
        sys.exit(1)

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training, {val_size} validation samples")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True)

    if model_type == 'attention_unet':
        model = AttentionUNet(n_channels=3, n_classes=1)
    elif model_type == 'classic_unet':
        model = ClassicUNet(n_channels=3, n_classes=1)
    else:
        model = UNet(n_channels=3, n_classes=1)
        
    train_model(model, train_loader, val_loader, config, model_save_path)
