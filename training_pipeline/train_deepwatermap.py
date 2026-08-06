"""
Train 4-Channel DeepWaterMap Model from Scratch on Planet Labs Satellite Tiles

Uses BCEDiceLoss, spatial data augmentations (flips & rotations), and 50 training epochs.

Usage:
    python train_deepwatermap.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add parent directory to sys.path to import load_config
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from load_config import load_config, get_augment_tiles_output_folder
from deepwatermap_model import DeepWaterMap4Chan, SegmentationDataset4Chan, BCEDiceLoss

def train_deepwatermap():
    config = load_config()
    data_dir = get_augment_tiles_output_folder(config)
    output_dir = os.path.join(repo_root, "output_models")
    os.makedirs(output_dir, exist_ok=True)
    
    model_save_path = os.path.join(output_dir, "deepwatermap_planetlabs_best.pth")
    checkpoint_path = os.path.join(output_dir, "deepwatermap_planetlabs_checkpoint.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 125
    batch_size = 16
    lr = 1e-4

    print(f"==================================================")
    print(f"TRAINING 4-CHANNEL DEEPWATERMAP MODEL ({epochs} EPOCHS)")
    print(f"Device: {device}")
    print(f"Data Directory: {data_dir}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"Loss Function: BCEDiceLoss (Combined BCE + Dice Loss)")
    print(f"Augmentations: Enabled (Flips & Rotations)")
    print(f"Model Save Path: {model_save_path}")
    print(f"==================================================")

    # Instantiate dataset
    full_dataset = SegmentationDataset4Chan(data_dir, image_size=(256, 256), is_train=False)
    if len(full_dataset) == 0:
        print("Error: No 4-channel image-mask pairs found!")
        return

    # 80/20 train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    train_set = SegmentationDataset4Chan(data_dir, image_size=(256, 256), is_train=True)
    train_set.image_mask_pairs = [full_dataset.image_mask_pairs[i] for i in train_indices.indices]
    
    val_set = SegmentationDataset4Chan(data_dir, image_size=(256, 256), is_train=False)
    val_set.image_mask_pairs = [full_dataset.image_mask_pairs[i] for i in val_indices.indices]

    print(f"Dataset split: {len(train_set)} training samples, {len(val_set)} validation samples")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepWaterMap4Chan(in_channels=4).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Saved new best 4-channel DeepWaterMap model to {model_save_path}")

    # Save final checkpoint
    checkpoint_data = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"\nTraining completed! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    train_deepwatermap()
