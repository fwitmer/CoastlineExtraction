import os
import sys
import glob
import re
import numpy as np
import geopandas as gpd
import rasterio as rio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import skimage.measure
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, box
import matplotlib.pyplot as plt

# Dynamic linker fix: import rasterio and numpy before torch
# Already done at the top.

# Set path for loading configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from load_config import load_config, get_augment_tiles_output_folder, get_training_config

UTM_ZONE_3N = 'EPSG:32603'

# ----------------------------
# Dataset Class
# ----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_mask_pairs = self._find_image_mask_pairs()

    def _find_image_mask_pairs(self):
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
        return pairs

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
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
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DoubleConv(nn.Module):
    """
    Deeper convolution block with 3 convolutional layers, GroupNorm, and residual connection.
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

# ----------------------------
# Attention U-Net
# ----------------------------
class AttentionGate(nn.Module):
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
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size()[2:] != x1.size()[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        out = self.relu(g1 + x1)
        out = self.psi(out)
        return x * out

class AttentionUNet(nn.Module):
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
# Loss Functions
# ----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1. - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_val = self.bce(logits, targets)
        dice_val = self.dice(logits, targets)
        total_val = self.bce_weight * bce_val + self.dice_weight * dice_val
        return total_val, bce_val, dice_val

# ----------------------------
# Training Loop Helper
# ----------------------------
def run_training(model, train_loader, val_loader, epochs, lr, device, save_path):
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_bce, train_dice = 0.0, 0.0, 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", file=sys.stdout):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss, b_loss, d_loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bce += b_loss.item()
            train_dice += d_loss.item()
            
        model.eval()
        val_loss, val_bce, val_dice = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss, b_loss, d_loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_bce += b_loss.item()
                val_dice += d_loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_train_bce = train_bce / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        avg_val = val_loss / len(val_loader)
        avg_val_bce = val_bce / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss = {avg_train:.4f} (BCE: {avg_train_bce:.4f}, Dice: {avg_train_dice:.4f})")
        print(f"  Val Loss   = {avg_val:.4f} (BCE: {avg_val_bce:.4f}, Dice: {avg_val_dice:.4f}) | LR = {current_lr:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model with Val Loss: {avg_val:.4f}")
        scheduler.step()
        sys.stdout.flush()
            
    return best_val_loss

def evaluate_metrics(model, dataloader, device):
    model.eval()
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            tp += ((preds == 1) & (masks == 1)).sum().item()
            fp += ((preds == 1) & (masks == 0)).sum().item()
            fn += ((preds == 0) & (masks == 1)).sum().item()
            tn += ((preds == 0) & (masks == 0)).sum().item()
            
    total = tp + fp + fn + tn
    pixel_acc = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return pixel_acc, precision, recall, f1_score, iou

# ----------------------------
# Georeferenced Contours & RMSE
# ----------------------------
def extract_model_coastline(model, image_path, transform, device):
    model.eval()
    with rio.open(image_path) as src:
        image_data = src.read([3, 2, 1])
        # Scale to 0-255
        image_data = (np.clip(image_data.astype(np.float32) / 10000.0, 0.0, 1.0) * 255.0).astype(np.uint8)
        image_data = np.transpose(image_data, (1, 2, 0))
        h_orig, w_orig = image_data.shape[0], image_data.shape[1]
        
    image = Image.fromarray(image_data)
    transform_resize = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform_resize(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
        pred_mask = (output > 0.5).astype("uint8")
        
    # Resize back to original size
    pred_mask_resized = Image.fromarray(pred_mask * 255).resize((w_orig, h_orig), Image.NEAREST)
    pred_mask_np = np.array(pred_mask_resized) > 0
    
    contours = skimage.measure.find_contours(pred_mask_np.astype(np.float32), 0.5)
    
    lines = []
    for contour in contours:
        col = contour[:, 1]
        row = contour[:, 0]
        xs, ys = rio.transform.xy(transform, row, col)
        if len(xs) >= 2:
            lines.append(LineString(list(zip(xs, ys))))
    return lines

def calculate_rmse_on_transects(predicted_lines, trans_deering, ref_distances, usgs_distances):
    if not predicted_lines:
        return float('nan'), float('nan'), {}
        
    combined_lines = MultiLineString(predicted_lines)
    errors_planet = []
    errors_usgs = []
    regional_errors = {r: [] for r in [1, 2, 3, 4, 5]}
    
    for idx, row in trans_deering.iterrows():
        oid = int(row['TransOrder'])
        t_geom = row.geometry
        
        # Determine region of transect
        region_id = 5
        if oid >= 17443:
            region_id = 1
        elif oid >= 17394:
            region_id = 2
        elif oid >= 17370:
            region_id = 3
        elif oid >= 17337:
            region_id = 4
            
        pt_int = combined_lines.intersection(t_geom)
        dist = None
        if not pt_int.is_empty:
            if isinstance(pt_int, Point):
                dist = t_geom.project(pt_int)
            elif isinstance(pt_int, MultiPoint):
                dist = min([t_geom.project(pt) for pt in pt_int.geoms])
            elif hasattr(pt_int, 'geoms'):
                pts = [pt for pt in pt_int.geoms if isinstance(pt, Point)]
                if pts:
                    dist = min([t_geom.project(pt) for pt in pts])
                    
        if dist is not None:
            if oid in ref_distances:
                err_p = dist - ref_distances[oid]
                errors_planet.append(err_p)
                regional_errors[region_id].append(err_p)
            if oid in usgs_distances:
                err_u = dist - usgs_distances[oid]
                errors_usgs.append(err_u)
                
    rmse_planet = np.sqrt(np.mean(np.square(errors_planet))) if errors_planet else float('nan')
    rmse_usgs = np.sqrt(np.mean(np.square(errors_usgs))) if errors_usgs else float('nan')
    
    regional_rmse = {}
    for r, errs in regional_errors.items():
        regional_rmse[r] = np.sqrt(np.mean(np.square(errs))) if errs else float('nan')
        
    return rmse_planet, rmse_usgs, regional_rmse

# ----------------------------
# Main Execution Block
# ----------------------------
def main():
    print("==================================================")
    print("STARTING PIPELINE: TRAINING & EVALUATION (8 EPOCHS)")
    print("==================================================")
    
    config = load_config()
    data_dir = get_augment_tiles_output_folder(config)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Datasets & Loaders
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = SegmentationDataset(data_dir, transform=transform)
    print(f"Total dataset size: {len(dataset)} samples")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    
    training_config = get_training_config(config)
    batch_size = training_config.get('batch_size', 4)
    print(f"Loaded batch size: {batch_size}")
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    output_dir = os.path.join(repo_root, "output_models")
    os.makedirs(output_dir, exist_ok=True)
    unet_save_path = os.path.join(output_dir, "best_deep_unet_8epochs.pth")
    attn_save_path = os.path.join(output_dir, "best_deep_attn_unet_8epochs.pth")
    
    # 2. Train Standard U-Net
    print("\n----------------------------------")
    print("TRAINING STANDARD U-NET (8 EPOCHS)")
    print("----------------------------------")
    unet_model = UNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists(unet_save_path):
        print(f"Found existing Standard U-Net weights at {unet_save_path}. Skipping training.")
        unet_model.load_state_dict(torch.load(unet_save_path, map_location=device))
    else:
        unet_best_val = run_training(unet_model, train_loader, val_loader, epochs=20, lr=1e-3, device=device, save_path=unet_save_path)
    
    # 3. Train Attention U-Net
    print("\n----------------------------------")
    print("TRAINING ATTENTION U-NET (8 EPOCHS)")
    print("----------------------------------")
    attn_model = AttentionUNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists(attn_save_path):
        print(f"Found existing Attention U-Net weights at {attn_save_path}. Skipping training.")
        attn_model.load_state_dict(torch.load(attn_save_path, map_location=device))
    else:
        attn_best_val = run_training(attn_model, train_loader, val_loader, epochs=20, lr=1e-3, device=device, save_path=attn_save_path)
    
    # 4. Load Best Weights for Evaluation
    unet_model.load_state_dict(torch.load(unet_save_path, map_location=device))
    attn_model.load_state_dict(torch.load(attn_save_path, map_location=device))
    
    # Evaluate Validation Metrics
    print("\n----------------------------------")
    print("EVALUATING SEGMENTATION METRICS ON VAL SET")
    print("----------------------------------")
    u_acc, u_prec, u_rec, u_f1, u_iou = evaluate_metrics(unet_model, val_loader, device)
    a_acc, a_prec, a_rec, a_f1, a_iou = evaluate_metrics(attn_model, val_loader, device)
    
    print("\n--- Standard U-Net Validation Metrics ---")
    print(f"Pixel Accuracy:       {u_acc:.4%}")
    print(f"Precision (PPV):      {u_prec:.4%}")
    print(f"Recall (Sensitivity): {u_rec:.4%}")
    print(f"F1-Score (Dice Coeff): {u_f1:.4%}")
    print(f"Mean IoU (Jaccard):   {u_iou:.4%}")
    
    print("\n--- Attention U-Net Validation Metrics ---")
    print(f"Pixel Accuracy:       {a_acc:.4%}")
    print(f"Precision (PPV):      {a_prec:.4%}")
    print(f"Recall (Sensitivity): {a_rec:.4%}")
    print(f"F1-Score (Dice Coeff): {a_f1:.4%}")
    print(f"Mean IoU (Jaccard):   {a_iou:.4%}")
    sys.stdout.flush()
    
    # 5. Georeferenced Testing on 2016 Files & RMSE Calculation
    print("\n----------------------------------")
    print("EVALUATING HISTORICAL 2016 TEST TILES")
    print("----------------------------------")
    p_transects = os.path.join(repo_root, "USGS_Coastlines", "WestChukchi_exposed_STepr_rates", "WestChukchi_exposed_STepr_rates.shp")
    p_hires = os.path.join(repo_root, "ground_truth", "2016_HiRes_Final_Coastline.shp")
    
    # Candidate paths for digitized Planet Labs reference coastline
    planet_candidates = [
        os.path.join(repo_root, "existing_data", "DigitizedCoastlines", "PlanetCoastline_gt", "09_09_2016", "9_9_16_PlanetCoastline.shp"),
        os.path.join(repo_root, "..", "existing_data", "DigitizedCoastlines", "PlanetCoastline_gt", "09_09_2016", "9_9_16_PlanetCoastline.shp")
    ]
    p_planet = next((p for p in planet_candidates if os.path.exists(p)), planet_candidates[0])
    
    trans_deering = gpd.read_file(p_transects).to_crs(UTM_ZONE_3N)
    trans_deering = trans_deering[trans_deering['BaselineID'] == 117].sort_values('TransOrder')
    hires = gpd.read_file(p_hires).to_crs(UTM_ZONE_3N)
    planet = gpd.read_file(p_planet).to_crs(UTM_ZONE_3N)
    
    hires_union = hires.union_all() if hasattr(hires, 'union_all') else hires.unary_union
    planet_union = planet.union_all() if hasattr(planet, 'union_all') else planet.unary_union
    
    # Precompute reference intersection distances
    ref_distances = {}
    usgs_distances = {}
    for idx, row in trans_deering.iterrows():
        oid = int(row['TransOrder'])
        t_geom = row.geometry
        p_int = t_geom.intersection(planet_union)
        if not p_int.is_empty:
            if isinstance(p_int, Point):
                ref_distances[oid] = t_geom.project(p_int)
            elif isinstance(p_int, MultiPoint):
                ref_distances[oid] = min([t_geom.project(pt) for pt in p_int.geoms])
            elif hasattr(p_int, 'geoms'):
                pts = [pt for pt in p_int.geoms if isinstance(pt, Point)]
                if pts:
                    ref_distances[oid] = min([t_geom.project(pt) for pt in pts])
                    
        u_int = t_geom.intersection(hires_union)
        if not u_int.is_empty:
            if isinstance(u_int, Point):
                usgs_distances[oid] = t_geom.project(u_int)
            elif isinstance(u_int, MultiPoint):
                usgs_distances[oid] = min([t_geom.project(pt) for pt in u_int.geoms])
            elif hasattr(u_int, 'geoms'):
                pts = [pt for pt in u_int.geoms if isinstance(pt, Point)]
                if pts:
                    usgs_distances[oid] = min([t_geom.project(pt) for pt in pts])
                    
    # Test tile paths from test_data_4_6_sept
    test_tiles = [
        os.path.join(repo_root, "test_data_4_6_sept", "sept_4", "files", "369619_2016-09-04_RE2_3A_Analytic_SR_clip.tif"),
        os.path.join(repo_root, "test_data_4_6_sept", "sept_6", "files", "369619_2016-09-06_RE5_3A_Analytic_SR_clip.tif")
    ]
    
    for t_path in test_tiles:
        print(f"\nAnalyzing test tile: {os.path.basename(t_path)}")
        with rio.open(t_path) as src:
            transform = src.transform
            
        u_lines = extract_model_coastline(unet_model, t_path, transform, device)
        a_lines = extract_model_coastline(attn_model, t_path, transform, device)
        
        # Calculate RMSE scores
        u_rmse_p, u_rmse_u, u_regional = calculate_rmse_on_transects(u_lines, trans_deering, ref_distances, usgs_distances)
        a_rmse_p, a_rmse_u, a_regional = calculate_rmse_on_transects(a_lines, trans_deering, ref_distances, usgs_distances)
        
        print("\n  >> U-NET RMSE results:")
        print(f"     vs Planet Labs Reference: {u_rmse_p:.2f} m")
        print(f"     vs USGS Ground Truth:    {u_rmse_u:.2f} m")
        print("     Regional RMSE values:")
        print(f"       Western Region (R1):   {u_regional[1]:.2f} m")
        print(f"       Northern Region (R2):  {u_regional[2]:.2f} m")
        print(f"       Central Region (R3):   {u_regional[3]:.2f} m")
        print(f"       Town Region (R4):      {u_regional[4]:.2f} m")
        print(f"       East Region (R5):      {u_regional[5]:.2f} m")
        
        print("\n  >> ATTENTION U-NET RMSE results:")
        print(f"     vs Planet Labs Reference: {a_rmse_p:.2f} m")
        print(f"     vs USGS Ground Truth:    {a_rmse_u:.2f} m")
        print("     Regional RMSE values:")
        print(f"       Western Region (R1):   {a_regional[1]:.2f} m")
        print(f"       Northern Region (R2):  {a_regional[2]:.2f} m")
        print(f"       Central Region (R3):   {a_regional[3]:.2f} m")
        print(f"       Town Region (R4):      {a_regional[4]:.2f} m")
        print(f"       East Region (R5):      {a_regional[5]:.2f} m")
        sys.stdout.flush()
        
        # Plot predicted vs actual coastlines for the first test tile
        if "2016-10-15" in t_path:
            plot_predictions_comparison(t_path, u_lines, a_lines, planet_union, hires_union, transform)

def plot_geometry(ax, geom, color, linewidth, label, linestyle="-"):
    if geom.is_empty:
        return
    if isinstance(geom, LineString):
        ax.plot(*geom.xy, color=color, linewidth=linewidth, label=label, linestyle=linestyle)
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            ax.plot(*line.xy, color=color, linewidth=linewidth, label=label, linestyle=linestyle)
    elif hasattr(geom, "geoms"):
        for sub_geom in geom.geoms:
            plot_geometry(ax, sub_geom, color, linewidth, label, linestyle)

def plot_predictions_comparison(t_path, u_lines, a_lines, planet_union, hires_union, transform):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Predicted vs. Actual Coastlines (Tile: 2016-10-15)", fontsize=14, fontweight="bold")
    
    # Read the RGB image to show as background
    with rio.open(t_path) as src:
        image_data = src.read([3, 2, 1])
        image_data = (np.clip(image_data.astype(np.float32) / 10000.0, 0.0, 1.0) * 255.0).astype(np.uint8)
        image_data = np.transpose(image_data, (1, 2, 0))
        tile_bounds = src.bounds
        extent = [tile_bounds.left, tile_bounds.right, tile_bounds.bottom, tile_bounds.top]
        
    ax.imshow(image_data, extent=extent, alpha=0.8)
    
    # Plot Planet reference and USGS Ground truth
    tile_box = box(*tile_bounds)
    
    p_cropped = planet_union.intersection(tile_box)
    u_cropped = hires_union.intersection(tile_box)
    
    # Plot Ground Truth using robust helper
    plot_geometry(ax, p_cropped, color="orange", linewidth=2.5, label="Planet Labs Reference")
    plot_geometry(ax, u_cropped, color="red", linewidth=2.5, label="USGS Ground Truth")
                    
    # Plot standard U-Net predictions
    for idx, line in enumerate(u_lines):
        lbl = "Standard U-Net Pred" if idx == 0 else ""
        ax.plot(*line.xy, color="cyan", linewidth=1.5, linestyle="--", label=lbl)
        
    # Plot Attention U-Net predictions
    for idx, line in enumerate(a_lines):
        lbl = "Attention U-Net Pred" if idx == 0 else ""
        ax.plot(*line.xy, color="lime", linewidth=2.0, label=lbl)
        
    # Simplify legend duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l: # skip empty labels
            by_label[l] = h
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    output_dir = os.path.join(repo_root, "output_models")
    os.makedirs(output_dir, exist_ok=True)
    plot_out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(t_path))[0]}_predicted_vs_actual.png")
    plt.tight_layout()
    plt.savefig(plot_out_path, dpi=300)
    plt.close()
    print(f"Saved visual predicted vs actual comparison plot to: {plot_out_path}")

if __name__ == "__main__":
    main()
