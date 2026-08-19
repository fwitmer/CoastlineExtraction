import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import pickle
from datetime import datetime

# Add parent directory to path to import load_config and model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import load_config, get_training_config, get_model_save_path
from train_unet import UNet, AttentionUNet, ClassicUNet

def load_trained_model(model_path, device="cuda", model_type="auto"):
    """
    Loads a trained PyTorch U-Net or Attention U-Net model from a file checkpoint.

    Args:
        model_path (str): Absolute or relative path to the saved model state dictionary (.pth file).
        device (str, optional): Target device for model loading ('cuda', 'cpu', etc.). Defaults to "cuda".
        model_type (str, optional): Architecture type ('auto', 'attention', 'classic', 'standard'). Defaults to "auto".

    Returns:
        torch.nn.Module: Loaded PyTorch neural network model set to evaluation mode.
    """
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    if model_type == "auto":
        is_attn = any("attn" in key for key in state_dict.keys())
        if is_attn:
            model = AttentionUNet(n_channels=3, n_classes=1)
        else:
            model = UNet(n_channels=3, n_classes=1)
    elif model_type == "attention":
        model = AttentionUNet(n_channels=3, n_classes=1)
    elif model_type == "classic":
        model = ClassicUNet(n_channels=3, n_classes=1)
    else:
        model = UNet(n_channels=3, n_classes=1)

    model.load_state_dict(state_dict)
    model.to(device)
    return model

def predict_image(model, image_path, device="cuda", threshold=0.5):
    """
    Generates a binary segmentation mask prediction for a single input image.

    Args:
        model (torch.nn.Module): Trained U-Net model instance.
        image_path (str): Path to the input RGB image file.
        device (str, optional): Computation device ('cuda' or 'cpu'). Defaults to "cuda".
        threshold (float, optional): Probability threshold for binarizing model output. Defaults to 0.5.

    Returns:
        numpy.ndarray: Binary mask array (0 or 1) resized to original image dimensions with shape (H, W).
    """
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if output.shape[1] == 1:
            output = torch.sigmoid(output)
        pred_mask = (output.squeeze().cpu().numpy() > threshold).astype("uint8")
        
        pred_mask_resized = Image.fromarray(pred_mask * 255).resize(original_size, Image.NEAREST)
        pred_mask_resized = np.array(pred_mask_resized) / 255

    return pred_mask_resized.astype("uint8")

def predict_batch(model, image_paths, device="cuda", threshold=0.5, checkpoint_path=None, resume=True):
    """
    Generates binary segmentation mask predictions for a batch of input images with checkpointing support.

    Args:
        model (torch.nn.Module): Trained U-Net model instance.
        image_paths (list of str): List of paths to input RGB image files.
        device (str, optional): Computation device ('cuda' or 'cpu'). Defaults to "cuda".
        threshold (float, optional): Binarization probability threshold. Defaults to 0.5.
        checkpoint_path (str, optional): File path to save/load batch progress state (.pkl file). Defaults to None.
        resume (bool, optional): Whether to resume processing from existing checkpoint file. Defaults to True.

    Returns:
        list of numpy.ndarray: List of binary mask arrays corresponding to input images.
    """
    predictions = []
    processed_images = []
    
    if resume and checkpoint_path:
        processed_images, predictions, metadata = load_checkpoint(checkpoint_path)
        if processed_images is None:
            processed_images = []
            predictions = []
        else:
            print(f"Resuming from checkpoint: {len(processed_images)} images already processed")
    
    for i, image_path in enumerate(image_paths):
        if image_path in processed_images:
            print(f"Skipping already processed: {os.path.basename(image_path)}")
            continue
            
        print(f"Processing ({i+1}/{len(image_paths)}): {os.path.basename(image_path)}")
        try:
            pred_mask = predict_image(model, image_path, device, threshold)
            predictions.append(pred_mask)
            processed_images.append(image_path)
            
            if checkpoint_path and (i + 1) % 10 == 0:
                metadata = {
                    'device': device,
                    'threshold': threshold,
                    'total_images': len(image_paths),
                    'model_path': 'loaded_from_config'
                }
                save_checkpoint(checkpoint_path, processed_images, predictions, metadata)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    if checkpoint_path:
        metadata = {
            'device': device,
            'threshold': threshold,
            'total_images': len(image_paths),
            'model_path': 'loaded_from_config',
            'completed': True
        }
        save_checkpoint(checkpoint_path, processed_images, predictions, metadata)
        print(f"Final checkpoint saved: {len(processed_images)} images processed")
    
    return predictions

def visualize_prediction(image_path, pred_mask, save_path=None):
    """
    Displays and optionally saves a 3-panel figure showing original image, predicted binary mask, and color overlay.

    Args:
        image_path (str): Path to the input original image.
        pred_mask (numpy.ndarray): Predicted binary mask array.
        save_path (str, optional): Destination file path to save visualization figure. Defaults to None.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    original_image = Image.open(image_path).convert("RGB")
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    overlay = np.array(original_image)
    overlay[pred_mask == 1] = [255, 0, 0]  # Red overlay for predicted coastline
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def save_checkpoint(checkpoint_path, processed_images, predictions, metadata):
    """
    Saves batch prediction state and metadata to a pickle checkpoint file.

    Args:
        checkpoint_path (str): Destination pickle file path.
        processed_images (list of str): List of image file paths already processed.
        predictions (list of numpy.ndarray): List of predicted masks generated so far.
        metadata (dict): Additional execution context (device, threshold, total images, etc.).
    """
    checkpoint_data = {
        'processed_images': processed_images,
        'predictions': predictions,
        'metadata': metadata,
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(processed_images)
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {len(processed_images)} images processed")

def load_checkpoint(checkpoint_path):
    """
    Loads batch prediction state and metadata from a pickle checkpoint file if it exists.

    Args:
        checkpoint_path (str): Path to pickle checkpoint file.

    Returns:
        tuple: (processed_images, predictions, metadata) if found, otherwise (None, None, None).
    """
    if not os.path.exists(checkpoint_path):
        return None, None, None
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print(f"Checkpoint loaded: {checkpoint_data['total_processed']} images already processed")
        return (checkpoint_data['processed_images'], 
                checkpoint_data['predictions'], 
                checkpoint_data['metadata'])
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None

def get_checkpoint_path(output_dir, batch_name="prediction_batch"):
    """
    Constructs standardized path for batch prediction pickle checkpoint files.

    Args:
        output_dir (str): Directory where checkpoint file will be stored.
        batch_name (str, optional): Prefix identifier for the batch run. Defaults to "prediction_batch".

    Returns:
        str: Absolute or relative file path for checkpoint pickle.
    """
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{batch_name}_checkpoint.pkl")

def main():
    """
    CLI entry point for running U-Net coastline prediction on single or multiple images.
    """
    parser = argparse.ArgumentParser(description="Predict coastline masks using trained U-Net model")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--images", type=str, nargs="+", help="Paths to multiple images for prediction")
    parser.add_argument("--model", type=str, help="Path to trained model (optional, uses config if not provided)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask (default: 0.5)")
    parser.add_argument("--save", type=str, help="Path to save visualization")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save/load checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from checkpoint")
    parser.add_argument("--batch-name", type=str, default="prediction_batch", help="Name for checkpoint batch")
    
    args = parser.parse_args()
    
    config = load_config()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.model:
        model_path = args.model
    else:
        model_path = get_model_save_path(config)
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    model = load_trained_model(model_path, device)
    print("Model loaded successfully")
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
        
        print(f"Predicting mask for: {args.image}")
        pred_mask = predict_image(model, args.image, device, args.threshold)
        visualize_prediction(args.image, pred_mask, args.save)
        
        if args.save and not args.save.endswith(('.png', '.jpg', '.jpeg')):
            mask_save_path = args.save.replace('.png', '_mask.png')
            Image.fromarray(pred_mask * 255).save(mask_save_path)
            print(f"Mask saved to: {mask_save_path}")
    
    elif args.images:
        valid_images = [img for img in args.images if os.path.exists(img)]
        if not valid_images:
            print("No valid image files found")
            return
        
        checkpoint_path = None
        resume = not args.no_resume
        if args.checkpoint_dir:
            checkpoint_path = get_checkpoint_path(args.checkpoint_dir, args.batch_name)
            print(f"Checkpoint path: {checkpoint_path}")
        
        print(f"Processing {len(valid_images)} images...")
        predictions = predict_batch(model, valid_images, device, args.threshold, 
                                  checkpoint_path, resume)
        
        for i, (image_path, pred_mask) in enumerate(zip(valid_images, predictions)):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_save_path = f"{base_name}_predicted_mask.png"
            Image.fromarray(pred_mask * 255).save(mask_save_path)
            print(f"Saved: {mask_save_path}")
        
        print(f"Batch processing completed: {len(predictions)} predictions saved")
    
    else:
        print("Please provide either --image or --images argument")
        parser.print_help()

if __name__ == "__main__":
    main()
