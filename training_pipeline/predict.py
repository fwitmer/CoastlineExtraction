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
from train_unet import UNet

def predict_image(model, image_path, device="cuda", threshold=0.5):
    """
    Predict mask for a single image.
    
    Args:
        model: Trained U-Net model
        image_path: Path to input image
        device: Device to run inference on
        threshold: Threshold for binary mask (default: 0.5)
    
    Returns:
        numpy array: Predicted binary mask
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)
    
    # Use same transform as training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output.squeeze().cpu().numpy() > threshold).astype("uint8")
        
        # Resize back to original size
        pred_mask_resized = Image.fromarray(pred_mask * 255).resize(original_size, Image.NEAREST)
        pred_mask_resized = np.array(pred_mask_resized) / 255

    return pred_mask_resized.astype("uint8")

def predict_batch(model, image_paths, device="cuda", threshold=0.5, checkpoint_path=None, resume=True):
    """
    Predict masks for multiple images.
    
    Args:
        model: Trained U-Net model
        image_paths: List of paths to input images
        device: Device to run inference on
        threshold: Threshold for binary mask (default: 0.5)
        checkpoint_path: Path to checkpoint file for resume capability
        resume: Whether to resume from checkpoint if available
    
    Returns:
        list: List of predicted binary masks
    """
    predictions = []
    processed_images = []
    
    # Try to load checkpoint if resume is enabled
    if resume and checkpoint_path:
        processed_images, predictions, metadata = load_checkpoint(checkpoint_path)
        if processed_images is None:
            processed_images = []
            predictions = []
        else:
            print(f"Resuming from checkpoint: {len(processed_images)} images already processed")
    
    # Process remaining images
    for i, image_path in enumerate(image_paths):
        if image_path in processed_images:
            print(f"Skipping already processed: {os.path.basename(image_path)}")
            continue
            
        print(f"Processing ({i+1}/{len(image_paths)}): {os.path.basename(image_path)}")
        try:
            pred_mask = predict_image(model, image_path, device, threshold)
            predictions.append(pred_mask)
            processed_images.append(image_path)
            
            # Save checkpoint every 10 images
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
    
    # Final checkpoint save
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
    Visualize the original image and predicted mask side by side.
    
    Args:
        image_path: Path to original image
        pred_mask: Predicted mask array
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    original_image = Image.open(image_path).convert("RGB")
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis('off')
    
    # Overlay
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

def load_trained_model(model_path, device="cuda"):
    """
    Load a trained U-Net model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
    
    Returns:
        Loaded U-Net model
    """
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def save_checkpoint(checkpoint_path, processed_images, predictions, metadata):
    """
    Save prediction checkpoint with progress information.
    
    Args:
        checkpoint_path (str): Path to save checkpoint file
        processed_images (list): List of processed image paths
        predictions (list): List of prediction results
        metadata (dict): Additional metadata about the prediction run
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
    Load prediction checkpoint to resume processing.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
    
    Returns:
        tuple: (processed_images, predictions, metadata) or (None, None, None) if not found
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
    Generate checkpoint file path.
    
    Args:
        output_dir (str): Directory to save checkpoint
        batch_name (str): Name for the batch
    
    Returns:
        str: Checkpoint file path
    """
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{batch_name}_checkpoint.pkl")

def main():
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
    
    # Load configuration
    config = load_config()
    training_config = get_training_config(config)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Get model path
    if args.model:
        model_path = args.model
    else:
        model_path = get_model_save_path(config)
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train the model first or provide a valid model path with --model")
        return
    
    model = load_trained_model(model_path, device)
    print("Model loaded successfully")
    
    # Process single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
        
        print(f"Predicting mask for: {args.image}")
        pred_mask = predict_image(model, args.image, device, args.threshold)
        
        # Visualize result
        visualize_prediction(args.image, pred_mask, args.save)
        
        # Save mask if save path provided
        if args.save and not args.save.endswith(('.png', '.jpg', '.jpeg')):
            mask_save_path = args.save.replace('.png', '_mask.png')
            Image.fromarray(pred_mask * 255).save(mask_save_path)
            print(f"Mask saved to: {mask_save_path}")
    
    # Process multiple images
    elif args.images:
        valid_images = [img for img in args.images if os.path.exists(img)]
        if not valid_images:
            print("No valid image files found")
            return
        
        # Set up checkpoint path if checkpoint directory is provided
        checkpoint_path = None
        resume = not args.no_resume
        if args.checkpoint_dir:
            checkpoint_path = get_checkpoint_path(args.checkpoint_dir, args.batch_name)
            print(f"Checkpoint path: {checkpoint_path}")
        
        print(f"Processing {len(valid_images)} images...")
        processed_images, predictions = predict_batch(model, valid_images, device, args.threshold, checkpoint_path, resume)
        
        # Save predictions
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
