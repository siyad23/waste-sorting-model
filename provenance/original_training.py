from ultralytics import YOLO
import torch
import os
import pandas as pd
import json
from datetime import datetime

def train_model():
    """Optimized YOLO training with interactive configuration."""
    
    print("\n" + "="*60)
    print("  YOLO11 Object Detection Training")
    print("="*60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🎮 GPU: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.1f} GB")
    else:
        print("\n⚠️  No GPU detected - training will use CPU (slower)")
    
    # Check system RAM
    try:
        import psutil
        ram_total = psutil.virtual_memory().total / 1024**3
        ram_available = psutil.virtual_memory().available / 1024**3
        print(f"💾 System RAM: {ram_available:.1f}/{ram_total:.1f} GB available")
        
        # Provide caching recommendation
        if ram_available < 20:
            print("💡 Recommendation: Use disk caching (option 2) due to limited RAM")
        else:
            print("💡 Recommendation: RAM caching (option 3) available")
    except ImportError:
        pass
    
    print("\n" + "="*60)
    
    # Get dataset path
    dataset_path = input("\nEnter dataset YAML path (default: data/automatedWasteSortingSystem-13/data.yaml): ").strip()
    if not dataset_path:
        dataset_path = "data/automatedWasteSortingSystem-13/data.yaml"
    
    # Model selection
    print("\nAvailable models:")
    print("  1. yolo11n.pt (Nano - Fast, lower accuracy)")
    print("  2. yolo11s.pt (Small - Balanced)")
    print("  3. yolo11m.pt (Medium - Better accuracy)")
    print("  4. yolo11l.pt (Large - High accuracy)")
    
    model_choice = input("\nSelect model (1-4, default: 1): ").strip()
    model_map = {
        '1': 'models/yolo11n.pt',
        '2': 'models/yolo11s.pt',
        '3': 'models/yolo11m.pt',
        '4': 'models/yolo11l.pt',
        '': 'models/yolo11n.pt'  # default
    }
    model_path = model_map.get(model_choice, 'models/yolo11n.pt')
    
    # Training parameters
    epochs = input("\nNumber of epochs (default: 100): ").strip()
    epochs = int(epochs) if epochs else 100
    
    batch_size = input("Batch size (default: -1 for auto): ").strip()
    batch_size = int(batch_size) if batch_size else -1
    
    img_size = input("Image size (default: 640): ").strip()
    img_size = int(img_size) if img_size else 640
    
    # Cache options
    print("\nCache options:")
    print("  1. No caching (slower, low memory)")
    print("  2. Cache on disk (balanced)")
    print("  3. Cache in RAM (fastest, high memory)")
    cache_choice = input("Select cache option (1-3, default: 2): ").strip()
    cache_map = {
        '1': False,
        '2': 'disk',
        '3': True,
        '': 'disk'  # default
    }
    cache_option = cache_map.get(cache_choice, 'disk')
    
    # Check if resuming training
    resume = input("\nResume from last checkpoint? (y/n, default: n): ").strip().lower()
    
    print("\n" + "="*60)
    print("🚀 Starting training with optimized settings...")
    print("="*60 + "\n")
    
    # Load model
    if resume == 'y' or resume == 'yes':
        # Find the most recent training run
        checkpoint_dir = input("Enter checkpoint directory (default: runs/detect/train): ").strip()
        if not checkpoint_dir:
            checkpoint_dir = "runs/detect/train"
        
        last_checkpoint = os.path.join(checkpoint_dir, "weights", "last.pt")
        
        if os.path.exists(last_checkpoint):
            print(f"📂 Resuming from: {last_checkpoint}")
            print(f"📝 Using dataset: {dataset_path}\n")
            model = YOLO(last_checkpoint)
            
            # Resume training with the NEW dataset path (not the one stored in checkpoint)
            results = model.train(
                data=dataset_path,  # Use the dataset path entered by user
                resume=True
            )
            
            # Export training data after resuming
            print("\n" + "="*60)
            print("💾 Exporting training data for analysis...")
            print("="*60 + "\n")
            export_training_data(results)
            print("="*60 + "\n")
            
            return results
        else:
            print(f"⚠️  Checkpoint not found at {last_checkpoint}")
            print(f"⚠️  Starting fresh training instead\n")
    
    model = YOLO(model_path)
    
    # Train with optimized parameters
    results = model.train(
        # Dataset
        data=dataset_path,
        
        # Training duration
        epochs=epochs,
        patience=50,              # Early stopping patience
        
        # Image and batch settings
        imgsz=img_size,
        batch=batch_size,         # Auto-detect optimal batch size
        
        # Hardware
        device=0,                 # Use first GPU (or CPU if no GPU)
        workers=8,                # Data loading workers
        
        # Performance optimizations
        cache=cache_option,       # Cache images (False, 'disk', or True for RAM)
        amp=True,                 # Automatic Mixed Precision
        
        # Optimizer settings
        optimizer='AdamW',        # AdamW optimizer
        lr0=0.01,                 # Initial learning rate
        lrf=0.01,                 # Final learning rate fraction
        momentum=0.937,           # SGD momentum/Adam beta1
        weight_decay=0.0005,      # Optimizer weight decay
        
        # Data augmentation
        hsv_h=0.015,              # HSV-Hue augmentation
        hsv_s=0.7,                # HSV-Saturation
        hsv_v=0.4,                # HSV-Value
        degrees=10.0,             # Rotation (+/- deg)
        translate=0.1,            # Translation (+/- fraction)
        scale=0.5,                # Scale (+/- gain)
        shear=0.0,                # Shear (+/- deg)
        perspective=0.0,          # Perspective (+/- fraction)
        flipud=0.5,               # Vertical flip probability
        fliplr=0.5,               # Horizontal flip probability
        mosaic=1.0,               # Mosaic augmentation probability
        mixup=0.1,                # Mixup augmentation probability
        copy_paste=0.0,           # Copy-paste augmentation probability
        
        # Augmentation control
        close_mosaic=10,          # Disable mosaic for final epochs
        
        # Validation
        val=True,                 # Validate during training
        
        # Logging and saving
        plots=True,               # Generate training plots
        save=True,                # Save checkpoints
        save_period=10,           # Save checkpoint every N epochs
        verbose=True,             # Verbose output
        
        # Project organization
        project='runs/detect',
        name='train',
        exist_ok=False,
    )
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print(f"📊 Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"📈 Results saved in: {results.save_dir}")
    
    # Export training data to CSV for analysis
    print("\n" + "="*60)
    print("💾 Exporting training data for analysis...")
    print("="*60 + "\n")
    
    export_training_data(results)
    
    print("="*60 + "\n")
    
    return results


def export_training_data(results):
    """
    Export comprehensive training data to CSV and JSON files.
    
    Args:
        results: Training results object from YOLO
    """
    save_dir = results.save_dir
    
    # 1. Export metrics from results.csv if it exists
    csv_path = os.path.join(save_dir, 'results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Save processed CSV
        processed_csv = os.path.join(save_dir, 'training_metrics.csv')
        df.to_csv(processed_csv, index=False)
        print(f"✅ Training metrics saved: {processed_csv}")
        
        # Create summary statistics
        summary = {
            'total_epochs': len(df),
            'best_epoch': int(df['metrics/mAP50(B)'].idxmax()) + 1 if 'metrics/mAP50(B)' in df.columns else None,
            'best_mAP50': float(df['metrics/mAP50(B)'].max()) if 'metrics/mAP50(B)' in df.columns else None,
            'best_mAP50-95': float(df['metrics/mAP50-95(B)'].max()) if 'metrics/mAP50-95(B)' in df.columns else None,
            'final_train_loss': float(df['train/box_loss'].iloc[-1]) if 'train/box_loss' in df.columns else None,
            'final_val_loss': float(df['val/box_loss'].iloc[-1]) if 'val/box_loss' in df.columns else None,
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary as JSON
        summary_json = os.path.join(save_dir, 'training_summary.json')
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"✅ Training summary saved: {summary_json}")
        
    # 2. Export per-class metrics if available
    try:
        # Check for class-wise results
        class_results_path = os.path.join(save_dir, 'results.txt')
        if os.path.exists(class_results_path):
            print(f"✅ Detailed results available: {class_results_path}")
    except Exception as e:
        print(f"⚠️  Could not export class metrics: {e}")
    
    # 3. Create metadata file
    metadata = {
        'save_directory': str(save_dir),
        'best_weights': os.path.join(save_dir, 'weights', 'best.pt'),
        'last_weights': os.path.join(save_dir, 'weights', 'last.pt'),
        'results_csv': csv_path if os.path.exists(csv_path) else None,
        'processed_csv': processed_csv if os.path.exists(csv_path) else None,
        'plots_directory': os.path.join(save_dir, 'plots') if os.path.exists(os.path.join(save_dir, 'plots')) else None,
    }
    
    metadata_json = os.path.join(save_dir, 'training_metadata.json')
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Training metadata saved: {metadata_json}")
    
    print(f"\n📂 All training data saved in: {save_dir}")
    print(f"📊 Use 'python analyze_training.py {save_dir}' to generate plots")


if __name__ == "__main__":
    train_model()