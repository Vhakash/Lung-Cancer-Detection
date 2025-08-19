"""
Model training utility for lung cancer detection.
This script can be used to train models on real medical imaging data.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from model import LungCancerCNN, train_model_on_real_data
from config import TARGET_IMAGE_SIZE, DATABASE_DIR
from logger import logger

def create_sample_dataset_structure():
    """Create sample dataset directory structure for training."""
    data_dir = Path("sample_dataset")
    
    # Create directories
    (data_dir / "train" / "normal").mkdir(parents=True, exist_ok=True)
    (data_dir / "train" / "cancer").mkdir(parents=True, exist_ok=True)
    (data_dir / "validation" / "normal").mkdir(parents=True, exist_ok=True)
    (data_dir / "validation" / "cancer").mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = """
# Sample Dataset Structure

This directory should contain your medical imaging data organized as follows:

```
sample_dataset/
├── train/
│   ├── normal/     # Normal lung images
│   └── cancer/     # Cancer lung images
└── validation/
    ├── normal/     # Normal validation images
    └── cancer/     # Cancer validation images
```

## Data Requirements:
- Images should be in JPG, PNG, or DICOM format
- Recommended size: 224x224 pixels (will be resized automatically)
- Ensure proper medical ethics approval for any real patient data
- Consider data privacy and HIPAA compliance

## Usage:
1. Place your training images in the appropriate folders
2. Run: python train_model.py --data_path sample_dataset --model_type basic --epochs 20
"""
    
    with open(data_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Created sample dataset structure at {data_dir}")
    print("📝 Please read the README.md file for data organization instructions")

def train_basic_model(data_path: str, epochs: int = 20):
    """Train a basic CNN model."""
    logger.info("Training basic CNN model...")
    
    model = LungCancerCNN(model_type="basic", input_shape=TARGET_IMAGE_SIZE + (3,))
    
    if train_model_on_real_data(model, data_path, epochs):
        print("✅ Basic CNN model training completed successfully!")
        return True
    else:
        print("❌ Basic CNN model training failed!")
        return False

def train_transfer_model(data_path: str, epochs: int = 15):
    """Train a transfer learning model."""
    logger.info("Training transfer learning model...")
    
    model = LungCancerCNN(model_type="transfer", input_shape=TARGET_IMAGE_SIZE + (3,))
    
    if train_model_on_real_data(model, data_path, epochs):
        print("✅ Transfer learning model training completed successfully!")
        return True
    else:
        print("❌ Transfer learning model training failed!")
        return False

def train_synthetic_demo():
    """Train models on synthetic data for demonstration."""
    print("🔬 Training models on synthetic data for demonstration...")
    
    # Train basic model
    print("\n📊 Training Basic CNN...")
    basic_model = LungCancerCNN(model_type="basic")
    if basic_model.train_on_synthetic_data(epochs=10):
        print("✅ Basic CNN trained successfully!")
    else:
        print("❌ Basic CNN training failed!")
    
    # Train transfer learning model
    print("\n🚀 Training Transfer Learning Model...")
    transfer_model = LungCancerCNN(model_type="transfer")
    if transfer_model.train_on_synthetic_data(epochs=5):
        print("✅ Transfer learning model trained successfully!")
    else:
        print("❌ Transfer learning model training failed!")
    
    print("\n🎉 Demo training completed! Models are ready for use in the application.")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train lung cancer detection models")
    parser.add_argument("--data_path", type=str, help="Path to training data directory")
    parser.add_argument("--model_type", choices=["basic", "transfer", "both"], 
                       default="both", help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--create_structure", action="store_true", 
                       help="Create sample dataset directory structure")
    parser.add_argument("--demo", action="store_true", 
                       help="Train on synthetic data for demonstration")
    
    args = parser.parse_args()
    
    # Create dataset structure if requested
    if args.create_structure:
        create_sample_dataset_structure()
        return
    
    # Train on synthetic data for demo
    if args.demo:
        train_synthetic_demo()
        return
    
    # Check if data path is provided for real training
    if not args.data_path:
        print("❌ Please provide --data_path for training on real data, or use --demo for synthetic training")
        print("💡 Use --create_structure to create sample dataset directory structure")
        return
    
    if not os.path.exists(args.data_path):
        print(f"❌ Data path not found: {args.data_path}")
        print("💡 Use --create_structure to create sample dataset directory structure")
        return
    
    # Ensure models directory exists
    models_dir = DATABASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting training with data from: {args.data_path}")
    print(f"📊 Training for {args.epochs} epochs")
    print(f"🎯 Model type: {args.model_type}")
    
    success = True
    
    # Train models based on selection
    if args.model_type in ["basic", "both"]:
        success &= train_basic_model(args.data_path, args.epochs)
    
    if args.model_type in ["transfer", "both"]:
        success &= train_transfer_model(args.data_path, args.epochs)
    
    if success:
        print("\n🎉 All training completed successfully!")
        print("🚀 You can now run the application: streamlit run app.py")
    else:
        print("\n⚠️ Some training tasks failed. Check the logs for details.")

if __name__ == "__main__":
    main()