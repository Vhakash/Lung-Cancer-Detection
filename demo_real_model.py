"""
Demo script to showcase the real CNN model vs the old mock model.
"""
import numpy as np
import time
from model import LungCancerCNN
from logger import logger

def demo_real_vs_mock():
    """Demonstrate the difference between real and mock models."""
    print("🚀 Lung Cancer Detection - Real CNN Model Demo")
    print("=" * 50)
    
    # Create test image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    print(f"📸 Test image shape: {test_image.shape}")
    
    # Test Basic CNN
    print("\n🔬 Testing Basic CNN Model:")
    print("-" * 30)
    
    try:
        basic_model = LungCancerCNN(model_type="basic")
        
        start_time = time.time()
        prediction = basic_model.predict(np.expand_dims(test_image, axis=0))
        end_time = time.time()
        
        print(f"✅ Model: {basic_model.name}")
        print(f"📊 Prediction: {prediction[0][0]:.4f}")
        print(f"⏱️ Processing time: {(end_time - start_time)*1000:.2f} ms")
        print(f"🏗️ Architecture: Real CNN with {basic_model.model.count_params():,} parameters")
        
        # Show model summary
        print("\n📋 Model Architecture Summary:")
        print(basic_model.get_model_summary()[:500] + "...")
        
    except Exception as e:
        print(f"❌ Basic CNN failed: {e}")
    
    # Test Transfer Learning Model
    print("\n🚀 Testing Transfer Learning Model:")
    print("-" * 40)
    
    try:
        transfer_model = LungCancerCNN(model_type="transfer")
        
        start_time = time.time()
        prediction = transfer_model.predict(np.expand_dims(test_image, axis=0))
        end_time = time.time()
        
        print(f"✅ Model: {transfer_model.name}")
        print(f"📊 Prediction: {prediction[0][0]:.4f}")
        print(f"⏱️ Processing time: {(end_time - start_time)*1000:.2f} ms")
        print(f"🏗️ Architecture: EfficientNetB0 + Custom Head with {transfer_model.model.count_params():,} parameters")
        
    except Exception as e:
        print(f"❌ Transfer learning model failed: {e}")
    
    # Demo training capability
    print("\n🎓 Training Demo (Synthetic Data):")
    print("-" * 35)
    
    try:
        demo_model = LungCancerCNN(model_type="basic")
        print("📚 Training on synthetic data...")
        
        start_time = time.time()
        success = demo_model.train_on_synthetic_data(epochs=3)
        end_time = time.time()
        
        if success:
            print(f"✅ Training completed in {(end_time - start_time):.2f} seconds")
            print(f"🎯 Model is now trained: {demo_model.is_trained}")
            
            # Test prediction after training
            trained_prediction = demo_model.predict(np.expand_dims(test_image, axis=0))
            print(f"📊 Post-training prediction: {trained_prediction[0][0]:.4f}")
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
    
    print("\n🎉 Demo completed!")
    print("\n💡 Key Improvements:")
    print("   • Real CNN architectures (not mock)")
    print("   • Actual TensorFlow/Keras models")
    print("   • Training capabilities on real data")
    print("   • Transfer learning with EfficientNet")
    print("   • Model saving and loading")
    print("   • Proper Grad-CAM visualization")
    print("   • Professional model architecture")

if __name__ == "__main__":
    demo_real_vs_mock()