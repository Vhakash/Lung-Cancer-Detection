"""
Model Comparison Script
Compare your current CNN with Transfer Learning models
"""

import numpy as np
import tensorflow as tf
import keras
import json
import time
import os

def load_test_data():
    """Load test data for comparison"""
    data = np.load('preprocessed_data.npz')
    return data['X_test'], data['y_test']

def evaluate_model(model_path, X_test, y_test, model_name):
    """Evaluate a single model"""
    print(f"\n📊 Evaluating {model_name}...")
    
    try:
        # Load model
        model = keras.models.load_model(model_path)
        
        # Time inference
        start_time = time.time()
        predictions = model.predict(X_test[:100])  # Test on 100 samples
        inference_time = (time.time() - start_time) / 100
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'inference_time_ms': inference_time * 1000,
            'model_size_mb': round(os.path.getsize(model_path) / (1024*1024), 2)
        }
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None

def main():
    """Compare all available models"""
    print("🔍 MODEL COMPARISON ANALYSIS")
    print("=" * 50)
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Models to compare
    models = {
        'Current CNN (Balanced)': 'lung_cancer_detector_balanced.h5',
        'EfficientNetB0 Transfer': 'lung_cancer_detector_efficientnetb0_transfer.h5',
        'ResNet50 Transfer': 'lung_cancer_detector_resnet50_transfer.h5',
        'DenseNet121 Transfer': 'lung_cancer_detector_densenet121_transfer.h5'
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            result = evaluate_model(model_path, X_test, y_test, model_name)
            if result:
                results[model_name] = result
    
    # Display comparison
    print("\n📊 COMPARISON RESULTS")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Loss':<10} {'Speed(ms)':<12} {'Size(MB)':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)"
              f"{'':>3} {metrics['loss']:.3f}"
              f"{'':>6} {metrics['inference_time_ms']:.1f}"
              f"{'':>8} {metrics['model_size_mb']}")
    
    # Find best model
    if results:
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        fastest_model = min(results.items(), key=lambda x: x[1]['inference_time_ms'])
        
        print(f"\n🏆 Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']*100:.1f}%)")
        print(f"⚡ Fastest Model: {fastest_model[0]} ({fastest_model[1]['inference_time_ms']:.1f}ms)")

if __name__ == "__main__":
    import os
    main()
