"""
Transfer Learning Implementation for Lung Cancer Detection
Using EfficientNetB0 pre-trained model for improved accuracy

This script demonstrates:
1. Loading pre-trained models
2. Feature extraction vs fine-tuning
3. Custom head for 3-class classification
4. Advanced training techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16  # Increased from 8 for transfer learning
EPOCHS = 50
LEARNING_RATE = 0.001
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LR = 0.0001

class TransferLearningTrainer:
    def __init__(self, model_name='EfficientNetB0'):
        """
        Initialize transfer learning trainer
        
        Args:
            model_name: 'EfficientNetB0', 'ResNet50', 'DenseNet121'
        """
        self.model_name = model_name
        self.img_size = IMG_SIZE
        self.num_classes = 3
        self.class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']
        
        print(f"🚀 Initializing Transfer Learning with {model_name}")
        
    def get_base_model(self):
        """Get pre-trained base model"""
        input_shape = (self.img_size, self.img_size, 3)
        
        if self.model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(
                weights='imagenet',  # Pre-trained on ImageNet
                include_top=False,   # Remove classification head
                input_shape=input_shape
            )
        elif self.model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif self.model_name == 'DenseNet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        return base_model
    
    def create_model(self, trainable_base=False):
        """
        Create transfer learning model
        
        Args:
            trainable_base: If True, allows fine-tuning of base model
        """
        # Get pre-trained base model
        base_model = self.get_base_model()
        
        # Freeze base model initially (Feature Extraction)
        base_model.trainable = trainable_base
        
        if trainable_base:
            print(f"🔧 Fine-tuning mode: Base model trainable")
            # Fine-tune only top layers
            fine_tune_at = len(base_model.layers) // 2
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
        else:
            print(f"🔒 Feature extraction mode: Base model frozen")
        
        # Create custom classification head
        model = keras.Sequential([
            # Base model (feature extractor)
            base_model,
            
            # Global Average Pooling (better than Flatten)
            layers.GlobalAveragePooling2D(),
            
            # Regularization
            layers.Dropout(0.5),
            
            # Dense layers for classification
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        print(f"📊 Model created with {model.count_params():,} parameters")
        print(f"📊 Trainable parameters: {sum([tf.size(var) for var in model.trainable_variables]):,}")
        
        return model
    
    def compile_model(self, model, learning_rate=LEARNING_RATE):
        """Compile model with optimizer and metrics"""
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Model compiled with learning rate: {learning_rate}")
        return model
    
    def get_callbacks(self, model_name_suffix=""):
        """Get training callbacks"""
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                f'best_model_{self.model_name.lower()}{model_name_suffix}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def plot_training_history(self, history, title_suffix=""):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} Training History {title_suffix}', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{self.model_name.lower()}{title_suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("📊 MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Accuracy
        test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"📉 Test Loss: {test_loss:.4f}")
        print(f"🎯 Test Precision: {test_precision:.4f}")
        print(f"🎯 Test Recall: {test_recall:.4f}")
        
        # F1 Score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print(f"🎯 F1 Score: {f1_score:.4f}")
        
        # Classification Report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'confusion_matrix_{self.model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score
        }

def load_data():
    """Load preprocessed data"""
    print("📁 Loading preprocessed data...")
    
    if not os.path.exists('preprocessed_data.npz'):
        print("❌ preprocessed_data.npz not found!")
        print("Please run preprocess_data.py first")
        return None, None, None, None
    
    data = np.load('preprocessed_data.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"✅ Data loaded successfully!")
    print(f"📊 Training samples: {X_train.shape[0]}")
    print(f"📊 Test samples: {X_test.shape[0]}")
    print(f"📊 Image shape: {X_train.shape[1:]}")
    print(f"📊 Number of classes: {y_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main training pipeline"""
    print("🫁 LUNG CANCER DETECTION - TRANSFER LEARNING")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return
    
    # Choose model (you can experiment with different models)
    models_to_try = ['EfficientNetB0']  # Add 'ResNet50', 'DenseNet121' to compare
    
    results = {}
    
    for model_name in models_to_try:
        print(f"\n🚀 Training {model_name}")
        print("-" * 40)
        
        # Initialize trainer
        trainer = TransferLearningTrainer(model_name)
        
        # PHASE 1: Feature Extraction (frozen base model)
        print("\n📍 PHASE 1: Feature Extraction Training")
        model = trainer.create_model(trainable_base=False)
        model = trainer.compile_model(model, LEARNING_RATE)
        
        # Train with frozen base
        history1 = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=trainer.get_callbacks("_feature_extraction"),
            verbose=1
        )
        
        # Plot feature extraction training
        trainer.plot_training_history(history1, "- Feature Extraction")
        
        # PHASE 2: Fine-tuning (unfreeze base model)
        print("\n📍 PHASE 2: Fine-tuning Training")
        
        # Create fine-tuning model
        model_ft = trainer.create_model(trainable_base=True)
        
        # Load best weights from feature extraction
        model_ft.set_weights(model.get_weights())
        
        # Compile with lower learning rate for fine-tuning
        model_ft = trainer.compile_model(model_ft, FINE_TUNE_LR)
        
        # Fine-tune
        history2 = model_ft.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=FINE_TUNE_EPOCHS,
            validation_data=(X_test, y_test),
            callbacks=trainer.get_callbacks("_fine_tuned"),
            verbose=1
        )
        
        # Plot fine-tuning training
        trainer.plot_training_history(history2, "- Fine-tuning")
        
        # Evaluate final model
        metrics = trainer.evaluate_model(model_ft, X_test, y_test)
        results[model_name] = metrics
        
        # Save final model
        final_model_name = f'lung_cancer_detector_{model_name.lower()}_transfer.h5'
        model_ft.save(final_model_name)
        print(f"💾 Model saved as: {final_model_name}")
        
        # Save model info
        model_info = {
            'model_name': model_name,
            'architecture': 'Transfer Learning',
            'input_shape': (IMG_SIZE, IMG_SIZE, 3),
            'num_classes': 3,
            'class_names': trainer.class_names,
            'metrics': metrics,
            'training_params': {
                'batch_size': BATCH_SIZE,
                'feature_extraction_epochs': EPOCHS,
                'fine_tune_epochs': FINE_TUNE_EPOCHS,
                'learning_rate': LEARNING_RATE,
                'fine_tune_lr': FINE_TUNE_LR
            }
        }
        
        with open(f'model_info_{model_name.lower()}.json', 'w') as f:
            json.dump(model_info, f, indent=2)
    
    # Compare results
    print("\n" + "="*60)
    print("🏆 FINAL COMPARISON RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n📊 {model_name}:")
        print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🥇 Best Model: {best_model[0]} with {best_model[1]['accuracy']*100:.2f}% accuracy")
    
    print("\n✅ Transfer learning training completed!")
    print("📁 Check the generated files:")
    print("   - Training history plots")
    print("   - Confusion matrices")
    print("   - Saved models (.h5 files)")
    print("   - Model information (.json files)")

if __name__ == "__main__":
    main()