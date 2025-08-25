"""
Alternative lung cancer detection using scikit-learn for systems without TensorFlow.
This provides a working demo when TensorFlow installation fails due to Windows Long Path issues.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import cv2
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


class SklearnLungCancerDetector:
    """
    Scikit-learn based lung cancer detector for systems without TensorFlow.
    Uses traditional machine learning with engineered features.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the detector.
        
        Args:
            model_type: 'random_forest', 'gradient_boost', or 'svm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=50)
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'random_forest', 'gradient_boost', or 'svm'")
    
    def extract_features(self, image):
        """
        Extract engineered features from a medical image.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to standard size
        image = cv2.resize(image, (128, 128))
        
        features = []
        
        # 1. Basic statistical features
        features.extend([
            np.mean(image),
            np.std(image),
            np.min(image),
            np.max(image),
            np.median(image),
            np.percentile(image, 25),
            np.percentile(image, 75)
        ])
        
        # 2. Histogram features
        hist = cv2.calcHist([image], [0], None, [32], [0, 256])
        features.extend(hist.flatten() / hist.sum())  # Normalized histogram
        
        # 3. Texture features (using Haralick-inspired features)
        # Compute gradients
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(grad_magnitude),
            np.std(grad_magnitude),
            np.max(grad_magnitude)
        ])
        
        # 4. Shape/morphological features
        # Apply threshold and find contours
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Shape features
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
            
            features.extend([area / (128*128), perimeter / (4*128), circularity])
        else:
            features.extend([0, 0, 0])
        
        # 5. Local Binary Pattern-inspired features
        # Simple version: compare center pixel with neighbors
        lbp_features = []
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                pattern = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if image[i+di, j+dj] > center:
                            pattern += 1
                lbp_features.append(pattern)
        
        # Histogram of LBP patterns
        lbp_hist, _ = np.histogram(lbp_features, bins=8, range=(0, 8))
        features.extend(lbp_hist / len(lbp_features))
        
        # 6. Frequency domain features (DCT)
        dct = cv2.dct(np.float32(image))
        # Take top-left corner (low frequencies)
        dct_features = dct[:8, :8].flatten()
        features.extend(dct_features)
        
        return np.array(features)
    
    def create_synthetic_dataset(self, n_samples=1000):
        """
        Create synthetic medical imaging dataset for demonstration.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=cancer)
        """
        print(f"Creating {n_samples} synthetic medical images...")
        
        X = []
        y = []
        
        np.random.seed(42)
        
        for i in range(n_samples):
            # Generate synthetic image
            if i % 2 == 0:  # Normal case
                # Normal lung tissue - smoother, more uniform
                image = np.random.normal(120, 30, (128, 128))
                
                # Add some normal anatomical structures
                center_x, center_y = 64, 64
                for _ in range(3):
                    x = int(np.random.normal(center_x, 20))
                    y = int(np.random.normal(center_y, 20))
                    if 20 < x < 108 and 20 < y < 108:
                        cv2.circle(image, (x, y), np.random.randint(5, 15), 
                                 int(np.random.normal(100, 10)), -1)
                
                label = 0  # Normal
            else:  # Cancer case
                # Cancer tissue - more heterogeneous
                image = np.random.normal(110, 40, (128, 128))
                
                # Add suspicious lesions (brighter spots)
                for _ in range(np.random.randint(1, 4)):
                    x = np.random.randint(20, 108)
                    y = np.random.randint(20, 108)
                    size = np.random.randint(8, 25)
                    intensity = int(np.random.normal(180, 20))
                    cv2.circle(image, (x, y), size, intensity, -1)
                
                # Add irregular patterns
                for _ in range(np.random.randint(2, 6)):
                    x = np.random.randint(10, 118)
                    y = np.random.randint(10, 118)
                    cv2.ellipse(image, (x, y), 
                              (np.random.randint(5, 15), np.random.randint(3, 10)),
                              np.random.randint(0, 180), 0, 360, 
                              int(np.random.normal(160, 15)), -1)
                
                label = 1  # Cancer
            
            # Ensure valid pixel range
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Extract features
            features = self.extract_features(image)
            X.append(features)
            y.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} samples...")
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction for test set
            
        Returns:
            Training results dictionary
        """
        print(f"Training {self.model_type} lung cancer detector...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Feature preprocessing
        print("Preprocessing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train_selected)
        X_test_pca = self.pca.transform(X_test_selected)
        
        print(f"Feature dimensions after preprocessing: {X_train_pca.shape[1]}")
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_pca, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_pca, y_train)
        test_score = self.model.score(X_test_pca, y_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_pca)
        y_pred_proba = self.model.predict_proba(X_test_pca)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_pca, y_train, cv=5)
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\n📊 Training Results:")
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        print(f"AUC-ROC: {auc:.3f}")
        print(f"Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def predict(self, image):
        """
        Predict lung cancer probability for a single image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extract_features(image)
        
        # Preprocess
        features_scaled = self.scaler.transform([features])
        features_selected = self.feature_selector.transform(features_scaled)
        features_pca = self.pca.transform(features_selected)
        
        # Predict
        prediction = self.model.predict(features_pca)[0]
        probability = self.model.predict_proba(features_pca)[0]
        
        return {
            'prediction': int(prediction),
            'cancer_probability': float(probability[1]),
            'normal_probability': float(probability[0])
        }
    
    def save_model(self, filepath):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.pca = model_data['pca']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")
    
    def plot_results(self, results, save_path=None):
        """Plot training results and evaluation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion matrix
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC curve data points (simplified)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_pred_proba'])
        axes[0, 1].plot(fpr, tpr, 'b-', label=f'AUC = {results["auc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'r--')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Prediction probabilities
        axes[1, 0].hist(results['y_pred_proba'][results['y_true'] == 0], 
                       alpha=0.5, label='Normal', bins=20)
        axes[1, 0].hist(results['y_pred_proba'][results['y_true'] == 1], 
                       alpha=0.5, label='Cancer', bins=20)
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].set_xlabel('Cancer Probability')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        
        # Performance metrics
        accuracy = results['test_accuracy']
        auc = results['auc']
        cv_mean = results['cv_mean']
        
        metrics_text = f"""
        Model: {self.model_type.replace('_', ' ').title()}
        
        Test Accuracy: {accuracy:.3f}
        AUC-ROC: {auc:.3f}
        CV Score: {cv_mean:.3f}
        
        Features: Engineered
        Algorithm: Scikit-learn
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                        verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def demo_sklearn_detector():
    """Demonstrate the scikit-learn lung cancer detector."""
    print("🫁 Scikit-learn Lung Cancer Detection Demo")
    print("=" * 45)
    print("This demo works without TensorFlow installation.")
    print()
    
    # Try different models
    models = ['random_forest', 'gradient_boost', 'svm']
    
    for model_type in models:
        print(f"\n🧠 Testing {model_type.replace('_', ' ').title()} Model")
        print("-" * 40)
        
        try:
            # Initialize detector
            detector = SklearnLungCancerDetector(model_type=model_type)
            
            # Create synthetic dataset
            X, y = detector.create_synthetic_dataset(n_samples=800)
            
            # Train model
            results = detector.train(X, y)
            
            # Save model
            model_filename = f"sklearn_{model_type}_lung_detector.joblib"
            detector.save_model(model_filename)
            
            # Plot results
            plot_filename = f"sklearn_{model_type}_results.png"
            detector.plot_results(results, save_path=plot_filename)
            
            # Test single prediction
            print("\n🔍 Testing single prediction...")
            test_image = np.random.normal(120, 30, (128, 128)).astype(np.uint8)
            pred_result = detector.predict(test_image)
            
            print(f"Cancer probability: {pred_result['cancer_probability']:.3f}")
            print(f"Prediction: {'Cancer' if pred_result['prediction'] else 'Normal'}")
            
        except Exception as e:
            print(f"❌ Error with {model_type}: {e}")
            continue
    
    print("\n✅ Scikit-learn demo completed!")
    print("Check generated .png files for results visualization.")


if __name__ == '__main__':
    demo_sklearn_detector()
