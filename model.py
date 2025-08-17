import numpy as np
import os
import random
import streamlit as st
from typing import Optional

def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False

def load_keras_model_from_path(path: str):
    """Load a Keras model (.h5/.keras/SavedModel) if TensorFlow is available.

    Falls back to MockModel with a warning when TF is unavailable or load fails.
    """
    if not _tf_available():
        st.warning("TensorFlow not installed. Using MockModel instead.")
        return MockModel(model_type="basic")
    try:
        from tensorflow import keras
        model = keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load Keras model from '{path}': {e}")
        return MockModel(model_type="basic")

class MockModel:
    """A mock model that simulates CNN predictions for lung cancer detection.
    
    This class is used for development and testing purposes when an actual
    trained model is not available or needed. It generates realistic-looking 
    predictions based on image characteristics.
    """
    
    def __init__(self, model_type="basic"):
        """Initialize mock model with specified type.
        
        Args:
            model_type (str): Type of model to mock - 'basic' or 'transfer'
        """
        self.model_type = model_type
        self.name = "Basic CNN" if model_type == "basic" else "Transfer Learning"
    
    def predict(self, img_array, sample_name=None):
        """Generate a prediction based on image characteristics.
        
        Args:
            img_array: A numpy array of shape (batch_size, height, width, channels)
                representing the input image(s)
            sample_name (str, optional): The name of the sample image, if applicable.
                
        Returns:
            numpy.ndarray: An array of predictions with shape (batch_size, 1)
        """
        batch_size = img_array.shape[0]
        predictions = np.zeros((batch_size, 1))
        
        for i in range(batch_size):
            # Extract image features to influence the prediction
            img = img_array[i]
            
            # Use image complexity as a factor (more complex images more likely to be classified as cancer)
            complexity = np.std(img)
            
            # Use average intensity as a factor
            avg_intensity = np.mean(img)
            
            # Use a simple deterministic approach based on image statistics
            # Convert image statistics to a valid seed value (0 to 2^32-1)
            img_sum = np.sum(img)
            img_mean = np.mean(img)
            img_std = np.std(img)
            seed_value = int((img_sum * img_mean * 1000) % (2**32 - 1))
            np.random.seed(seed_value)
            
            # Set fixed predictions based on sample names
            if sample_name:
                if sample_name == "Normal Lung Scan":
                    # Healthy with high confidence
                    prediction_value = 0.15
                elif sample_name == "Nodule Present":
                    # Suspicious but not cancer
                    prediction_value = 0.35
                elif sample_name == "Early Cancer Signs":
                    # Early cancer with medium confidence
                    prediction_value = 0.65
                elif sample_name == "Advanced Cancer":
                    # Advanced cancer with high confidence
                    prediction_value = 0.90
                elif sample_name == "Pneumonia Case":
                    # Not cancer but abnormal
                    prediction_value = 0.25
                else:
                    # For non-sample images or if pattern matching failed
                    
                    # Calculate a base prediction score (0-1)
                    # This formula is adjusted to generally produce a more balanced distribution
                    base_score = (complexity * 0.6 + (1 - avg_intensity) * 0.3) * 0.6
                    
                    # Add variability based on image statistics
                    variance_factor = np.var(img) * 3
                    
                    # Add randomness for more balanced predictions
                    random_factor = np.random.uniform(-0.3, 0.3)
                    
                    # Adjust the bias based on model type
                    if self.model_type == "transfer":
                        model_bias = 0.03 if base_score > 0.4 else -0.03
                    else:
                        model_bias = 0.0
                    
                    # Final prediction - we use a slightly different formula for more variety
                    prediction_value = base_score + variance_factor + random_factor + model_bias
                    
                    # Ensure more balanced predictions by applying curve
                    if np.random.random() > 0.7:  # 30% chance to flip from one class to another
                        if prediction_value > 0.5:
                            prediction_value = 0.3 + np.random.uniform(0, 0.15)
                        else:
                            prediction_value = 0.7 + np.random.uniform(0, 0.15)
            else:
                # For non-sample images
                
                # Calculate a base prediction score (0-1)
                # This formula is adjusted to generally produce a more balanced distribution
                base_score = (complexity * 0.6 + (1 - avg_intensity) * 0.3) * 0.6
                
                # Add variability based on image statistics
                variance_factor = np.var(img) * 3
                
                # Add randomness for more balanced predictions
                random_factor = np.random.uniform(-0.3, 0.3)
                
                # Adjust the bias based on model type
                if self.model_type == "transfer":
                    model_bias = 0.03 if base_score > 0.4 else -0.03
                else:
                    model_bias = 0.0
                
                # Final prediction - we use a slightly different formula for more variety
                prediction_value = base_score + variance_factor + random_factor + model_bias
                
                # Ensure more balanced predictions by applying curve
                if np.random.random() > 0.7:  # 30% chance to flip from one class to another
                    if prediction_value > 0.5:
                        prediction_value = 0.3 + np.random.uniform(0, 0.15)
                    else:
                        prediction_value = 0.7 + np.random.uniform(0, 0.15)
            
            # Final prediction (ensure it's between 0 and 1)
            final_prediction = max(0, min(1, prediction_value))
            
            predictions[i, 0] = final_prediction
            
        return predictions
    
    def get_feature_maps(self, img_array, layer_index=1):
        """Extract feature maps from a specific layer for visualization.
        
        Args:
            img_array: A numpy array representing the input image
            layer_index: Index of the layer to extract features from
            
        Returns:
            numpy.ndarray: Feature maps from the specified layer
        """
        # Set a safe random seed based on image statistics for consistent results
        # Make sure the seed is within valid range (0 to 2^32-1)
        img_mean = np.mean(img_array)
        safe_seed = int(abs(img_mean * 1000) % (2**32 - 1))
        np.random.seed(safe_seed)
        
        # Create a mock feature map
        if layer_index == 1:
            # First layer typically has fewer feature maps
            feature_maps = np.random.rand(56, 56, 8)
        else:
            # Deeper layers typically have more feature maps
            feature_maps = np.random.rand(28, 28, 16)
        
        # Reset random seed
        np.random.seed(None)
        
        return feature_maps
    
    def get_activation_map(self, img_array):
        """Generate a mock class activation map.
        
        Args:
            img_array: A numpy array representing the input image
            
        Returns:
            numpy.ndarray: A mock activation map highlighting areas of interest
        """
        # Set a safe random seed based on image statistics for consistent results
        # Make sure the seed is within valid range (0 to 2^32-1)
        img_sum = abs(np.sum(img_array))
        safe_seed = int(img_sum * 100) % (2**32 - 1)
        np.random.seed(safe_seed)
        
        # Create a mock activation map
        h, w = img_array.shape[0], img_array.shape[1]
        
        # Create a gaussian-like activation centered somewhere in the image
        y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
        
        # Determine if this should be cancer-like or normal
        # Get sample name from session state if available
        is_cancer = False
        sample_name = ""
        if 'sample_option' in st.session_state:
            sample_name = st.session_state.sample_option
            if "Cancer" in sample_name:
                is_cancer = True
        
        # For cancer cases, create specific patterns that look like tumors
        if is_cancer:
            # Create multiple activation centers for cancer cases
            activation_map = np.zeros((h, w))
            
            # Primary spot
            center_y = np.random.randint(-h//4, h//4)
            center_x = np.random.randint(-w//4, w//4)
            radius = min(h, w)//4
            mask1 = (x - center_x)**2 + (y - center_y)**2 <= radius
            activation_map[mask1] = np.random.uniform(0.8, 1.0, size=np.sum(mask1))
            
            # Secondary spots for advanced cancer
            if "Advanced" in sample_name:
                for _ in range(2):
                    sec_y = np.random.randint(-h//3, h//3)
                    sec_x = np.random.randint(-w//3, w//3)
                    sec_radius = min(h, w)//8
                    sec_mask = (x - sec_x)**2 + (y - sec_y)**2 <= sec_radius
                    activation_map[sec_mask] = np.random.uniform(0.6, 0.9, size=np.sum(sec_mask))
        else:
            # For normal cases, minimal activation or diffuse pattern
            activation_map = np.zeros((h, w))
            center_y = np.random.randint(-h//4, h//4)
            center_x = np.random.randint(-w//4, w//4)
            mask = (x - center_x)**2 + (y - center_y)**2 <= min(h, w)//5
            activation_map[mask] = np.random.uniform(0.3, 0.6, size=np.sum(mask))
        
        # Add some noise
        activation_map += np.random.uniform(0, 0.2, size=(h, w))
        
        # Normalize
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
        
        # Reset random seed
        np.random.seed(None)
        
        return activation_map

def create_model():
    """Create and return a basic CNN model for lung cancer detection.
    
    Returns:
        MockModel: A model instance for lung cancer prediction
    """
    return MockModel(model_type="basic")

def load_pretrained_model():
    """Load a pre-trained model based on InceptionV3 for lung cancer detection.
    
    Returns:
        A tf.keras.Model binary classifier if TensorFlow is available, otherwise MockModel
    """
    if not _tf_available():
        st.warning("TensorFlow not installed. Using MockModel (transfer) instead.")
        return MockModel(model_type="transfer")
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.applications import InceptionV3

        base = InceptionV3(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        base.trainable = False
        inputs = keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs, name="inceptionv3_binary")
        return model
    except Exception as e:
        st.error(f"Failed to build InceptionV3 Keras model: {e}")
        return MockModel(model_type="transfer")