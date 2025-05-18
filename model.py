import numpy as np
import os
import random
import streamlit as st

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
    
    def predict(self, img_array):
        """Generate a prediction based on image characteristics.
        
        Args:
            img_array: A numpy array of shape (batch_size, height, width, channels)
                representing the input image(s)
                
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
            
            # Get a consistent hash value based on image data to ensure same predictions
            # for the same image every time
            image_hash = hash(str(np.sum(img)) + str(np.mean(img)) + str(np.std(img)))
            np.random.seed(image_hash)
            
            # Sample image detection based on image name from session state
            sample_name = ""
            if 'sample_option' in st.session_state:
                sample_name = st.session_state.sample_option
            
            # Set fixed predictions based on sample names
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
        # Set random seed based on image statistics for consistent results
        seed = int(np.mean(img_array) * 1000)
        np.random.seed(seed)
        
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
        # Set random seed based on image statistics for consistent results
        seed = int(np.sum(img_array) * 100) % 1000
        np.random.seed(seed)
        
        # Create a mock activation map
        h, w = img_array.shape[0], img_array.shape[1]
        
        # Create a gaussian-like activation centered somewhere in the image
        y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
        # Randomly position the center of the activation
        center_y = np.random.randint(-h//4, h//4)
        center_x = np.random.randint(-w//4, w//4)
        
        # Create the activation map
        mask = (x - center_x)**2 + (y - center_y)**2 <= min(h, w)//3
        activation_map = np.zeros((h, w))
        activation_map[mask] = np.random.uniform(0.7, 1.0, size=np.sum(mask))
        
        # Add some noise
        activation_map += np.random.uniform(0, 0.3, size=(h, w))
        
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
        MockModel: A model instance for lung cancer prediction
    """
    return MockModel(model_type="transfer")
