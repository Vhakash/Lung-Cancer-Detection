import numpy as np
from PIL import Image
import cv2

def preprocess_image(image_array, target_size=(224, 224)):
    """Preprocess image for model input.
    
    Args:
        image_array (numpy.ndarray): Input image as a numpy array
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    # Ensure image is in numpy array format
    if not isinstance(image_array, np.ndarray):
        image_array = np.array(image_array)
    
    # Ensure image has 3 channels (convert grayscale to RGB if needed)
    image_array = ensure_color_channels(image_array)
    
    # Resize image to target size
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    normalized_image = resized_image / 255.0
    
    return normalized_image

def ensure_color_channels(image_array):
    """Ensure the image has 3 color channels (convert grayscale to RGB if needed).
    
    Args:
        image_array (numpy.ndarray): Input image as a numpy array
        
    Returns:
        numpy.ndarray: Image with 3 color channels
    """
    # If image is grayscale (2D), convert to RGB
    if len(image_array.shape) == 2:
        # Convert grayscale to RGB
        image_array = np.stack((image_array,) * 3, axis=-1)
    
    # If image has 4 channels (RGBA), convert to RGB
    elif image_array.shape[2] == 4:
        # Convert RGBA to RGB
        image_array = image_array[:, :, :3]
    
    # If image has 1 channel with shape (h, w, 1), convert to RGB
    elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = np.concatenate([image_array] * 3, axis=2)
    
    return image_array

def normalize_dicom_pixel_array(pixel_array):
    """Normalize DICOM pixel array for better visualization.
    
    Args:
        pixel_array (numpy.ndarray): DICOM pixel array
        
    Returns:
        numpy.ndarray: Normalized pixel array for visualization
    """
    # DICOM images often have higher bit depth, normalize to 0-255 range
    if pixel_array.max() > 255:
        # Apply window-level adjustment for better contrast
        min_val = np.percentile(pixel_array, 1)
        max_val = np.percentile(pixel_array, 99)
        
        # Clip values outside the window
        clipped = np.clip(pixel_array, min_val, max_val)
        
        # Rescale to 0-255
        rescaled = (clipped - min_val) / (max_val - min_val) * 255.0
        return rescaled.astype(np.uint8)
    
    return pixel_array
