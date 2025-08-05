import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Union, Optional, List

def preprocess_image(image_array: Union[np.ndarray, Image.Image], 
                    target_size: Tuple[int, int] = (224, 224),
                    normalize: bool = True,
                    norm_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Preprocess image for model input.
    
    Args:
        image_array (numpy.ndarray or PIL.Image): Input image as a numpy array or PIL Image
        target_size (tuple): Target size for resizing (width, height)
        normalize (bool): Whether to normalize pixel values
        norm_range (tuple): Normalization range, either (0, 1) or (-1, 1)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
        
    Raises:
        ValueError: If image is empty, None, or target_size is invalid
        TypeError: If image_array is not a supported type
    """
    # Input validation
    if image_array is None:
        raise ValueError("Image array cannot be None")
    
    if len(target_size) != 2 or any(s <= 0 for s in target_size):
        raise ValueError("Target size must be a tuple of two positive integers")
    
    if norm_range not in [(0, 1), (-1, 1)]:
        raise ValueError("Normalization range must be either (0, 1) or (-1, 1)")
    
    # Convert to numpy array if needed
    if isinstance(image_array, Image.Image):
        image_array = np.array(image_array)
    elif not isinstance(image_array, np.ndarray):
        raise TypeError("Image must be a numpy array or PIL Image")
    
    # Check for empty array
    if image_array.size == 0:
        raise ValueError("Empty image array provided")
    
    # Ensure image has 3 channels (convert grayscale to RGB if needed)
    image_array = ensure_color_channels(image_array)
    
    # Resize image to target size (cv2.resize expects (width, height))
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values if requested
    if normalize:
        if norm_range == (0, 1):
            normalized_image = resized_image / 255.0
        else:  # (-1, 1)
            normalized_image = (resized_image / 255.0) * 2 - 1
        return normalized_image
    
    return resized_image

def preprocess_batch(image_list: List[Union[np.ndarray, Image.Image]], 
                    target_size: Tuple[int, int] = (224, 224),
                    normalize: bool = True,
                    norm_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """Preprocess a batch of images for model input.
    
    Args:
        image_list (list): List of images as numpy arrays or PIL Images
        target_size (tuple): Target size for resizing (width, height)
        normalize (bool): Whether to normalize pixel values
        norm_range (tuple): Normalization range, either (0, 1) or (-1, 1)
        
    Returns:
        numpy.ndarray: Batch of preprocessed images with shape (batch_size, height, width, channels)
        
    Raises:
        ValueError: If image_list is empty
    """
    if not image_list:
        raise ValueError("Image list cannot be empty")
    
    processed_images = []
    for i, image in enumerate(image_list):
        try:
            processed_img = preprocess_image(image, target_size, normalize, norm_range)
            processed_images.append(processed_img)
        except Exception as e:
            raise ValueError(f"Error processing image at index {i}: {str(e)}")
    
    return np.array(processed_images)

def ensure_color_channels(image_array: np.ndarray) -> np.ndarray:
    """Ensure the image has 3 color channels (convert grayscale to RGB if needed).
    
    Args:
        image_array (numpy.ndarray): Input image as a numpy array
        
    Returns:
        numpy.ndarray: Image with 3 color channels
        
    Raises:
        ValueError: If image has unsupported number of dimensions or channels
    """
    # Handle different image formats
    if len(image_array.shape) == 2:
        # Grayscale image (H, W) -> RGB (H, W, 3)
        image_array = np.stack((image_array,) * 3, axis=-1)
    
    elif len(image_array.shape) == 3:
        channels = image_array.shape[2]
        
        if channels == 1:
            # Single channel (H, W, 1) -> RGB (H, W, 3)
            image_array = np.concatenate([image_array] * 3, axis=2)
        
        elif channels == 3:
            # Already RGB, no change needed
            pass
        
        elif channels == 4:
            # RGBA -> RGB (remove alpha channel)
            image_array = image_array[:, :, :3]
        
        else:
            raise ValueError(f"Unsupported number of channels: {channels}")
    
    else:
        raise ValueError(f"Unsupported image dimensions: {image_array.shape}")
    
    return image_array

def normalize_dicom_pixel_array(pixel_array: np.ndarray, 
                               window_percentiles: Tuple[float, float] = (1, 99),
                               apply_clahe: bool = False) -> np.ndarray:
    """Normalize DICOM pixel array for better visualization and processing.
    
    Args:
        pixel_array (numpy.ndarray): DICOM pixel array
        window_percentiles (tuple): Percentiles for windowing (min_percentile, max_percentile)
        apply_clahe (bool): Whether to apply Contrast Limited Adaptive Histogram Equalization
        
    Returns:
        numpy.ndarray: Normalized pixel array for visualization (0-255 range, uint8)
        
    Raises:
        ValueError: If pixel_array is empty or percentiles are invalid
    """
    if pixel_array.size == 0:
        raise ValueError("Empty pixel array provided")
    
    if not (0 <= window_percentiles[0] < window_percentiles[1] <= 100):
        raise ValueError("Invalid percentiles: must be 0 <= min_percentile < max_percentile <= 100")
    
    # Handle different data types and ranges
    if pixel_array.dtype in [np.uint8] and pixel_array.max() <= 255:
        # Already in 0-255 range
        normalized = pixel_array.astype(np.uint8)
    else:
        # Apply windowing for better contrast
        min_val = np.percentile(pixel_array, window_percentiles[0])
        max_val = np.percentile(pixel_array, window_percentiles[1])
        
        # Avoid division by zero
        if max_val == min_val:
            normalized = np.full_like(pixel_array, 128, dtype=np.uint8)
        else:
            # Clip values outside the window
            clipped = np.clip(pixel_array, min_val, max_val)
            
            # Rescale to 0-255
            rescaled = (clipped - min_val) / (max_val - min_val) * 255.0
            normalized = rescaled.astype(np.uint8)
    
    # Apply CLAHE if requested (useful for medical images)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(normalized)
    
    return normalized

def save_preprocessed_image(image_array: np.ndarray, 
                           output_path: str,
                           denormalize: bool = True,
                           original_range: Tuple[float, float] = (0, 1)) -> None:
    """Save a preprocessed image to disk.
    
    Args:
        image_array (numpy.ndarray): Preprocessed image array
        output_path (str): Path to save the image
        denormalize (bool): Whether to denormalize the image before saving
        original_range (tuple): Original normalization range used
        
    Raises:
        ValueError: If image_array is invalid
        IOError: If unable to save the image
    """
    if image_array.size == 0:
        raise ValueError("Empty image array provided")
    
    # Denormalize if needed
    if denormalize:
        if original_range == (0, 1):
            image_array = (image_array * 255).astype(np.uint8)
        elif original_range == (-1, 1):
            image_array = ((image_array + 1) / 2 * 255).astype(np.uint8)
    
    # Ensure proper format for saving
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(output_path, image_bgr)
    else:
        success = cv2.imwrite(output_path, image_array)
    
    if not success:
        raise IOError(f"Failed to save image to {output_path}")

def get_image_stats(image_array: np.ndarray) -> dict:
    """Get statistical information about an image array.
    
    Args:
        image_array (numpy.ndarray): Input image array
        
    Returns:
        dict: Dictionary containing image statistics
    """
    if image_array.size == 0:
        return {"error": "Empty image array"}
    
    stats = {
        "shape": image_array.shape,
        "dtype": str(image_array.dtype),
        "min_value": float(np.min(image_array)),
        "max_value": float(np.max(image_array)),
        "mean_value": float(np.mean(image_array)),
        "std_value": float(np.std(image_array)),
        "unique_values": len(np.unique(image_array))
    }
    
    return stats

# Example usage and testing functions
def example_usage():
    """Example usage of the preprocessing functions."""
    try:
        # Create a sample image
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Preprocess single image
        processed = preprocess_image(sample_image, target_size=(224, 224))
        print(f"Original shape: {sample_image.shape}")
        print(f"Processed shape: {processed.shape}")
        print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # Process batch
        image_batch = [sample_image] * 5
        batch_processed = preprocess_batch(image_batch, target_size=(224, 224))
        print(f"Batch shape: {batch_processed.shape}")
        
        # Get image statistics
        stats = get_image_stats(processed)
        print(f"Image stats: {stats}")
        
    except Exception as e:
        print(f"Error in example usage: {e}")

if __name__ == "__main__":
    example_usage()