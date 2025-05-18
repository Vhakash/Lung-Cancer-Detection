import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import cv2

def get_available_enhancements():
    """Get list of available image enhancement techniques.
    
    Returns:
        list: List of available enhancement techniques
    """
    return [
        "Contrast Enhancement",
        "Histogram Equalization",
        "Adaptive Histogram Equalization",
        "Gaussian Smoothing",
        "Edge Enhancement",
        "Sharpening"
    ]

def apply_enhancement(image, enhancement_type, strength=1.0):
    """Apply the specified enhancement to the image.
    
    Args:
        image (numpy.ndarray): Input image (normalized to [0, 1])
        enhancement_type (str): Type of enhancement to apply
        strength (float): Strength of the enhancement (0.5-1.5)
        
    Returns:
        numpy.ndarray: Enhanced image (normalized to [0, 1])
    """
    # Convert image to 0-255 range for processing
    img_255 = (image * 255).astype(np.uint8)
    
    # Apply the selected enhancement
    if enhancement_type == "Contrast Enhancement":
        enhanced = apply_contrast_enhancement(img_255, strength)
    elif enhancement_type == "Histogram Equalization":
        enhanced = apply_histogram_equalization(img_255)
    elif enhancement_type == "Adaptive Histogram Equalization":
        enhanced = apply_adaptive_histogram_equalization(img_255, strength)
    elif enhancement_type == "Gaussian Smoothing":
        enhanced = apply_gaussian_smoothing(img_255, strength)
    elif enhancement_type == "Edge Enhancement":
        enhanced = apply_edge_enhancement(img_255, strength)
    elif enhancement_type == "Sharpening":
        enhanced = apply_sharpening(img_255, strength)
    else:
        # If no valid enhancement is selected, return the original image
        return image
    
    # Normalize back to [0, 1] range
    return enhanced / 255.0

def apply_contrast_enhancement(image, strength=1.0):
    """Apply contrast enhancement.
    
    Args:
        image (numpy.ndarray): Input image (0-255 range)
        strength (float): Enhancement strength
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Adjust the alpha (contrast) based on strength
    alpha = 1.0 + (strength - 1.0) * 2  # Map 0.5-1.5 to 0-2
    
    # Apply contrast adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    return adjusted

def apply_histogram_equalization(image):
    """Apply histogram equalization.
    
    Args:
        image (numpy.ndarray): Input image (0-255 range)
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # If image has 3 channels, convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        # Convert back to RGB
        equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    else:
        equalized = cv2.equalizeHist(image)
    
    return equalized

def apply_adaptive_histogram_equalization(image, strength=1.0):
    """Apply adaptive histogram equalization (CLAHE).
    
    Args:
        image (numpy.ndarray): Input image (0-255 range)
        strength (float): Enhancement strength
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Create CLAHE object
    clip_limit = 2.0 + (strength - 1.0) * 3  # Adjust clip limit based on strength
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    
    # If image has 3 channels, apply CLAHE to each channel
    if len(image.shape) == 3:
        enhanced = np.zeros_like(image)
        for i in range(3):
            enhanced[:, :, i] = clahe.apply(image[:, :, i])
    else:
        enhanced = clahe.apply(image)
    
    return enhanced

def apply_gaussian_smoothing(image, strength=1.0):
    """Apply Gaussian smoothing.
    
    Args:
        image (numpy.ndarray): Input image (0-255 range)
        strength (float): Smoothing strength
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Adjust kernel size based on strength
    kernel_size = int(3 + (strength - 1.0) * 4)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd size
    
    # Apply Gaussian blur
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return smoothed

def apply_edge_enhancement(image, strength=1.0):
    """Enhance edges in the image.
    
    Args:
        image (numpy.ndarray): Input image (0-255 range)
        strength (float): Enhancement strength
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Detect edges
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Adjust edge strength
    edge_weight = 0.5 * strength
    
    # Combine original image with edges
    if len(image.shape) == 3:
        # For RGB images, add edges to each channel
        enhanced = image.copy()
        for i in range(3):
            enhanced[:, :, i] = cv2.addWeighted(image[:, :, i], 1.0, magnitude, edge_weight, 0)
    else:
        # For grayscale images
        enhanced = cv2.addWeighted(image, 1.0, magnitude, edge_weight, 0)
    
    return enhanced

def apply_sharpening(image, strength=1.0):
    """Apply sharpening to the image.
    
    Args:
        image (numpy.ndarray): Input image (0-255 range)
        strength (float): Sharpening strength
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Define sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Adjust kernel based on strength
    kernel_center = 9 * strength
    kernel = np.array([[-1, -1, -1],
                       [-1, kernel_center, -1],
                       [-1, -1, -1]])
    kernel = kernel / kernel.sum()  # Normalize
    
    # Apply sharpening
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Ensure values are within valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened
