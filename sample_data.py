import numpy as np
from PIL import Image
import io
import os
import base64
import streamlit as st

# The sample image data will be generated programmatically using numpy
# We'll create different types of sample images that mimic lung scans

def get_sample_image_names():
    """Get names of available sample images.
    
    Returns:
        list: List of sample image names
    """
    return [
        "Normal Lung Scan",
        "Nodule Present",
        "Early Cancer Signs",
        "Advanced Cancer",
        "Pneumonia Case"
    ]

def get_sample_image(image_name):
    """Get a sample image by name.
    
    Args:
        image_name (str): Name of the sample image
        
    Returns:
        PIL.Image: The requested sample image
    """
    # Create a mapping of image names to generation functions
    image_generators = {
        "Normal Lung Scan": generate_normal_lung,
        "Nodule Present": generate_nodule_lung,
        "Early Cancer Signs": generate_early_cancer,
        "Advanced Cancer": generate_advanced_cancer,
        "Pneumonia Case": generate_pneumonia
    }
    
    # Check if the requested image name exists
    if image_name not in image_generators:
        return None
    
    # Generate the image
    return image_generators[image_name]()

def generate_normal_lung(size=(512, 512)):
    """Generate a synthetic image of a normal lung scan.
    
    Args:
        size (tuple): Size of the image to generate
        
    Returns:
        PIL.Image: A synthetic normal lung scan image
    """
    # Create a base gray image
    img = np.ones(size, dtype=np.uint8) * 200
    
    # Create lung shapes (two elliptical dark regions)
    h, w = size
    y, x = np.ogrid[:h, :w]
    
    # Left lung
    left_ellipse = ((x - w/4)**2 / (w/6)**2 + (y - h/2)**2 / (h/3)**2) <= 1
    
    # Right lung
    right_ellipse = ((x - 3*w/4)**2 / (w/6)**2 + (y - h/2)**2 / (h/3)**2) <= 1
    
    # Apply lung shapes
    img[left_ellipse] = 100
    img[right_ellipse] = 100
    
    # Add some texture/noise
    noise = np.random.normal(0, 10, size).astype(np.uint8)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    # Add subtle lung structures (bronchi, vessels)
    for i in range(20):
        # Random line-like structures within lungs
        if np.random.random() < 0.5:
            # Left lung vessels
            start_x = int(w/4 + np.random.normal(0, w/12))
            start_y = int(h/2 + np.random.normal(0, h/6))
            end_x = int(start_x + np.random.normal(0, w/20))
            end_y = int(start_y + np.random.normal(0, h/20))
            
            if left_ellipse[start_y, start_x] and left_ellipse[end_y, end_x]:
                cv2_line(img, (start_x, start_y), (end_x, end_y), 130, 2)
        else:
            # Right lung vessels
            start_x = int(3*w/4 + np.random.normal(0, w/12))
            start_y = int(h/2 + np.random.normal(0, h/6))
            end_x = int(start_x + np.random.normal(0, w/20))
            end_y = int(start_y + np.random.normal(0, h/20))
            
            if right_ellipse[start_y, start_x] and right_ellipse[end_y, end_x]:
                cv2_line(img, (start_x, start_y), (end_x, end_y), 130, 2)
    
    # Convert to PIL Image
    return Image.fromarray(img)

def generate_nodule_lung(size=(512, 512)):
    """Generate a synthetic image of a lung scan with a nodule.
    
    Args:
        size (tuple): Size of the image to generate
        
    Returns:
        PIL.Image: A synthetic lung scan with nodule
    """
    # Start with a normal lung
    img_normal = np.array(generate_normal_lung(size))
    
    h, w = size
    y, x = np.ogrid[:h, :w]
    
    # Create a nodule (small circular region)
    # Place it randomly in one of the lungs
    if np.random.random() < 0.5:
        # Left lung nodule
        center_x = int(w/4 + np.random.normal(0, w/12))
        center_y = int(h/2 + np.random.normal(0, h/6))
    else:
        # Right lung nodule
        center_x = int(3*w/4 + np.random.normal(0, w/12))
        center_y = int(h/2 + np.random.normal(0, h/6))
    
    # Create nodule
    radius = int(min(h, w) * 0.03)  # Small nodule
    nodule = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
    
    # Make nodule brighter (abnormal tissue)
    img_normal[nodule] = 160
    
    # Add a slight halo around the nodule (tissue reaction)
    halo = ((x - center_x)**2 + (y - center_y)**2) <= (radius*1.5)**2
    halo_only = halo & ~nodule
    img_normal[halo_only] = np.clip(img_normal[halo_only] + 20, 0, 255)
    
    # Convert to PIL Image
    return Image.fromarray(img_normal)

def generate_early_cancer(size=(512, 512)):
    """Generate a synthetic image of a lung scan with early cancer signs.
    
    Args:
        size (tuple): Size of the image to generate
        
    Returns:
        PIL.Image: A synthetic lung scan with early cancer signs
    """
    # Start with a normal lung
    img_normal = np.array(generate_normal_lung(size))
    
    h, w = size
    y, x = np.ogrid[:h, :w]
    
    # Create multiple small nodules and opacity areas
    for _ in range(3):
        # Randomly choose left or right lung
        if np.random.random() < 0.5:
            # Left lung
            center_x = int(w/4 + np.random.normal(0, w/12))
            center_y = int(h/2 + np.random.normal(0, h/6))
        else:
            # Right lung
            center_x = int(3*w/4 + np.random.normal(0, w/12))
            center_y = int(h/2 + np.random.normal(0, h/6))
        
        # Create small nodule or opacity
        radius = int(min(h, w) * (0.02 + 0.03 * np.random.random()))
        lesion = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        
        # Make lesion brighter or darker (abnormal tissue)
        intensity = 140 + int(np.random.normal(0, 20))
        img_normal[lesion] = intensity
    
    # Add some irregular borders to mimic infiltration
    img_normal = distort_edges(img_normal)
    
    # Convert to PIL Image
    return Image.fromarray(img_normal)

def generate_advanced_cancer(size=(512, 512)):
    """Generate a synthetic image of a lung scan with advanced cancer.
    
    Args:
        size (tuple): Size of the image to generate
        
    Returns:
        PIL.Image: A synthetic lung scan with advanced cancer
    """
    # Start with a normal lung
    img_normal = np.array(generate_normal_lung(size))
    
    h, w = size
    y, x = np.ogrid[:h, :w]
    
    # Choose a lung for the primary mass
    if np.random.random() < 0.5:
        # Left lung mass
        center_x = int(w/4 + np.random.normal(0, w/16))
        center_y = int(h/2 + np.random.normal(0, h/8))
        primary_lung = "left"
    else:
        # Right lung mass
        center_x = int(3*w/4 + np.random.normal(0, w/16))
        center_y = int(h/2 + np.random.normal(0, h/8))
        primary_lung = "right"
    
    # Create a large irregular mass
    radius = int(min(h, w) * 0.15)  # Large mass
    
    # Create irregular shape by distorting a circle
    base_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
    
    # Apply distortion to create irregularity
    distorted_mask = distort_mask(base_mask, strength=0.3)
    
    # Make the mass brighter with internal heterogeneity
    mass_base = np.random.uniform(150, 180, size=(h, w)).astype(np.uint8)
    img_normal[distorted_mask] = mass_base[distorted_mask]
    
    # Add secondary smaller nodules in both lungs (metastases)
    for _ in range(5):
        if (primary_lung == "left" and np.random.random() < 0.7) or primary_lung == "right":
            # Add to right lung
            sec_center_x = int(3*w/4 + np.random.normal(0, w/10))
            sec_center_y = int(h/2 + np.random.normal(0, h/5))
        else:
            # Add to left lung
            sec_center_x = int(w/4 + np.random.normal(0, w/10))
            sec_center_y = int(h/2 + np.random.normal(0, h/5))
        
        # Create small nodule
        sec_radius = int(min(h, w) * (0.02 + 0.04 * np.random.random()))
        sec_mask = ((x - sec_center_x)**2 + (y - sec_center_y)**2) <= sec_radius**2
        
        # Make nodule brighter
        sec_intensity = 150 + int(np.random.normal(0, 15))
        img_normal[sec_mask] = sec_intensity
    
    # Add pleural effusion (fluid) to the affected side
    if primary_lung == "left":
        # Add to left side
        effusion_region = (x < w/3) & (y > 2*h/3)
    else:
        # Add to right side
        effusion_region = (x > 2*w/3) & (y > 2*h/3)
    
    # Make effusion region more opaque (brighter)
    img_normal[effusion_region] = np.clip(img_normal[effusion_region] + 40, 0, 255)
    
    # Convert to PIL Image
    return Image.fromarray(img_normal)

def generate_pneumonia(size=(512, 512)):
    """Generate a synthetic image of a lung scan with pneumonia.
    
    Args:
        size (tuple): Size of the image to generate
        
    Returns:
        PIL.Image: A synthetic lung scan with pneumonia
    """
    # Start with a normal lung
    img_normal = np.array(generate_normal_lung(size))
    
    h, w = size
    
    # Create patchy opacities in the lungs
    # First define the lung regions more precisely
    y, x = np.ogrid[:h, :w]
    
    # Left lung
    left_lung = ((x - w/4)**2 / (w/6)**2 + (y - h/2)**2 / (h/3)**2) <= 1
    
    # Right lung
    right_lung = ((x - 3*w/4)**2 / (w/6)**2 + (y - h/2)**2 / (h/3)**2) <= 1
    
    # Choose which lung(s) to affect
    both_lungs = np.random.random() < 0.3
    right_affected = both_lungs or (not both_lungs and np.random.random() < 0.5)
    left_affected = both_lungs or (not both_lungs and not right_affected)
    
    # Create patchy opacities
    if left_affected:
        # Apply multiple patches to left lung
        for _ in range(3):
            patch_center_x = int(w/4 + np.random.normal(0, w/12))
            patch_center_y = int(h/2 + np.random.normal(0, h/6))
            
            # Create an irregular patch
            base_radius = int(min(h, w) * (0.05 + 0.07 * np.random.random()))
            base_patch = ((x - patch_center_x)**2 + (y - patch_center_y)**2) <= base_radius**2
            
            # Only keep parts within the lung
            patch = base_patch & left_lung
            
            # Make the patch brighter (pneumonia opacity)
            img_normal[patch] = np.clip(img_normal[patch] + 60, 0, 255)
    
    if right_affected:
        # Apply multiple patches to right lung
        for _ in range(3):
            patch_center_x = int(3*w/4 + np.random.normal(0, w/12))
            patch_center_y = int(h/2 + np.random.normal(0, h/6))
            
            # Create an irregular patch
            base_radius = int(min(h, w) * (0.05 + 0.07 * np.random.random()))
            base_patch = ((x - patch_center_x)**2 + (y - patch_center_y)**2) <= base_radius**2
            
            # Only keep parts within the lung
            patch = base_patch & right_lung
            
            # Make the patch brighter (pneumonia opacity)
            img_normal[patch] = np.clip(img_normal[patch] + 60, 0, 255)
    
    # Add some haziness overall to affected lungs
    if left_affected:
        img_normal[left_lung] = np.clip(img_normal[left_lung] + 15, 0, 255)
    if right_affected:
        img_normal[right_lung] = np.clip(img_normal[right_lung] + 15, 0, 255)
    
    # Convert to PIL Image
    return Image.fromarray(img_normal)

# Helper functions for image generation

def cv2_line(img, pt1, pt2, color, thickness=1):
    """Draw a line on the image (simplified version of cv2.line)."""
    # Bresenham's line algorithm (simplified)
    x1, y1 = pt1
    x2, y2 = pt2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while x1 != x2 or y1 != y2:
        # Check bounds
        if 0 <= y1 < img.shape[0] and 0 <= x1 < img.shape[1]:
            # Draw a small region around the point for thickness
            for i in range(max(0, y1-thickness//2), min(img.shape[0], y1+thickness//2+1)):
                for j in range(max(0, x1-thickness//2), min(img.shape[1], x1+thickness//2+1)):
                    img[i, j] = color
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def distort_mask(mask, strength=0.2):
    """Apply random distortion to a binary mask to create irregular shapes."""
    h, w = mask.shape
    distorted = mask.copy()
    
    # Apply random distortions
    for _ in range(int(50 * strength)):
        # Find a point on the edge of the mask
        edge_points = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                if mask[i, j] and not (mask[i-1:i+2, j-1:j+2].all()):
                    edge_points.append((i, j))
        
        if not edge_points:
            continue
        
        # Select a random edge point
        i, j = edge_points[np.random.randint(0, len(edge_points))]
        
        # Add or remove a small region
        radius = int(min(h, w) * 0.02 * strength)
        y, x = np.ogrid[:h, :w]
        region = ((x - j)**2 + (y - i)**2) <= radius**2
        
        if np.random.random() < 0.6:
            # Add to the mask
            distorted[region] = True
        else:
            # Remove from the mask
            distorted[region] = False
    
    return distorted

def distort_edges(img):
    """Apply distortion to lung edges to create irregularity."""
    h, w = img.shape
    result = img.copy()
    
    # Add some irregular protrusions/indentations
    for _ in range(10):
        # Random position
        x = np.random.randint(w//4, 3*w//4)
        y = np.random.randint(h//3, 2*h//3)
        
        # Random direction
        angle = np.random.uniform(0, 2*np.pi)
        length = np.random.randint(10, 30)
        
        # Calculate end point
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        
        # Ensure end point is within bounds
        end_x = max(0, min(w-1, end_x))
        end_y = max(0, min(h-1, end_y))
        
        # Create a line-like distortion
        cv2_line(result, (x, y), (end_x, end_y), 
                 color=int(img[y, x] + np.random.normal(20, 10)), 
                 thickness=np.random.randint(3, 8))
    
    return result
