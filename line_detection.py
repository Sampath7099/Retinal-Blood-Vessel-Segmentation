import cv2
import numpy as np
from scipy.ndimage import convolve
from typing import List

def create_line_kernel(length, angle_degrees):
    """
    Creates a linear kernel (structuring element) oriented at a specific angle.
    """
    # Create a blank square grid large enough to hold the rotating line
    grid_size = int(np.ceil(length * 1.5))
    if grid_size % 2 == 0: 
        grid_size += 1
    
    kernel = np.zeros((grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    
    # Draw a horizontal line of 1s in the center
    # Length is spread around the center
    half_len = length // 2
    kernel[center, center-half_len : center+half_len+1] = 1.0
    
    # Normalize so the sum is 1 (Mean filter behavior along the line)
    if np.sum(kernel) > 0:
        kernel /= np.sum(kernel)
        
    # Rotate the kernel to the desired angle
    M = cv2.getRotationMatrix2D((center, center), angle_degrees, 1.0)
    rotated_kernel = cv2.warpAffine(kernel, M, (grid_size, grid_size), flags=cv2.INTER_LINEAR)
    
    # Re-normalize after rotation interpolation to maintain energy
    if np.sum(rotated_kernel) > 0:
        rotated_kernel /= np.sum(rotated_kernel)
        
    return rotated_kernel

def compute_single_scale_response(image, length, num_angles=12):
    """
    Computes the maximum line response for a specific line length across multiple angles.
    """
    max_response = np.zeros_like(image, dtype=np.float32)
    angle_step = 180.0 / num_angles
    
    for i in range(num_angles):
        angle = i * angle_step
        kernel = create_line_kernel(length, angle)
        
        # Convolve: Measures average intensity along the line
        # Bright vessels match with bright lines
        response = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
        
        # Take the maximum response across all orientations
        np.maximum(max_response, response, out=max_response)
        
    return max_response

def multiscale_line_detection(image: np.ndarray):
    """
    Aggregates line detection responses across multiple scales (lengths).
    """
    # Define scales (lengths of vessels to detect)
    # Shorter lengths for capillaries, longer for main vessels
    line_lengths = [3, 5, 7, 9, 11, 15]
    
    combined_response = np.zeros_like(image, dtype=np.float32)
    
    for length in line_lengths:
        response = compute_single_scale_response(image, length, num_angles=12)
        
        # Normalize response relative to length? 
        # Longer lines accumulate more "signal" naturally if summed, but here we used mean (normalized kernel).
        # Averaging responses across scales is robust.
        combined_response += response
        
    # Average across scales
    combined_response /= len(line_lengths)
    
    return combined_response

def normalize_output(image: np.ndarray):
    """Normalizes the final confidence map to 0.0 - 1.0 range."""
    min_val = image.min()
    max_val = image.max()
    
    if max_val - min_val > 1e-6:
        return (image - min_val) / (max_val - min_val)
    return np.zeros_like(image)

def line_detection(dataset: List[np.ndarray]) -> List[np.ndarray]:
    """
    Wrapper function for the Line Detection Algorithm.
    
    Args:
        dataset (list): List of preprocessed images (float32). 
                        Expected to be output from preprocess.py (vessels bright).
                        
    Returns:
        list: List of confidence maps (float32, 0.0 to 1.0).
    """
    confidence_maps = []
    
    for raw_float_img in dataset:
        # 1. Normalize input to 0-1 range (Standardization)
        mn, mx = raw_float_img.min(), raw_float_img.max()
        if mx > mn:
            norm_input = (raw_float_img - mn) / (mx - mn)
        else:
            norm_input = np.zeros_like(raw_float_img)
            
        # 2. Run Multi-scale Line Detection
        raw_response = multiscale_line_detection(norm_input)
        
        # 3. Final Normalization
        confidence_map = normalize_output(raw_response)
        confidence_maps.append(confidence_map)
        
    return confidence_maps