import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List

def compute_hessian_eigenvalues(image: np.ndarray, sigma: float):
    """Computes the eigenvalues of the Hessian matrix at a specific scale."""
    sigma_sq = sigma * sigma
    Ixx = gaussian_filter(image, sigma=sigma, order=[0, 2]) * sigma_sq
    Ixy = gaussian_filter(image, sigma=sigma, order=[1, 1]) * sigma_sq
    Iyy = gaussian_filter(image, sigma=sigma, order=[2, 0]) * sigma_sq
    
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    delta = trace * trace * 0.25 - det
    delta = np.maximum(delta, 0.0)
    sqrt_delta = np.sqrt(delta)
    
    lambda1 = trace * 0.5 - sqrt_delta
    lambda2 = trace * 0.5 + sqrt_delta
    return lambda1, lambda2

def frangi_vesselness(image: np.ndarray, sigma: float, alpha: float = 0.5, beta: float = 15.0):
    """Computes Frangi vesselness response for a single scale."""
    lambda1, lambda2 = compute_hessian_eigenvalues(image, sigma)
    epsilon = 1e-10
    lambda1_abs = np.abs(lambda1)
    lambda2_abs = np.abs(lambda2)
    
    Rb = np.divide(lambda1_abs, lambda2_abs + epsilon)
    S = np.sqrt(lambda1**2 + lambda2**2)
    
    vesselness = np.zeros_like(image, dtype=np.float32)
    
    # Assumption: Preprocessed input has bright vessels (positive structure).
    # For bright ridges, the major eigenvalue (lambda2) should be negative (large curvature).
    valid_mask = lambda2 < 0
    
    vesselness[valid_mask] = (
        np.exp(-Rb[valid_mask]**2 / (2 * alpha**2)) *
        (1 - np.exp(-S[valid_mask]**2 / (2 * beta**2)))
    )
    return vesselness

def multi_scale_frangi(image: np.ndarray, scales=None, alpha=0.5, beta=15.0):
    """Aggregates Frangi response across multiple scales."""
    if scales is None:
        scales = np.arange(0.5, 8.5, 0.5)
    
    max_response = np.zeros_like(image, dtype=np.float32)
    for sigma in scales:
        response = frangi_vesselness(image, sigma, alpha=alpha, beta=beta)
        # Weight by sigma helps preserve larger vessels
        response = response * sigma  
        np.maximum(max_response, response, out=max_response)
        
    if max_response.max() > 0:
        max_response /= max_response.max()
    return max_response

def multiscale_tophat(image: np.ndarray):
    """Computes Multi-scale Top-Hat transform for contrast enhancement."""
    scales = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Input is float 0-1. Convert to uint8 0-255 for OpenCV morphology
    image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    max_response = np.zeros_like(image, dtype=np.float32)
    
    for scale in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*scale+1, 2*scale+1))
        tophat = cv2.morphologyEx(image_uint8, cv2.MORPH_TOPHAT, kernel)
        
        tophat_norm = tophat.astype(np.float32) / 255.0
        # Weighting
        weighted = tophat_norm / scale
        np.maximum(max_response, weighted, out=max_response)
        
    if max_response.max() > 0:
        max_response /= max_response.max()
    return max_response

def compute_vessel_confidence(image: np.ndarray):
    """
    Computes combined confidence map using Frangi and Tophat filters.
    Args:
        image: Normalized float image (0.0 to 1.0) with bright vessels.
    """
    # 1. Multi-scale Frangi (Structure based)
    frangi_response = multi_scale_frangi(
        image, 
        scales=np.arange(0.5, 8.5, 0.5),
        alpha=0.5,
        beta=15.0
    )
    
    # 2. Multi-scale Top-hat (Contrast based)
    tophat_response = multiscale_tophat(image)
    
    # 3. Weighted Fusion
    confidence_map = 0.7 * frangi_response + 0.3 * tophat_response
    confidence_map = np.clip(confidence_map, 0, 1)
    return confidence_map

def scale_space(dataset: List[np.ndarray]) -> List[np.ndarray]:
    """
    Wrapper function for the Scale-Space Algorithm.
    
    Args:
        dataset (list): List of preprocessed images (float32). 
                        Expected to be output from preprocess.py (vessels bright).
                        
    Returns:
        list: List of confidence maps (float32, 0.0 to 1.0).
    """
    confidence_maps = []
    
    for raw_float_img in dataset:
        # 1. Normalize input to 0-1 range
        # preprocess.py outputs range approx -3 to +3 (IQR scaling).
        # We need 0-1 for the Top-Hat and Frangi weighting to work correctly.
        mn, mx = raw_float_img.min(), raw_float_img.max()
        if mx > mn:
            norm_img = (raw_float_img - mn) / (mx - mn)
        else:
            norm_img = np.zeros_like(raw_float_img)
            
        # 2. Compute Confidence
        confidence_map = compute_vessel_confidence(norm_img)
        confidence_maps.append(confidence_map)
        
    return confidence_maps