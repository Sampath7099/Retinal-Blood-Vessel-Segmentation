import cv2
import numpy as np

def create_gaussian_matched_filter(sigma, length, theta):
    """
    Creates a single Gaussian matched filter kernel oriented at specific theta.
    Uses a POSITIVE Gaussian to detect bright vessels (inverted input).
    """
    half_width = int(np.ceil(3 * sigma))
    half_length = length // 2
    x = np.arange(-half_length, half_length + 1, dtype=np.float32)
    y = np.arange(-half_width, half_width + 1, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    
    # Rotate coordinates
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Positive Gaussian kernel (detects bright structures)
    kernel = np.exp(-(Y_rot ** 2) / (2 * sigma ** 2))
    
    # Mask out values outside the length
    kernel[np.abs(X_rot) > half_length] = 0
    
    # Zero-mean normalization
    kernel = kernel - np.mean(kernel)
    
    # Norm-1 normalization
    kernel_norm = np.linalg.norm(kernel)
    if kernel_norm > 1e-10:
        kernel = kernel / kernel_norm
        
    return kernel

def create_multiscale_filter_bank(sigmas, lengths, num_orientations=12):
    """Generates a bank of filters across multiple scales and orientations."""
    filter_bank = {}
    angles = np.linspace(0, np.pi, num_orientations, endpoint=False)
    
    for sigma, length in zip(sigmas, lengths):
        filters_at_scale = []
        for theta in angles:
            kernel = create_gaussian_matched_filter(sigma, length, theta)
            filters_at_scale.append(kernel)
        filter_bank[(sigma, length)] = filters_at_scale
        
    return filter_bank

def apply_matched_filter_single_scale(image, filters):
    """Convolves image with all orientations at a single scale and takes max response."""
    max_response = np.zeros_like(image, dtype=np.float32)
    for kernel in filters:
        # cv2.filter2D is faster than scipy.signal.convolve2d
        response = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
        max_response = np.maximum(max_response, response)
    return max_response

def multiscale_matched_filter(image, filter_bank, enhance_contrast=True):
    """Aggregates responses across multiple scales."""
    img_float = image.astype(np.float32)
    
    # Ensure float range usually 0-1 if coming from preprocess, 
    # but preprocess returns standardized float, so we use as-is.
    
    combined_response = np.zeros_like(img_float, dtype=np.float32)
    scale_count = 0
    
    for (sigma, length), filters in filter_bank.items():
        scale_response = apply_matched_filter_single_scale(img_float, filters)
        
        if enhance_contrast:
            min_val = scale_response.min()
            max_val = scale_response.max()
            if max_val - min_val > 1e-6:
                scale_response = (scale_response - min_val) / (max_val - min_val)
            
            # Gamma < 1 boosts weak vessel responses
            scale_response = np.power(scale_response, 0.5) 
            
        # Weight response by scale (sigma) to normalize magnitude across scales
        weighted_response = scale_response * sigma
        combined_response += weighted_response
        scale_count += 1
        
    if scale_count > 0:
        combined_response = combined_response / scale_count
        
    # Final normalization
    min_val = combined_response.min()
    max_val = combined_response.max()
    if max_val - min_val > 1e-6:
        combined_response = (combined_response - min_val) / (max_val - min_val)
        
    return combined_response

def enhance_vessels(response_map, percentile_threshold=5):
    """Applies sigmoid non-linear enhancement to separate vessels from noise."""
    threshold = np.percentile(response_map, percentile_threshold)
    enhanced = response_map.copy()
    
    # Suppress very low values
    enhanced[enhanced < threshold] = 0
    if enhanced.max() > 0:
        enhanced = enhanced / enhanced.max()
    
    # Sigmoid function centered around 0.4
    k = 10 
    enhanced = 1.0 / (1.0 + np.exp(-k * (enhanced - 0.4))) 
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-10)
    return enhanced

def apply_morphological_cleanup(confidence_map):
    """Cleans up small disconnected noise using morphology."""
    conf_uint8 = (confidence_map * 255).astype(np.uint8)
    
    # Closing bridges gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(conf_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening removes small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    cleaned = cleaned.astype(np.float32) / 255.0
    return cleaned

def compute_matched_filter_confidence(image, filter_bank):
    """Runs the full pipeline for a single image."""
    response = multiscale_matched_filter(image, filter_bank, enhance_contrast=True)
    enhanced = enhance_vessels(response, percentile_threshold=5)
    cleaned = apply_morphological_cleanup(enhanced)
    return cleaned

def matched_filter(dataset):
    """
    Wrapper function to run Matched Filter Algorithm on a preprocessed dataset.
    
    Args:
        dataset (list): List of preprocessed images (float32, vessels bright).
        
    Returns:
        list: List of confidence maps (float32, 0.0 to 1.0).
    """
    # Define parameters for high sensitivity
    sigmas = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    lengths = [5, 7, 9, 11, 13, 15, 17]
    num_orientations = 12 
    
    # Create the filter bank once
    filter_bank = create_multiscale_filter_bank(sigmas, lengths, num_orientations)

    confidence_maps = []
    
    for img in dataset:
        confidence_map = compute_matched_filter_confidence(img, filter_bank)
        confidence_maps.append(confidence_map)
    
    return confidence_maps