import cv2
import numpy as np
from typing import List

def build_morph_input(preproc_list):
    """
    Adapts the output of preprocess.py (list of float images) 
    to uint8 [0-255] format required by the morphological segmenter.
    """
    out = []
    for img_float in preproc_list:
        # Normalize float output (roughly -3 to +3) to 0-255 uint8
        mn, mx = img_float.min(), img_float.max()
        if mx > mn:
            norm = (img_float - mn) / (mx - mn)
        else:
            norm = np.zeros_like(img_float)
            
        out.append((norm * 255.0).astype(np.uint8))
    return out

def morphological_vessel_segmentation(dataset, image_size=565, drive_size=565):
    """
    Core algorithm: Multi-scale morphological processing to extract vessel features.
    """
    processed_data = []
    
    # Scale parameters relative to original DRIVE resolution
    scale_factor = image_size / float(drive_size)
    wc_radius = int(round(2 * scale_factor))
    wc_size = 2 * wc_radius + 1
    
    # Structuring element for Closing (Connects dark gaps in bright vessels)
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wc_size, wc_size))

    # Generate detection kernels for various vessel widths
    base_radii = np.arange(1, 9, dtype=np.int32)
    detection_radii = np.maximum(1, np.round(base_radii * scale_factor).astype(np.int32))
    
    kernels_w = {}
    for r in np.unique(detection_radii):
        w_size = 2 * r + 1
        kernels_w[r] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w_size, w_size))

    radii_for_scales = detection_radii.tolist()
    num_scales = len(radii_for_scales)
    
    # Generate weights for scale averaging
    num_averaged = (num_scales + 1) // 2
    weights = np.arange(num_averaged, 0, -1, dtype=np.float32)
    weights /= weights.sum()
    weights *= 2.0

    for img_u8 in dataset:
        img_float = img_u8.astype(np.float32) / 255.0
        
        # 1. Morphological Closing
        # Fills small dark holes/gaps inside the bright vessel structures
        img_closed = cv2.morphologyEx(img_float, cv2.MORPH_CLOSE, kernel_c)

        scale_results = []
        for r in radii_for_scales:
            kernel_w = kernels_w[r]
            
            # 2. Opening
            # Removes bright features smaller than the current kernel 'r'
            img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_w)
            
            # 3. Top-Hat-like Extraction
            # Difference between the Closed image and the Opened image (Background Estimate)
            # This isolates bright structures of size 'r'
            background_est = np.minimum(img_opened, img_float)
            scale_results.append(img_float - background_est)

        # 4. Multi-scale Aggregation
        averaged_results = []
        for i in range(0, num_scales, 2):
            if i + 1 < num_scales:
                avg = 0.5 * (scale_results[i] + scale_results[i + 1])
                averaged_results.append(avg)
            else:
                averaged_results.append(scale_results[i])

        final_response = np.zeros_like(img_float, dtype=np.float32)
        for w, res in zip(weights, averaged_results):
            final_response = np.maximum(final_response, w * res)

        # 5. Final Normalization
        min_val, max_val = final_response.min(), final_response.max()
        if max_val > min_val:
            normalized = (final_response - min_val) / (max_val - min_val)
        else:
            normalized = final_response
            
        processed_data.append(normalized)

    return processed_data

def morphological(dataset: List[np.ndarray]) -> List[np.ndarray]:
    """
    Wrapper function for the Morphological Segmentation Algorithm.
    
    Args:
        dataset (list): List of preprocessed images (float32). 
                        Expected to be output from preprocess.py (vessels bright).
                        
    Returns:
        list: List of confidence maps (float32, 0.0 to 1.0).
    """
    # 1. Convert float inputs to uint8 for morphology operations
    morph_input = build_morph_input(dataset)
    
    # 2. Run the core algorithm
    # Assuming standard image size 565x584 (DRIVE) or similar
    if len(morph_input) > 0:
        h, w = morph_input[0].shape
        # Use width as the reference dimension size
        confidence_maps = morphological_vessel_segmentation(morph_input, image_size=w)
    else:
        confidence_maps = []
        
    return confidence_maps