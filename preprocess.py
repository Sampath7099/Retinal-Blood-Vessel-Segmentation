import cv2
import numpy as np
import gc

def extract_green_channel(img_rgb):
    if len(img_rgb.shape) == 2:
        return img_rgb
    return img_rgb[:, :, 1]

def apply_clahe(img_gray, clip_limit=2.0, tile_size=(8, 8)):
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img_gray)

def perform_opening(img_gray, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

def background_homogenization(image):
    img_float = image.astype(np.float32)
    mean_bg = cv2.blur(img_float, (69, 69))
    diff = img_float - mean_bg
    q75, q25 = np.percentile(diff, [75, 25])
    iqr = q75 - q25
    median = np.median(diff)
    
    if iqr == 0:
        iqr = 1e-6
        
    normalized = (diff - median) / iqr
    return normalized

def invert_image(image):
    return -1.0 * image

def preprocess(dataset):
    processed_dataset = []
    
    # Iterate through the dataset directly (no batching)
    for img_rgb in dataset:
        green = extract_green_channel(img_rgb)
        clahed = apply_clahe(green)
        opened = perform_opening(clahed)
        homogenized = background_homogenization(opened)
        final_img = invert_image(homogenized)
        
        processed_dataset.append(final_img)
        
    return processed_dataset