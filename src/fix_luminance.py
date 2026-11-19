import re

with open('D:/Projects/EDD/v2/src/train_stage1_svm.py', 'r') as f:
    content = f.read()

# Find and replace the entire extract_luminance_features function
# Match from "def extract_luminance_features" to just before "# Data Loading"

start_marker = 'def extract_luminance_features(img_rgb, img_size=(224, 224)):'
end_marker = '# Data Loading'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find markers")
else:
    new_func = '''def extract_luminance_features(img_rgb, img_size=(224, 224)):
    """
    Extract mean luminance from lens ROI (paper Section II-E).
    
    Steps:
    1. Resize to standard size (uint8 for OpenCV ops)
    2. Median filter for denoising (uint8)
    3. Watershed segmentation for lens ROI (uint8)
    4. Normalize to [0,1] for BV computation (float)
    5. Compute BV = 0.299*R + 0.587*G + 0.114*B (paper Eq. 6)
    6. Convert BV to luminance: luminance = 2^BV / 0.0929 (paper Eq. 8)
    7. Return mean luminance of lens region
    
    Returns:
        mean_luminance: float, mean luminance value of lens region
        lens_mask: np.ndarray, binary mask of lens region (for debugging)
    """
    # Resize (uint8 for OpenCV compatibility)
    img_uint8 = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)

    # Median filter for denoising (paper Section II-C) — requires uint8
    img_filtered = median_filter(img_uint8, ksize=5)

    # Watershed segmentation for lens ROI (paper Section II-D) — requires uint8
    lens_mask = watershed_segmentation(img_filtered)

    # Normalize filtered image to [0,1] for BV computation
    img_float = img_filtered.astype(np.float32) / 255.0

    # Compute Brightness Value (BV) = 0.299*R + 0.587*G + 0.114*B (paper Eq. 6)
    # With normalized [0,1] pixels, BV ranges 0-1
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    BV = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert BV to luminance (paper Eq. 8, calibrated for iPhone X)
    # luminance = 2^BV / 0.0929. With BV in [0,1], luminance ranges [1.08, 2.15]
    luminance = np.power(2.0, BV) / 0.0929

    # Mask to lens region only
    lens_mask_bool = lens_mask > 0
    if np.sum(lens_mask_bool) == 0:
        # Fallback: use center region if watershed fails
        h, w = luminance.shape
        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[h//4:3*h//4, w//4:3*w//4] = True
        lens_mask_bool = center_mask

    mean_luminance = float(np.mean(luminance[lens_mask_bool]))

    return mean_luminance, lens_mask


'''

    content = content[:start_idx] + new_func + content[end_idx:]
    
    with open('D:/Projects/EDD/v2/src/train_stage1_svm.py', 'w') as f:
        f.write(content)
    print('Function replacement successful')
