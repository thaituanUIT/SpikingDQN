import cv2
import numpy as np

def crop_and_resize(image, box, output_size=(224, 224)):
    """
    Crops the image according to the bounding box and resizes it to output_size
    while maintaining aspect ratio by padding with black pixels on the shorter edge.
    
    Args:
        image (np.ndarray): Original image (H, W, C)
        box (list or np.ndarray): Bounding box [xmin, ymin, xmax, ymax]
        output_size (tuple): Desired output dimensions (W, H)
        
    Returns:
        np.ndarray: Cropped and padded image with dimensions output_size.
    """
    height, width, _ = image.shape
    new_mask = np.asarray(box).astype("int")
    
    # Clip box to image boundaries
    new_mask[0] = max(new_mask[0], 0)
    new_mask[1] = max(new_mask[1], 0)
    new_mask[2] = min(new_mask[2], width)
    new_mask[3] = min(new_mask[3], height)
    
    cropped_image = image[new_mask[1]:new_mask[3], new_mask[0]:new_mask[2]]
    new_height, new_width = cropped_image.shape[:2]

    if new_height == 0 or new_width == 0:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.float32)

    # Maintain aspect ratio by padding the shortest axis
    max_dim = max(new_height, new_width)
    pad_top = (max_dim - new_height) // 2
    pad_bot = max_dim - new_height - pad_top
    pad_left = (max_dim - new_width) // 2
    pad_right = max_dim - new_width - pad_left
    
    padded_image = cv2.copyMakeBorder(
        cropped_image, 
        pad_top, pad_bot, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    resized_image = cv2.resize(padded_image, output_size)
    
    return resized_image
