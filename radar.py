import numpy as np
import cv2


w = .33
h = .07
roi_height = 3  


height, width = 240, 320
white_ceil = np.array([255, 255, 255])
white_floor = np.array([236, 236, 236])
light_blue = np.array([218, 184, 170])
tolerance = 20

roi_width = int(width * w) 
y_start = int(height * (1 - h))  
x_start = int((width - roi_width) / 2)
x_end = x_start + roi_width

def radar_mask(frame):
    roi = frame[y_start:y_start + roi_height, x_start:x_end]

    mask_light_blue = np.all(np.abs(roi - light_blue) <= tolerance, axis=-1).astype(np.uint8) * 255
    mask_white_ceil = np.all(np.abs(roi - white_ceil) <= tolerance, axis=-1)
    mask_white_floor = np.all(np.abs(roi - white_floor) <= tolerance, axis=-1)
    
    mask_white = mask_white_ceil | mask_white_floor
    mask_both = np.logical_or(mask_light_blue.astype(bool), mask_white.astype(bool))

    roi[~mask_both] = [0, 0, 255]
    roi[mask_both] = [0, 255, 0] 

    frame[y_start:y_start + roi_height, x_start:x_end] = roi

    return np.any(np.all(roi == [0, 0, 255], axis=-1))