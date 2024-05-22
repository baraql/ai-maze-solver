import numpy as np

focal_length = 92
principal_point = (160, 120)
H = 2.5
image_height, image_width = 240, 320

white_ceil = np.array([255, 255, 255])
white_floor = np.array([236, 236, 236])
light_blue = np.array([218, 184, 170])
tolerance = 20

def map_walls(frame, min_wall_size):
    mask_white_ceil = np.all(np.abs(frame - white_ceil) <= tolerance, axis=-1)
    mask_white_floor = np.all(np.abs(frame - white_floor) <= tolerance, axis=-1)
    mask_light_blue = np.all(np.abs(frame - light_blue) <= tolerance, axis=-1)

    mask_white = mask_white_ceil | mask_white_floor
    mask_walls = ~(mask_white | mask_light_blue)

    wall_heights = []
    valid_columns = []
    for col in range(image_width):
        column_pixels = mask_walls[:, col]
        if column_pixels[0] == 0 and column_pixels[-1] == 0:
            height = column_pixels.sum()
            if height > min_wall_size:
                wall_heights.append(height)
                valid_columns.append(col)

    wall_heights = np.array(wall_heights)
    Z_coordinates = (focal_length * H) / wall_heights
    X_coordinates = ((np.array(valid_columns) - principal_point[0]) * Z_coordinates) / focal_length

    wall_points = np.vstack((X_coordinates, Z_coordinates)).T

    return wall_points