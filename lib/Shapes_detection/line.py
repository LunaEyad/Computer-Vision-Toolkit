import math
from lib.Edge_detection.canny import apply_canny
import numpy as np




def line_detection( image, threshold):
    # Extract image edges using Canny detector
    edges = apply_canny(image, low_threshold=0.1, high_threshold=0.3)

    height, width = edges.shape[:2]
    diag_len = math.ceil(math.sqrt(height * height + width * width))

    # Initialize parameter space (r, theta) with suitable steps
    thetas = np.deg2rad(np.arange(-90, 90))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    num_thetas = len(thetas)

    # Create accumulator array and initialize to zero for each edge pixel
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edges)

    # For each theta, calculate r = x * cos(theta) + y * sin(theta) and increment accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            r = int(round(x * cos_thetas[t_idx] + y * sin_thetas[t_idx]))
            accumulator[r + diag_len, t_idx] += 1

    # Find lines based on threshold
    filtered_lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] >= threshold:
                r = r_idx - diag_len
                theta = thetas[t_idx]
                filtered_lines.append((r, theta))

    return filtered_lines
