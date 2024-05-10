import cv2
import numpy as np


def apply_gaussian_channel(image: np.ndarray, sigma=5) -> np.ndarray:
    """
    Applies the gaussian filter to the input image.

    """
    # Calculate filter size based on sigma
    # filter_size = 2 * int(2 * sigma * np.pi) + 1
    filter_size = 5  # odd
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    # Calculate the center of the kernel
    m = filter_size // 2
    n = filter_size // 2

    # Generate Gaussian filter kernel
    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma**2)
            x2 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2

    # Normalize Gaussian filter kernel
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

    # Apply Gaussian blur using cv2.filter2D
    im_filtered = cv2.filter2D(image, -1, gaussian_filter)

    return im_filtered
