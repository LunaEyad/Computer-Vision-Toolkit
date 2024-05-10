import cv2
import numpy as np


def apply_prewitt( image: np.ndarray) -> np.ndarray:
    """
    Applies the Prewitt edge detection algorithm to the input image.
    """
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    grad_x = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, prewitt_y)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude *= (
            255.0 / gradient_magnitude.max()
    )  # Normalize the gradient magnitude
    return gradient_magnitude