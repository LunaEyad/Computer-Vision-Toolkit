import cv2
import numpy as np


def apply_roberts( image: np.ndarray) -> np.ndarray:
    """
    Applies the Roberts edge detection algorithm to the input image.
    """
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    grad_x = cv2.filter2D(image, cv2.CV_64F, roberts_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, roberts_y)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    gradient_magnitude *= (
            255.0 / gradient_magnitude.max()
    )  # Normalize the gradient magnitude
    return gradient_magnitude