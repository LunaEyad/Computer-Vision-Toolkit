import cv2
import numpy as np


def apply_sobel( image: np.ndarray) -> np.ndarray:
    """
    Applies the Sobel edge detection algorithm to the input image.
    :return:
    numpy.ndarray: The edge-detected image after applying the Sobel algorithm.
    """

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)
    # kernel size and data type (cv2.CV_64F)
    # specifies that the output image should have a depth of 64-bit floating-point numbers,
    # which is suitable for gradient computations.
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_direction = np.arctan2(grad_x, grad_y)

    print("Gradient Magnitude Max bef:", gradient_magnitude.max())

    gradient_magnitude *= (
            255.0 / gradient_magnitude.max()
    )  # Normalize the gradient magnitude
    print("Gradient Magnitude Max:", gradient_magnitude.max())

    return gradient_magnitude ,gradient_direction