import numpy as np


def apply_average_channel( image: np.ndarray) -> np.ndarray:
    """
    Applies the average filter to the input image.

    """
    # Create an empty array with the same size as the input image
    output = np.zeros_like(image)

    # Iterate over each pixel of the input image
    for j in range(1, image.shape[0] - 1):
        for i in range(1, image.shape[1] - 1):
            # Extract the 3x3 kernel centered around the pixel
            kernel = [
                image[j - 1, i - 1],
                image[j - 1, i],
                image[j - 1, i + 1],
                image[j, i - 1],
                image[j, i],
                image[j, i + 1],
                image[j + 1, i - 1],
                image[j + 1, i],
                image[j + 1, i + 1],
            ]

            # Calculate the mean value of the pixel values in the kernel
            mean_value = sum(kernel) / len(kernel)

            # Assign the mean value to the corresponding pixel in the output array
            output[j, i] = mean_value

    return output