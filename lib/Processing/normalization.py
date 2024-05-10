import numpy as np


def normalize_image( image):
    """
    Normalize the image data to the 0-255 range for each color channel
    """
    if len(image.shape) == 2:
        # Grayscale image
        min_val = np.min(image)
        max_val = np.max(image)

        if min_val == max_val:
            # Handle the case where the image is a constant (avoid division by zero)
            normalized_image = (image * 255).astype(np.uint8)
        else:
            normalized_image = (
                    (image - min_val) / (max_val - min_val) * 255
            ).astype(np.uint8)
    elif len(image.shape) == 3:
        # Colored image
        min_val = np.min(image, axis=(0, 1))
        max_val = np.max(image, axis=(0, 1))

        normalized_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[-1]):
            if min_val[i] == max_val[i]:
                # Handle the case where the image is a constant (avoid division by zero)
                normalized_image[:, :, i] = (image[:, :, i] * 255).astype(np.uint8)
            else:
                normalized_image[:, :, i] = (
                        (image[:, :, i] - min_val[i]) / (max_val[i] - min_val[i]) * 255
                ).astype(np.uint8)

    return normalized_image