import numpy as np


def apply_median_channel( channel: np.ndarray) -> np.ndarray:
    """
    Applies the median filter to a single channel.
    """
    output = np.zeros_like(channel)

    # Deal with filter size = 3x3
    for j in range(1, channel.shape[0] - 1):
        for i in range(1, channel.shape[1] - 1):
            # Extract pixel values from the neighborhood
            neighborhood = [
                channel[j - 1, i - 1],
                channel[j, i - 1],
                channel[j + 1, i - 1],
                channel[j - 1, i],
                channel[j, i],
                channel[j + 1, i],
                channel[j - 1, i + 1],
                channel[j, i + 1],
                channel[j + 1, i + 1],
            ]

            # Assign the median value to the output pixel
            output[j, i] = np.median(neighborhood)
    return output
