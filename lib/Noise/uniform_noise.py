import numpy as np


def uniform_noise(original_img, colored=False):
    """
    Applies uniform noise to the original image and displays the resulting noisy image.

    Parameters:
        None (reads the original image from self.read_img())

    Returns:
        None (displays the noisy image in the specified widget)
    """

    # Get the original image
    # original_img = self.read_img()

    # Find the minimum and maximum values
    min_vals = np.min(original_img, axis=(0, 1))
    max_vals = np.max(original_img, axis=(0, 1))

    # Normalize the image
    original_img = (original_img - min_vals) / (max_vals - min_vals)
    # Create uniform noise within the specified range
    noisy_img = np.copy(original_img)
    a = 0  # the minimum number of the noise
    b = 0.2  # the maximum number of the noise

    # Generate uniform noise for each pixel
    if colored:
        x, y, _ = original_img.shape
        noise = np.random.uniform(a, b, size=(x, y, 3))
        # Add the noise to the original image
        noisy_img = original_img + noise

        # Clip the resulting image to the valid range [0, 1]
        noisy_img = np.clip(noisy_img, 0, 1)

        # Display the resulting noisy image in the specified UI widget
        # widget = self.ui.widget_noised
        # self.label_for_colored_image(noisy_img, widget)
    else:
        x, y = original_img.shape
        noise = np.random.uniform(a, b, size=(x, y))

        # Add the noise to the original image
        noisy_img = original_img + noise

        # Clip the resulting image to the valid range [0, 1]
        noisy_img = np.clip(noisy_img, 0, 1)

        # Display the resulting noisy image in the specified UI widget
        # widget = self.ui.widget_noised
        # self.label_for_image(self.noisy_img, widget)
    return noisy_img
