import numpy as np


def salt_pepper_noise(original_img, colored=False):
    """
    Applies salt-and-pepper noise to the original image and displays the resulting noisy image.

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
    # Create a blank image with the same shape as the original
    noisy_img = np.zeros_like(original_img, dtype=np.float32)
    # Define salt and pepper amounts
    pepper = 0.1  # 10% of pepper
    salt = 1 - pepper  # 10% of salt

    # Create salt and pepper noise in the blank image
    if colored:
        x, y, _ = original_img.shape
        for i in range(x):
            for j in range(y):
                rdn = np.random.random()
                if rdn < pepper:
                    noisy_img[i, j] = [0, 0, 0]  # Pepper noise (black)
                elif rdn > salt:
                    noisy_img[i, j] = [1, 1, 1]  # Salt noise (white)
                else:
                    noisy_img[i, j] = original_img[
                        i, j
                    ]  # Original pixel value
        # Display the resulting noisy image in the specified UI widget
        # widget = self.ui.widget_noised
        # self.label_for_colored_image(self.noisy_img, widget)

    else:
        x, y = original_img.shape
        for i in range(x):
            for j in range(y):
                rdn = np.random.random()
                if rdn < pepper:
                    noisy_img[i, j] = 0  # Pepper noise (black)
                elif rdn > salt:
                    noisy_img[i, j] = 1  # Salt noise (white)
                else:
                    noisy_img[i, j] = original_img[
                        i, j
                    ]  # Original pixel value

        # # Display the resulting noisy image in the specified UI widget
        # widget = self.ui.widget_noised
        # self.label_for_image(self.noisy_img, widget)
    return noisy_img
