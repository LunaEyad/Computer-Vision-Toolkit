import numpy as np
import matplotlib.pyplot as plt
from lib.Filters.gaussian_filter import apply_gaussian_channel
from lib.Edge_detection.sobel import apply_sobel


def display(img, title):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(title)
    plt.show()


def apply_canny(image, low_threshold, high_threshold):
    # Step 1: Convert image to grayscale : done already when reading the image

    # Step 2: Apply Gaussian blur to reduce noise
    blurred_image = apply_gaussian_channel(image)
    # display(blurred_image, "Gaussian Blur")  # Display Gaussian blur result

    # Step 3: Compute gradients using Sobel operator
    # Step 4: Compute gradient magnitude and direction
    gradient_magnitude, gradient_direction = apply_sobel(blurred_image)
    # display(gradient_magnitude, "Gradient Magnitude")  # Display gradient magnitude result

    # Step 5: Non-maximum suppression

    # Iterate over each pixel in the gradient magnitude image to perform suppression.
    height, width = gradient_magnitude.shape
    suppressed_image = np.zeros_like(gradient_magnitude)
    for i in range(1, height - 1):
        for j in range(1, width - 1):

            # Calculate the angle of the gradient direction at the current pixel.
            angle = gradient_direction[i, j] * (180 / np.pi)

            # Based on the angle, apply different suppression rules to preserve only the local maximum gradient values along the edges.

            # For angles close to vertical (0°, 180°) or horizontal (90°), compare the current pixel's gradient magnitude with its neighbors along the same orientation.
            if (
                    (0 <= angle < 22.5)
                    or (157.5 <= angle <= 180)
                    or (-22.5 <= angle < 0)
                    or (-180 <= angle < -157.5)
            ):
                if (gradient_magnitude[i, j] > gradient_magnitude[i, j + 1]) and (
                        gradient_magnitude[i, j] > gradient_magnitude[i, j - 1]
                ):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

            # For angles at approximately 45° or -135°, compare the current pixel's gradient magnitude with its neighbors along the diagonal.
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j + 1]) and (
                        gradient_magnitude[i, j] > gradient_magnitude[i - 1, j - 1]
                ):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

            # For angles close to horizontal (90°) or vertical (0°), compare the current pixel's gradient magnitude with its neighbors along the same orientation.
            elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j]) and (
                        gradient_magnitude[i, j] > gradient_magnitude[i - 1, j]
                ):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

            # For angles at approximately 135° or -45°, compare the current pixel's gradient magnitude with its neighbors along the diagonal.
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                if (gradient_magnitude[i, j] > gradient_magnitude[i + 1, j - 1]) and (
                        gradient_magnitude[i, j] > gradient_magnitude[i - 1, j + 1]
                ):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

    # display(suppressed_image, "Non-maximum Suppression")  # Display suppressed image result

    # Step 6: Double threshold
    low_threshold_value = np.max(suppressed_image) * low_threshold
    high_threshold_value = np.max(suppressed_image) * high_threshold

    strong_edges = suppressed_image > high_threshold_value
    weak_edges = (suppressed_image >= low_threshold_value) & (
            suppressed_image <= high_threshold_value
    )

    # Step 7: Edge tracking by hysteresis
    edges = np.zeros_like(suppressed_image)
    edges[strong_edges] = 255

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if weak_edges[i, j]:
                if (edges[i - 1: i + 2, j - 1: j + 2] > high_threshold_value).any():
                    edges[i, j] = 255

    # display(edges, "Final Edges")  # Display final edges
    print("done")

    return edges
