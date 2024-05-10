import numpy as np

def globalOtsuThresholding(img):
    """
    Performs global Otsu thresholding on the input image.

    Args:
        img (numpy.ndarray): The input grayscale image.

    Returns:
        numpy.ndarray: The thresholded image.

    """
    # Calculate the histogram of the image
    histogram = np.histogram(img.flatten(), bins=256, range=[0, 256])[0]
    # Calculate the cumulative sum of the histogram
    cumulative_sum = histogram.cumsum()
    # Calculate the total number of pixels
    total_pixels = img.size
    # Calculate the between-class variance for each possible threshold value
    between_class_variance = np.zeros(256)
    for threshold in range(256):
        if cumulative_sum[threshold] == 0 or cumulative_sum[threshold] == total_pixels:
            continue

        w0 = cumulative_sum[threshold]
        w1 = total_pixels - w0
        mean0 = np.sum(np.arange(threshold) * histogram[:threshold]) / w0
        mean1 = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / w1
        between_class_variance[threshold] = w0 * w1 * (mean0 - mean1) ** 2

    # Find the threshold that maximizes the between-class variance
    best_threshold = np.argmax(between_class_variance)
    # Apply the threshold to the image
    thresholded_img = (img > best_threshold).astype(np.uint8) * 255

    return thresholded_img

def localOtsuThresholding(img, block_size):
    """
    Performs local Otsu thresholding on the input image.

    Args:
        img (numpy.ndarray): The input grayscale image.
        block_size (int): The size of the local neighborhood for thresholding.

    Returns:
        numpy.ndarray: The thresholded image.

    """
    thresh = np.zeros(img.shape, dtype=np.uint8)

    # Iterate over the image in local neighborhoods
    for row in range(0, img.shape[0], block_size):
        for col in range(0, img.shape[1], block_size):
            # Define the region of interest (ROI) for local Otsu thresholding
            roi = img[row : row + block_size, col : col + block_size]
            local_thresh = globalOtsuThresholding(roi)
            thresh[row : row + block_size, col : col + block_size] = local_thresh

    return thresh