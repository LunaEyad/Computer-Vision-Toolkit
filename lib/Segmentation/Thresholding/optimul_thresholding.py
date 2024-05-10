import numpy as np
import cv2
import matplotlib.pyplot as plt

def optimal_global_thresholding(image):
    """
    Perform global thresholding on a grayscale image using the optimal thresholding algorithm.

    Args:
        image (numpy.ndarray): Input image. If the image is in color, it will be converted to grayscale.

    Returns:
        numpy.ndarray: Thresholded image where pixels with intensities greater than the threshold are set to 255 (white),
                       and pixels with intensities less than or equal to the threshold are set to 0 (black).
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the mean of the four corners
    corner_mean = np.mean([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]])
    
    # Get the mean of the object without the four corners
    object_pixels = image[1:-1, 1:-1]
    object_mean = np.mean(object_pixels)
    
    # Calculate initial threshold
    threshold = (corner_mean + object_mean) / 2
    
    while True:
        # Segment the image based on current threshold
        object_pixels = image[image >= threshold]
        background_pixels = image[image < threshold]
        
        # Calculate mean of object and background pixels
        object_mean = np.mean(object_pixels)
        background_mean = np.mean(background_pixels)
        
        # Calculate new threshold
        new_threshold = (object_mean + background_mean) / 2
        
        # Check if threshold has converged
        if abs(new_threshold - threshold) < 1e-5:
            break
        
        # Update threshold
        threshold = new_threshold
    
    # Threshold the image
    thresholded_image = np.zeros_like(image)
    thresholded_image[image > threshold] = 255
    
    return thresholded_image


def optimal_local_thresholding(image, block_size):
    """
    Perform local thresholding on a grayscale image using the optimal thresholding algorithm.

    Args:
        image (numpy.ndarray): Input image. If the image is in color, it will be converted to grayscale.
        block_size (int, optional): Size of the square block for local thresholding. Default is 90.

    Returns:
        numpy.ndarray: Thresholded image where pixels with intensities greater than the threshold are set to 255 (white),
                       and pixels with intensities less than or equal to the threshold are set to 0 (black).
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the image dimensions
    height, width = image.shape
    
    # Create an output image with the same dimensions as the input image
    thresholded_image = np.zeros_like(image)
    
    # Iterate over the image in blocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Get the current block
            block = image[i:i+block_size, j:j+block_size]
            
            # Calculate the mean of the block
            block_mean = np.mean(block)
            
            # Calculate the initial threshold for the block
            threshold = block_mean
            
            while True:
                # Segment the block based on current threshold
                object_pixels = block[block >= threshold]
                background_pixels = block[block < threshold]
                
                # Calculate mean of object and background pixels
                object_mean = np.mean(object_pixels)
                background_mean = np.mean(background_pixels)
                
                # Calculate new threshold
                new_threshold = (object_mean + background_mean) / 2
                
                # Check if threshold has converged
                if abs(new_threshold - threshold) < 1e-5:
                    break
                
                # Update threshold
                threshold = new_threshold
            
            # Threshold the block
            thresholded_block = np.zeros_like(block)
            thresholded_block[block > threshold] = 255
            
            # Place the thresholded block into the output image
            thresholded_image[i:i+block_size, j:j+block_size] = thresholded_block
    
    return thresholded_image
