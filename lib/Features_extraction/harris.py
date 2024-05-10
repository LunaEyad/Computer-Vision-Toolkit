import cv2
import numpy as np
import time


def harris(image, harris_flag, window_size, k, threshold, colored_flag):
    # Make a copy of the input image to draw circles on
    output_image = image.copy()

    # Retrieve height, width, and number of channels from the image
    if colored_flag:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    # Define the padding for window size (assuming square window)
    # ex: 3*3 window so padding will be 1 ,as we start from the 2nd R AND col
    padding = window_size // 2

    # Initialize a matrix to store Harris response values for each pixel
    matrix_R = np.zeros((height, width))

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Convert the blurred image to grayscale
    if colored_flag:
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    # Compute gradients using Sobel operator
    dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    print("start")
    start_time = time.time()  # Start timing

    # Iterate through each pixel in the image
    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            # Compute the sums of squares and cross-products of derivatives in the window
            Ix2 = np.sum(
                np.square(
                    dx[y - padding : y + 1 + padding, x - padding : x + 1 + padding]
                )
            )
            Iy2 = np.sum(
                np.square(
                    dy[y - padding : y + 1 + padding, x - padding : x + 1 + padding]
                )
            )
            Ixy = np.sum(
                dx[y - padding : y + 1 + padding, x - padding : x + 1 + padding]
                * dy[y - padding : y + 1 + padding, x - padding : x + 1 + padding]
            )
            # This is H Matrix
            # [ Ix2        Ixy ]
            # [ Ixy     Iy2    ]

            # Construct the structure matrix M
            M = np.array([[Ix2, Ixy], [Ixy, Iy2]])

            if harris_flag:
                # Compute the Harris corner response function
                det_M = np.linalg.det(M)
                trace_M = np.trace(M)
                R = det_M - k * (trace_M**2)
            else:
                # Use λ-based response function
                # This ratio is indicative of the local image structure, with higher values suggesting corner-like structures
                R = np.linalg.det(M) / np.trace(M)

            # Store the computed response value in the matrix_R
            matrix_R[y - padding, x - padding] = R

    # Apply a threshold and draw circles around detected corners
    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            value = matrix_R[y - padding, x - padding]
            if value > threshold:
                cv2.circle(output_image, (x, y), 1, (0, 255, 0), -1)

    end_time = time.time()  # End timing
    elapsed_time_seconds = end_time - start_time
    elapsed_time_microseconds = elapsed_time_seconds * 1e6  # Convert to microseconds

    print("Time taken:", elapsed_time_seconds, "seconds")
    print("Time taken:", elapsed_time_microseconds, "microseconds")

    print("done")
    # Return the image with drawn circles around corners
    return output_image


# image_path = "/Users/lunaeyad/PycharmProjects/CV_task3/Images/cow_step_harris.png"

# input_image = cv2.imread(image_path)

# harris_flag = True
# window_size = 3
# k = 0.04
# threshold = 1000000000


# output_image = harris(input_image, harris_flag, window_size, k, threshold)
# cv2.imshow('Harris Corner Detection', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
