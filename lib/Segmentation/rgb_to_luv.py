import cv2
import numpy as np
from matplotlib import pyplot as plt

# Conversion matrix from RGB to XYZ
RGB_to_XYZ = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])


def RGB_to_Luv(R, G, B):

    # In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and scaled to fit 0 to 1 range.
    # Scale RGB values to the range [0, 1]
    r = R / 255.0
    g = G / 255.0
    b = B / 255.0

    # Convert RGB to XYZ
    XYZ = np.dot(RGB_to_XYZ, np.array([r, g, b]))
    XYZ_normalized = XYZ / np.array([0.950456, 1.0, 1.088754])


    # Calculate L*u*v*
    X, Y, Z = XYZ_normalized
    epsilon = 0.008856

    if Y > epsilon:
        L = (116 * np.power(Y , 1 / 3)) - 16
    else:
        L = 903.3 * Y

    u_dash= 4 * X/ (X + 15 * Y + 3 * Z)
    v_dash = 9 * Y / (X + 15 * Y+ 3 * Z)
    U= 13 * L * (( u_dash-0.19793943) )
    V= 13 * L * ((v_dash -0.46831096))

    # Scale LUV values to [0, 255] range
    L_scaled =  255/100* L
    u_scaled = (354.0 / 256.0) * (U + 134.0)
    v_scaled = (262.0 / 256.0) * (V + 140.0)

    return L_scaled, u_scaled, v_scaled






# # Load the image
# image = cv2.imread("/Users/lunaeyad/PycharmProjects/CV_task4/images/landscape.png")
# # Check the number of bits per channel
# num_bits = image.dtype.itemsize * 8
#
# print("Number of bits per channel:", num_bits)
# # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #
# # # Convert RGB to L*u*v*
# # luv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Luv)
#
# #
# # # Show the original and converted images
# # cv2.imshow("Original Image", image)
# # cv2.imshow("RGB to L*u*v* ", luv_image)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# # Get the height and width of the image
# height, width, _ = image.shape
#
# # Create a new blank image for the mapped L*u*v* values
# luv_image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Iterate through each pixel of the image
# for y in range(height):
#     for x in range(width):
#         # Get the RGB values of the current pixel
#         R, G, B = image[y, x]
#
#         # Convert RGB to L*u*v*
#         L, u, v = RGB_to_Luv(R, G, B)
#
#         # Store the L*u*v* values in the new image
#         luv_image[y, x] = [L, u, v]
#
# # Visualize the images using Matplotlib
# plt.figure(figsize=(12, 6))
#
# # Original Image
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')
#
# # Image mapped to L*u*v*
# plt.subplot(1, 2, 2)
# plt.imshow(luv_image.astype(np.uint8))
# plt.title('Mapped to L*u*v*')
# plt.axis('off')
#
#
# plt.show()