import numpy as np


def apply_filter_low_high_pass(image, filter_type):
    # Fourier transform to convert the image to the frequency domain
    f_image = np.fft.fft2(image, axes=(0, 1))
    fshift_imag = np.fft.fftshift(f_image, axes=(0, 1))

    # Low-pass filter creation
    x, y = image.shape[:2]
    filter_matrix = np.zeros((x, y), dtype=np.float32)
    cutoff = 50

    for u in range(x):
        for v in range(y):
            r_from_center = np.sqrt((u - x / 2) ** 2 + (v - y / 2) ** 2)
            if filter_type == "Low pass":
                if r_from_center <= cutoff:
                    filter_matrix[u, v] = 1
            elif filter_type == "High pass":
                if r_from_center > cutoff:
                    filter_matrix[u, v] = 1

    # Extend the filter to match the number of color channels in the image
    if len(image.shape) == 3:
        filter_matrix = np.expand_dims(filter_matrix, axis=-1)

    # Apply the selected filter to the image in the frequency domain
    Gshift = fshift_imag * filter_matrix

    # Inverse Fourier transform to obtain the filtered image in the spatial domain
    G = np.fft.ifftshift(Gshift, axes=(0, 1))
    g = np.abs(np.fft.ifft2(G, axes=(0, 1)))
    return g