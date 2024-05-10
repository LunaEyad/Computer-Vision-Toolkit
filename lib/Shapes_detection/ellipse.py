import numpy as np


def ellipse_detection( image_edges, threshold, step_angle):
    height, width = image_edges.shape[:2]
    max_axis = min(height, width) // 2  # Set max_axis based on image dimensions

    # Create accumulator array
    accumulator = np.zeros((height, width, max_axis, max_axis), dtype=np.uint64)

    # Precompute cosine and sine values for all angles
    angles = np.deg2rad(np.arange(0, 360, step_angle))  # Use step_angle
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Get indices of edge pixels
    y_idxs, x_idxs = np.nonzero(image_edges)

    # Iterate over edge pixels and radii
    for x, y in zip(x_idxs, y_idxs):
        for axis_a in range(1, max_axis):  # Exclude axis 0 as it's meaningless
            for axis_b in range(1, max_axis):  # Exclude axis 0 as it's meaningless
                # Calculate (x - a)^2 / a^2 + (y - b)^2 / b^2 = 1 for all angles
                a_values = np.round(x - axis_a * cos_angles).astype(int)
                b_values = np.round(y - axis_b * sin_angles).astype(int)

                # Filter valid indices
                valid_indices = (a_values >= 0) & (a_values < width) & (b_values >= 0) & (b_values < height)
                a_values = a_values[valid_indices]
                b_values = b_values[valid_indices]

                # Increment accumulator for valid indices
                accumulator[b_values, a_values, axis_a - 1, axis_b - 1] += 1

    # Find ellipses with accumulator values above threshold
    ellipse_indices = np.where(accumulator >= threshold)
    ellipses = [(x, y, axis_a, axis_b) for y, x, axis_a, axis_b in zip(*ellipse_indices)]

    return ellipses