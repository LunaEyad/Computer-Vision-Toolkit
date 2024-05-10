from numpy import all, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, deg2rad, rad2deg, where, zeros, floor, round, float32
from numpy.linalg import det, lstsq, norm
from functools import cmp_to_key
import cv2

#used for floating-point comparison 
float_tolerance = 1e-7

# Main function
def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    #convert input image to float32 data type; operations on image require floating point precision
    image = image.astype('float32')
    # generate base image by upscaling and applying a Gaussian blur to input image
    base_image = generateBaseImage(image, sigma, assumed_blur)
    # calculate number of octaves in the scale space based on the shape of the base image
    num_octaves = computeNumberOfOctaves(base_image.shape)
    # generate Gaussian kernels that will be used to create the scale space
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    # generate Gaussian images for each octave in the scale space.
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    # generate the Difference of Gaussian (DoG) images - used to find keypoints in the image
    dog_images = generateDoGImages(gaussian_images)
    # find all potential keypoints in the scale space
    keypoints_of_all_extrema = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    # remove duplicate keypoints
    keypoints = removeDuplicateKeypoints(keypoints_of_all_extrema)
    # scale keypoints back to the size of the input image
    keypoints = convertKeypointsToInputImageSize(keypoints)
    # generate SIFT descriptors for each keypoint
    descriptors = generateDescriptors(keypoints, gaussian_images)
    
    return keypoints, descriptors


def generateBaseImage(image, sigma, assumed_blur):
    """Generate base image from input image by upsampling by a factor of 2 in both x and y directions and blurring

    parameters:
        image: input image array
        sigma: desired level of blur
        assumed_blur: level of blur assumed to already be present in the image
    """
    # upsample the input image to avoid loss of information (increase resolution of image to detect small features)
    # -> helps to create scale invariance if reference image is scaled up
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR) # bilinear interpolation used for upsampling
    # calculate difference in the standard deviation of the Gaussian blur to be applied to 
    # blurring is additive -> apply sigma_diff to assumed_blur to achieve image with blur equal sigma
    # Pythagorean-like equation used because variances (square of standard deviations) of independent random variables are additive
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01)) # max function ensures that value inside square root is not negative
    # apply Gaussian blur with blur sigma_diff to the upsampled image & return blurred image as base image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff) 

def computeNumberOfOctaves(image_shape) -> int:
    """Compute number of octaves based on the shape of the base image
    
    returns:
        int: No. of octaves
    """
    # log(min(image_shape)) / log(2) - 1: calculates number of times image can be halved (downsampled) until the smallest dimension of the image is less than or equal to 1. 
    #                                     done by taking the logarithm base 2 of the smallest dimension of the image and subtracting 1.
    # int(round(...)): result is then rounded to the nearest integer -> the number of octaves must be an integer
    return int(round(log(min(image_shape)) / log(2)))

def generateGaussianKernels(sigma, num_intervals):
    """Generate list of gaussian kernels to blur the input image at different scales.
    
    parameters:
        num_intervals: specifies the number of intervals within each octave of the scale-space pyramid. 
                        Each interval corresponds to a different amount of blur applied to the image.
    """
    # calculate total number of images per octave - a series of images in scale-space pyramid where each image is a progressively blurred version of the base image
    num_images_per_octave = num_intervals + 3
    # calculates constant factor k - used to progressively increase the scale (or level of blur) at each interval within an octave
    k = 2 ** (1. / num_intervals)
    # initialize an array of zeros that will hold the standard deviations of the Gaussian kernels
    gaussian_kernels = zeros(num_images_per_octave)  
    # set standard deviation of the first Gaussian kernel to sigma
    gaussian_kernels[0] = sigma

    # calculate standard deviation of the Gaussian kernel for each image in the octave, starting from second kernel
    for image_index in range(1, (num_images_per_octave)):
        # amount of blur  applied to get to the previous image
        sigma_previous = (k ** (image_index - 1)) * sigma
        # calculate total amount of blur that should be applied to base image to get to this image in the octave
        sigma_total = k * sigma_previous
        # store value of additional amount of blur that needs to be applied to the previous image to get to the current image
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """Generate scale-space pyramid of Gaussian images
    """
    # list of lists of Gaussian images per octave
    gaussian_images = []

    # outer loop: iterate over each octave
    for _ in range(num_octaves):
        gaussian_images_in_octave = [] # initialize list to hold Gaussian images for current octave
        gaussian_images_in_octave.append(image)  # first image (base image) in octave already has the correct blur
        
        # inner loop: iterate over each Gaussian kernel, starting from the second one - Each Gaussian kernel represents a different level of blur.
        for gaussian_kernel in gaussian_kernels[1:]:
            # apply Gaussian blur to the image using the current Gaussian kernel
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            # add blurred image to the list of Gaussian images for the current octave
            gaussian_images_in_octave.append(image)
        
        # add list of Gaussian images for the current octave to the overall list of Gaussian images
        gaussian_images.append(gaussian_images_in_octave)
        
        # select the third last image in the current octave as the base image for the next octave
        # based on empirical data from David Lowe's SIFT paper
        octave_base = gaussian_images_in_octave[-3]
        
        # downsample the octave base image for the next octave by a factor of 2 in both the x and y directions
        # necessary to scale-invariance (detect features at scaled down images with different scales)
        if octave_base.shape[0] > 1 and octave_base.shape[1] > 1:
            image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
    
    # return converted list of Gaussian images to a numpy array
    return array(gaussian_images, dtype=object)

def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussian (DoG) image pyramid
    """
    # list of lists of DoG images per octave
    dog_images = []

    # outer loop: iterate over each octave of Gaussian images
    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = [] # initialize an empty list to hold DoG images for the current octave
        
        # inner loop: iterate over each pair of consecutive Gaussian images in the current octave.
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            # calculate the difference between each pair of consecutive Gaussian images 
            # and add resulting DoG image to the list of DoG images for the current octave
            dog_images_in_octave.append(cv2.subtract(second_image, first_image))  # subtract function is an opencv function that performs element-wise subtraction between two arrays (i.e. images)
        
        # add list of DoG images for the current octave to the overall list of DoG images.
        dog_images.append(dog_images_in_octave)
        
    # return converted list of DoG images to a numpy array
    return array(dog_images, dtype=object)

# Scale-space extrema related
def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    # calculate a threshold value that is used to filter out extrema that have low contrast
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = [] # initialize empty list to hold the keypoints

    # outer 2 loops: iterate over each DoG image in each octave
    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # inner 2 loops: iterate over each pixel in the current DoG image, excluding a border around the image that is "image_border_width" pixels wide
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    # check if the current pixel at (i,j) is an extremum (either a local maximum or minimum) by comparing it to its 26 neighbors in 3D space (9 in the previous scale, 8 in the current scale, and 9 in the next scale)
                    # each image has a 3x3 patch around the patch center pixel location (i,j)
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        # If the current pixel is an extremum, the localizeExtremumViaQuadraticFit function is called to perform subpixel localization of the extremum
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        # If this function returns a result, 
                        if localization_result is not None:
                            # the result is a keypoint and its localized image index
                            keypoint, localized_image_index = localization_result
                            # compute one or more keypoints with orientations for each image pixel
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            
                            # add computed keypoints to the list of all keypoints
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    
    # return list of ALL keypoints
    return keypoints

def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
    """
    # get value of the center pixel in the 3x3 patch of the second image (the image at the current scale)
    center_pixel_value = second_subimage[1, 1]
    
    # check if absolute value of the center pixel is greater than a certain threshold (calculated in findScaleSpaceExtrema method)
    # A preliminary check to filter out pixels with low contrast
    if abs(center_pixel_value) > threshold:
        
        # check of center pixel is local maximum or minimum
        # by comparing the center pixel value to the values of its neighbors in the 3x3x3 neighborhood in the scale space 
        # This neighborhood includes 8 neighbors in the same image (second_subimage), 9 neighbors in the previous scale (first_subimage), and 9 neighbors in the next scale (third_subimage)
        if center_pixel_value > 0:
            # "all" function returns True if all values in the input array are True. 
            # If the center pixel’s value is greater than or equal to all its neighbors’ values, the function returns True, indicating that the pixel is a local maximum
            # & all pixel values in the same image (second_subimage) are also compared for each row of the 3x3 patch  
            return all(center_pixel_value >= first_subimage) and \
                   all(center_pixel_value >= third_subimage) and \
                   all(center_pixel_value >= second_subimage[0, :]) and \
                   all(center_pixel_value >= second_subimage[2, :]) and \
                   center_pixel_value >= second_subimage[1, 0] and \
                   center_pixel_value >= second_subimage[1, 2]
        elif center_pixel_value < 0:
            # If the center pixel’s value is less than or equal to all its neighbors’ values, the function returns True, indicating that the pixel is a local minimum
            return all(center_pixel_value <= first_subimage) and \
                   all(center_pixel_value <= third_subimage) and \
                   all(center_pixel_value <= second_subimage[0, :]) and \
                   all(center_pixel_value <= second_subimage[2, :]) and \
                   center_pixel_value <= second_subimage[1, 0] and \
                   center_pixel_value <= second_subimage[1, 2]
    return False

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    # 
    extremum_is_outside_image = False
    # 
    image_shape = dog_images_in_octave[0].shape
    
    # refine the position of an extremum up to a certain number of times
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        # create 3D array (pixel_cube) containing the values of the 3x3x3 neighborhood around the pixel in the DoG images - pixel values are converted to float32 and scaled to [0, 1]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        # compute the gradient and Hessian matrix at the center pixel of the pixel_cube
        # gradient: gives the direction of the steepest ascent - used to update the position of the keypoint
        #           negative of the gradient gives the direction in which the function (the DoG) increases the most
        # hessian: second-order derivative of a function and it gives information about the curvature of the function. 
        #           the Hessian matrix is used to estimate the 3D quadratic function around the keypoint which is then used to find the extremum (maximum or minimum)
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        # solves the linear system Hx = -g to get the update to the extremum’s position (extremum_update), where 
        # H : Hessian matrix and 
        # g : gradient
        # The solution to this system, -H^(-1)g, gives the amount by which the keypoint’s position needs to be updated for it to be at the extremum
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        
        # update the position of the extremum in the x, y, and scale dimensions
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        
        # check if the updated position of the extremum (new pixel_cube) is outside the image or scale space. If it is, the flag is set to True
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    # Updated extremum moved outside of image before reaching convergence -> function returns None 
    if extremum_is_outside_image:
        return None
    # Exceeded maximum number of attempts without reaching convergence for this extremum -> function returns None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    
    # compute the value of the DoG function at the updated position of the extremum
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    
    # check if the contrast at the extremum is above a certain threshold. If it is not, the function returns None at the end
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        # compute the trace and determinant of the 2x2 Hessian matrix in the x and y dimensions, 
        # and check if the ratio of the determinant to the square of the trace is above a certain threshold (eigenvalue_ratio)
        # -> to filter out edge-like responses
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -> construct and return OpenCV KeyPoint object
            # with updates position, scale, and response
            keypoint = cv2.KeyPoint()
            
            # position: (j, i) is the initial position of the keypoint, extremum_update is the update to the keypoint’s position found by the quadratic fit, and octave_index is the index of the octave from which the keypoint was extracted
            # The position is scaled by 2 ** octave_index to account for the doubling of the image size at each octave
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            
            # octave: index of the octave from which the keypoint was extracted, plus some additional information about the specific scale within the octave and the amount of interpolation performed by the quadratic fit
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            
            # size: represents the scale at which the keypoint was detected
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            
            # response: absolute value of the DoG function at the updated position of the keypoint
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        rotation invariance
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    # return gradient as array - gradient vector points in the direction of the greatest rate of increase of the pixel intensities in the 3x3x3 neighborhood around the center pixel
    return array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        calculate the Hessian matrix at the center pixel of a 3x3x3 array. The Hessian matrix is a square matrix of second-order partial derivatives of a function.
        rotation invariance
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    
    # Compute Second-Order Derivatives
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    
    # Compute Mixed Derivatives 
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    
    # return Hessian matrix as 3x3 array - provides complete description of the local curvature of the function at the center pixel
    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

#########################
# Keypoint orientations #
#########################

def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    keypoints_with_orientations = [] #initialize list to store gradients of each keypoint
    image_shape = gaussian_image.shape #initialize image shape with current gaussian_image

    # compute scale at which the keypoint was detected and the radius of the region around the keypoint that will be used to compute the keypoint’s orientation
    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    # compute histogram of gradient orientations within the region around the keypoint. 
    # This is done by iterating over each pixel in the region, computing the gradient at that pixel, and then adding the gradient’s magnitude to the appropriate bin in the histogram. 
    # The bin is determined by the gradient’s orientation.
    for i in range(-radius, radius + 1):
        # calculate the y-coordinate of the current point in the region around the keypoint
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    # calculate weight using gaussian
                    weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    # he raw histogram is then smoothed using a weighted average of each bin and its two neighbors on either side
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    
    # find the peaks in the smoothed histogram. These peaks correspond to dominant gradient orientations within the region around the keypoint.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    
    # For each peak in the histogram, if the peak value is above a certain threshold, the function assigns the corresponding orientation to the keypoint. 
    # This is done by creating a new KeyPoint object with the same position, scale, and response as the original keypoint, but with the new orientation. 
    # If there are multiple peaks above the threshold, the function creates multiple keypoints, one for each orientation
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

##############################
# Duplicate keypoint removal #
##############################

def compareKeypoints(keypoint1, keypoint2):
    """Return True if keypoint1 is less than keypoint2
    """
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    """Sort keypoints and remove duplicate keypoints
    """
    if len(keypoints) < 2:
        return keypoints

    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]

    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

#############################
# Keypoint scale conversion #
#############################

def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size - maps keypoint onto the image space
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

#########################
# Descriptor generation #
#########################

def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint using histogram binning 
    This results in an orientation vector that acts as a descriptor for each keypoint
    """
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = round(scale * array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = cos(deg2rad(angle))
        sin_angle = sin(deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = rad2deg(arctan2(dy, dx)) % 360
                        weight = exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            # Smoothing via trilinear interpolation
            # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), float_tolerance)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')

