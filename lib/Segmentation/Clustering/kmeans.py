import numpy as np
import matplotlib.pyplot as plt



def euclidean_distance(x1, x2):
    """
        Calculate the Euclidean distance between two points.

        Parameters:
        x1 : numpy array
            Coordinates of the first point.
        x2 : numpy array
            Coordinates of the second point.

        Returns:
        float
            Euclidean distance between the two points.
        """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def initialize_centroids(X, K):
    """
      Initialize centroids randomly from the data points.

      Parameters:
      X : numpy array
          Input data points.
      K : int
          Number of centroids.

      Returns:
      numpy array
          Initial centroids.
      """
    #  total number of pixels in the image.
    n_samples, _ = X.shape
    print(X.shape)
    print(n_samples)
    random_sample_idxs = np.random.choice(n_samples, K, replace=False)
    centroids = [X[idx] for idx in random_sample_idxs]
    #Each centroid would represent the average color (or intensity, if grayscale) of the pixels assigned to that centroid.
    print(centroids)
    return np.array(centroids)


def create_clusters(X, centroids):
    """
       Create clusters based on the closest centroids.

       Parameters:
       X : numpy array
           Input data points.
       centroids : numpy array
           Centroids.

       Returns:
       list of lists
           List of clusters, where each cluster contains indices of data points.
       """
    # Initialize clusters as a list of empty lists, one for each centroid
    clusters = [[] for _ in range(len(centroids))]
    # Iterate over each data point
    for idx, sample in enumerate(X):
        # Find the index of the closest centroid for the current data point
        centroid_idx = closest_centroid(sample, centroids)
        # Append the index of the current data point to the cluster of the closest centroid
        clusters[centroid_idx].append(idx)
    return clusters


def closest_centroid(sample, centroids):
    """
        Find the index of the closest centroid to a given point.

        Parameters:
        sample : numpy array
            Data point.
        centroids : numpy array
            Centroids.

        Returns:
        int
            Index of the closest centroid.
        """
    # Calculate the Euclidean distance between the sample and each centroid
    distances = [euclidean_distance(sample, point) for point in centroids]
    # Find the index of the centroid with the minimum distance
    closest_index = np.argmin(distances)
    return closest_index


def get_centroids(clusters, X):
    """
    Calculate new centroids based on cluster means.

    Parameters:
    clusters : list of lists
        List of clusters, where each cluster contains indices of data points.
    X : numpy array
        Input data points.

    Returns:
    numpy array
        New centroids.
    """
    # Initialize centroids array with zeros

    centroids = np.zeros((len(clusters), X.shape[1]))
    # Iterate over each cluster

    for cluster_idx, cluster in enumerate(clusters):
        # Calculate the mean of data points in the current cluster
        cluster_mean = np.mean(X[cluster], axis=0)
        # Assign the mean as the new centroid for the current cluster
        centroids[cluster_idx] = cluster_mean
    return centroids


def is_converged(centroids_old, centroids):
    """
        Check if centroids have converged.

        Parameters:
        centroids_old : numpy array
            Previous centroids.
        centroids : numpy array
            Current centroids.

        Returns:
        bool
            True if centroids have converged, False otherwise.
        """
    # Comparing centroids directly with == might lead to issues due to floating-point precision errors.
    distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(len(centroids))]
    return sum(distances) == 0



# Function for K-means clustering
def kmeans(X, K=5, max_iters=100, plot_steps=False):
    """
     Perform K-means clustering.

     Parameters:
     X : numpy array
         Input data points.
     K : int, optional
         Number of clusters (default is 5).
     max_iters : int, optional
         Maximum number of iterations (default is 100).
     plot_steps : bool, optional
         Whether to plot intermediate steps (default is False).

     Returns:
     numpy array
         Final centroids.
     """
    # Initialize centroids randomly
    centroids = initialize_centroids(X, K)

    for _ in range(max_iters):
        # Assign samples to closest centroids (create clusters)
        clusters = create_clusters(X, centroids)
        centroids_old = centroids
        centroids = get_centroids(clusters, X)
        # Check for convergence
        if is_converged(centroids_old, centroids):
            break

    return centroids


# # Read in the image
# image = cv2.imread("/Users/lunaeyad/PycharmProjects/CV_task4/lib/Segmentation/Notebook-2.PNG")
# # Change color to RGB (from BGR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# plt.imshow(image)
# plt.show()
#
# # Reshape the image into a 2D array of pixels and 3 color values (RGB)
# pixel_vals = image.reshape((-1, 3))
# # Convert to float type
# pixel_vals = np.float32(pixel_vals)
#
# # Perform K-means clustering
# centroids = kmeans(pixel_vals, K=3, max_iters=100, plot_steps=False)
#
# # Convert centroids to uint8
# centers = np.uint8(centroids)
#
# # Assign each pixel to the closest centroid
# labels = np.array([closest_centroid(pixel, centroids) for pixel in pixel_vals])
#
# # Convert data into 8-bit values
# segmented_data = centers[labels.flatten()]
#
# # Reshape data into the original image dimensions
# segmented_image = segmented_data.reshape((image.shape))
#
# plt.imshow(segmented_image)
# # Example usage
# np.random.seed(42)
# X = np.random.rand(100, 2)  # Sample data
# clusters = kmeans(X, K=7, plot_steps=True)
# Example usage
# image_path = "/Users/lunaeyad/PycharmProjects/CV_task4/lib/Segmentation/Notebook-2.PNG"  # Replace this with the path to your image file
# segmented_image = kmeans(image_path, K=3, max_iters=100, plot_steps=True)

# # Display the segmented image
# plt.figure(figsize=(8, 8))
# plt.imshow(segmented_image)
# plt.axis('off')
# plt.title('Segmented Image')
# plt.show()
