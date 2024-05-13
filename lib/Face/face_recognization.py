import numpy as np
import os
from PIL import Image

# Reshape and resize images to 1D vectors
def reshape_images(images, size=(100, 100)):
    """
    Reshapes a list of images to a specified size.
    """
    reshaped_images = []
    for image in images:
        resized_image = image.resize(size, Image.BILINEAR)
        reshaped_image = np.array(resized_image).flatten()
        reshaped_images.append(reshaped_image)
    return reshaped_images

def construct_data_matrix(images):
    """
    Constructs a data matrix from a list of images.
    """
    data_matrix = np.vstack(images)
    return data_matrix.T

def get_mean_image(data_matrix):
    """
    Computes the mean image from a data matrix.
    """
    mean_image = np.mean(data_matrix, axis=0)
    mean_image = mean_image.reshape(1, -1)  # Reshape mean_image to (1, 10000)
    print("Shape of mean_image after reshaping:", mean_image.shape)
    return mean_image

def subtract_mean_image(data_matrix, mean_image):
    """
    Subtracts the mean image from a data matrix.
    """
    print("Shape of mean_image:", mean_image.shape)
    print("Shape of data_matrix:", data_matrix.shape)
    subtracted_images = data_matrix - mean_image
    return subtracted_images

def perform_pca(data_matrix, num_components):
    """
    Performs Principal Component Analysis (PCA) on a data matrix.
    """
    # Compute covariance matrix directly
    covariance_matrix = np.dot(data_matrix.T, data_matrix) / data_matrix.shape[1]
    # Perform eigen decomposition
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
    # Select top k eigenvectors
    selected_eigen_vectors = sorted_eigen_vectors[:, :num_components]
    # Project data onto selected eigenvectors
    projected_data = np.dot(data_matrix, selected_eigen_vectors)
    
    return projected_data, selected_eigen_vectors

def map_to_components(subtracted_images, selected_eigen_vectors):
    """
    Maps subtracted images to new components.
    """
    components = np.dot(subtracted_images, selected_eigen_vectors)
    return components

def recognize_face(test_image, mean_image, selected_eigen_vectors, threshold=4500):
    """
    Recognizes a facein a given test image.
    """
    print("Shape of selected_eigen_vectors:", selected_eigen_vectors.shape)
    
    # Flatten and reshape test image
    reshaped_test_image = np.array(test_image).flatten()
    print("Shape of reshaped_test_image:", reshaped_test_image.shape)
    # Ensure mean image has the same shape as the test image
    mean_image = mean_image.reshape(-1)  # Flatten mean image
    # Subtract mean image from test image
    subtracted_test_image = reshaped_test_image - mean_image
    # Project test image onto selected eigenvectors
    test_component = np.dot(subtracted_test_image, selected_eigen_vectors)
    # Calculate distances and find minimum distance index
    distances = np.linalg.norm(mapped_components - test_component, axis=1)
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]
    
    # Check if minimum distance meets threshold
    if min_distance < threshold:
        return min_distance_index
    else:
        return -1

# Load training images
training_path = r"C:\Users\maria\Downloads\dataset_gif\train"
training_images = []
training_labels = []
for person_folder in os.listdir(training_path):
    person_folder_path = os.path.join(training_path, person_folder)
    if not os.path.isdir(person_folder_path):
        continue
    for image_file in os.listdir(person_folder_path):
        image_path = os.path.join(person_folder_path, image_file)
        if not os.path.isfile(image_path):
            continue
        try:
            image = Image.open(image_path)
            training_images.append(image)
            training_labels.append(person_folder)
        except (IOError, OSError) as e:
            print(f"Error loading image: {image_path}")
            print(f"Error message: {str(e)}")

# Check if training images are empty
if len(training_images) == 0:
    print("No training images found.")
else:
    print(f"Number of training images: {len(training_images)}")

# Reshape and resize images
reshaped_training_images = reshape_images(training_images)
# Convert reshaped training images to a numpy array
data_matrix = np.array(reshaped_training_images)
# Calculate mean image
mean_image = get_mean_image(data_matrix)
# Subtract the mean image from all images
subtracted_images = subtract_mean_image(data_matrix, mean_image)
# Perform PCA to reduce dimensionality
num_components = 23
projected_data, selected_eigen_vectors = perform_pca(subtracted_images, num_components)
# Map images to new components
mapped_components = map_to_components(subtracted_images, selected_eigen_vectors)

def find_faces(test_image_path):
    test_image = Image.open(test_image_path).convert("L")  # Convert to grayscale
    resized_test_image = test_image.resize((100, 100), Image.BILINEAR)
    recognized_index = recognize_face(resized_test_image, mean_image, selected_eigen_vectors)

    detected_label = []
    detected_images = []

    if recognized_index != -1:
        detected_label = training_labels[recognized_index]
        print("Recognized Label:", detected_label)
        # Load all training images for the recognized label
        detected_images = [training_images[i] for i, label in enumerate(training_labels) if label == detected_label]

    return detected_label, detected_images
