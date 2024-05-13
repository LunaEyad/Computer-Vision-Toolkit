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

def get_mean_image(data_matrix):
    """
    Computes the mean image from a data matrix.
    """
    # take mean along the column
    mean_image = np.mean(data_matrix, axis=0)
    # Reshape mean_image to 1 row with same number of columns as mean_image
    mean_image = mean_image.reshape(1, -1) # 1D array
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
    Recognizes a face in a given test image.
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
        return min_distance_index, min_distance
    else:
        return -1

# Load training images
training_path = r"dataset_recognize\train"
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
            training_labels.append(person_folder) # serves as label for image
            
        except (IOError, OSError) as e:
            print(f"Error loading image: {image_path}")
            print(f"Error message: {str(e)}")
        
# Load testing images    
testing_path = r"dataset_recognize\test"           
testing_paths = []  # Initialize list to store paths of testing images
testing_labels = []
for image_file in os.listdir(testing_path):
    image_path = os.path.join(testing_path, image_file)
    if not os.path.isfile(image_path):
        continue
    try:
        testing_paths.append(image_path)  # add image path to the list
        formatted_label = image_file.split('.')[0]  # split the file name by '.' and take the first part
        testing_labels.append(formatted_label)  # serves as label for image
        # print(testing_labels[:6])
        
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

def find_faces(test_image_path, c=0):
    test_image = Image.open(test_image_path).convert("L")  # Convert to grayscale
    resized_test_image = test_image.resize((100, 100), Image.BILINEAR)
    recognized_index, min_distance = recognize_face(resized_test_image, mean_image, selected_eigen_vectors)

    detected_label = []
    detected_images = []

    if recognized_index != -1:
        detected_label = training_labels[recognized_index]
        print("Recognized Label:", detected_label)
        # Load all training images for the recognized label
        detected_images = [training_images[i] for i, label in enumerate(training_labels) if label == detected_label]

    if c:
        # return detected_label, min_distance
        return detected_label, recognized_index
    else:
        return detected_label, detected_images



def calculate_performance(threshold = 200):
    tp = 0 # true positive
    tn = 0 # true negative
    fp = 0 # false positive
    fn = 0 # false negative

    # True Positive Rate
    tpr_values = []
    # False Positive Rate
    fpr_values = []
    bestMatches = []

    for i, path in enumerate(testing_paths):
        
        train_labels, best_match = find_faces(path, c=1)
        best_match = int(best_match)
        
        bestMatches.append(best_match)
        positive = testing_labels[i] == train_labels[best_match]

        if (best_match <= threshold):
            # True Positive
            if (positive == 1) :
                print('Matched:' +  train_labels[best_match], end = '\t')
                tp +=1
            # False Positive
            elif (positive == 0):
                print('F/Matched:'+train_labels[best_match], end = '\t')
                fp +=1
        
        elif (best_match >= threshold):
            # False Negative
            if (positive == 1) :
                print('Unknown face!'+train_labels[best_match], end = '\t')
                fn +=1
            # True Negative
            elif (positive == 0):
                tn +=1
        print(tp,tn,fp,fn)

        if (tp+fn) !=0 :
            tpr = tp/(tp+fn)
        elif (tp+fn) == 0:
            tpr = 0
        
        if (fp+tn) !=0 :
            fpr = fp/(fp+tn)
        elif (fp+tn) == 0:
            fpr = 0

        tpr_values.append(tpr)
        fpr_values.append(fpr)


    num_images = tp + tn + fp +fn
        
    FMR = fp/num_images
    FNMR = fn/num_images           
    accuracy = (tp+tn) / num_images
    precision = tp/(tp+fp)
    specificity = tn/(tn+fn)


    print(bestMatches)
    print('fpr = {} \t'.format(FMR),end= ' ')
    print('accuracy = {} \t'.format(accuracy),end= ' ')
    print('precision = {} \t'.format(precision),end= ' ')
    print('specificity = {} \t'.format(specificity))


    return fpr_values, tpr_values, accuracy, precision, specificity

fpr_values, tpr_values, accuracy, precision, specificity = calculate_performance()

from matplotlib import pyplot as plt

def ROC_plot(tpr_values, fpr_values):
    fig = plt.figure(figsize=(8,6))
    plt.plot(fpr_values, tpr_values)
    plt.plot([0, 1], [0, 1], '--', color='gray')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.savefig("../images/ROC_CURVE.png")

