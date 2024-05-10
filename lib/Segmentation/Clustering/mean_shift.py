import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
# from skimage.segmentation import mark_boundaries
# from skimage import io, transform


def shift_mean(img, window_size=70, convergence_threshold=1.0):
    def calculate_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    rows, cols, _ = img.shape
    segmented_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    feature_space = np.zeros((rows * cols, 5))
    counter = 0 
    current_mean_random = True
    current_mean_arr = np.zeros((1, 5))

    for i in range(rows):
        for j in range(cols):      
            feature_space[counter] = [img[i][j][0], img[i][j][1], img[i][j][2], i, j]
            counter += 1

    while len(feature_space) > 0:
        t1 = time.time()
        if current_mean_random:
            current_mean_index = random.randint(0, feature_space.shape[0] - 1)
            current_mean_arr[0] = feature_space[current_mean_index]
        
        below_threshold_arr = []
        distances = np.zeros(feature_space.shape[0])

        for i in range(len(feature_space)):
            distance = 0
            for j in range(5):
                distance += ((current_mean_arr[0][j] - feature_space[i][j]) ** 2)
                    
            distances[i] = distance ** 0.5

        below_threshold_arr = np.where(distances < window_size)[0]
        
        mean_color = np.mean(feature_space[below_threshold_arr, :3], axis=0)
        mean_pos = np.mean(feature_space[below_threshold_arr, 3:], axis=0)
        
        mean_color_distance = calculate_distance(mean_color, current_mean_arr[0][:3])
        mean_pos_distance = calculate_distance(mean_pos, current_mean_arr[0][3:])
        mean_e_distance = mean_color_distance + mean_pos_distance

        if mean_e_distance < convergence_threshold:                
            new_arr = np.zeros((1, 3))
            new_arr[0] = mean_color
            current_mean_random = True
            segmented_image[feature_space[below_threshold_arr, 3].astype(int), feature_space[below_threshold_arr, 4].astype(int)] = new_arr
            feature_space[below_threshold_arr, :] = -1
            feature_space = feature_space[feature_space[:, 0] != -1]
            
        else:
            current_mean_random = False
            current_mean_arr[0, :3] = mean_color
            current_mean_arr[0, 3:] = mean_pos
    return segmented_image

