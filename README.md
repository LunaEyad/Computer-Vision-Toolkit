# Computer Vision Tools

This project is a comprehensive graphical user interface (GUI) application designed for a variety of computer vision tasks. Whether you're interested in edge detection, shape recognition, face detection, or image segmentation, this tool provides a user-friendly interface that harnesses the power of advanced image processing techniques. Built using PyQt and powered by the robust OpenCV library.

## Tasks Implemented:

### Tab 1: Image Processing

1. **Noise Addition**:
   - Add Uniform, Gaussian, and Salt & Pepper noise to images.
2. **Filtering**:
   - Apply Low Pass Filters:
     - Average filter
     - Gaussian filter
     - Median filter
3. **Edge Detection**:
   - Perform edge detection using the following masks:
     - Sobel
     - Roberts
     - Prewitt
     - Canny

![Canny Edge Detection](https://github.com/LunaEyad/CV_task2/assets/103345380/17603cff-63c0-4d20-ac5d-d9fda30dc683)

4. **Frequency Domain Filters**:
   - Apply high pass and low pass frequency domain filters.

   ![Frequency Domain Filters 1](https://github.com/LunaEyad/CV_task1/assets/103345380/2a8c5353-7d1a-4431-a873-dc1e5dbc5215)
   ![Frequency Domain Filters 2](https://github.com/LunaEyad/CV_task1/assets/103345380/07679645-3250-4315-b76d-028b2de752e8)

### Tab 2: Image Analysis

1. **Histogram and Distribution Curve**:
   - Generate and display histograms and distribution curves for images.
2. **Equalization**:
   - Apply image equalization.
3. **Normalization**:
   - Perform image normalization.
4. **Thresholding**:
   - Apply local and global thresholding techniques.
5. **Color Image Transformation**:
   - Convert color images to grayscale and plot R, G, and B histograms along with their distribution functions.

   ![Color Image Transformation 1](https://github.com/LunaEyad/CV_task1/assets/103345380/109f49e0-e750-47b2-8e96-6085d8137663)
   ![Color Image Transformation 2](https://github.com/LunaEyad/CV_task1/assets/103345380/88733d28-33d8-4134-9272-a1fb7d0a2c30)

### Tab 3: Hybrid Images

- **Hybrid Images**:
  - Create hybrid images by combining low and high-frequency content from different images.

   ![Hybrid Images](https://github.com/LunaEyad/CV_task1/assets/103345380/136fcd4c-5192-459f-b156-4bcc78db7c72)

### Tab 4: Hough Transform

- **Shape Detection**:
  - Detect lines, circles, and ellipses in images.
  - Superimpose the detected shapes onto the images for visualization.

   ![Shape Detection 1](https://github.com/LunaEyad/CV_task2/assets/103345380/4e64fe64-fd27-4ea6-a55a-bdf6d8592ff5)
   ![Shape Detection 2](https://github.com/LunaEyad/CV_task2/assets/103345380/5195ec53-fad2-4673-b459-5e7fddcaef2c)

### Tab 5: Active Contour

- **Active Contour Model (Snake)**:
  - Calculate the perimeter and area enclosed by the evolved contours.

   ![Active Contour](https://github.com/LunaEyad/CV_task2/assets/55236680/6efd5683-caac-46f8-8a68-b8b4417847ae)

### Tab 6: Harris Operator

- **Harris Corner Detection**:
  - Detect corners in images using the Harris corner detection algorithm.
  - Visualize the detected corners by overlaying them on the original image.
    ![image](https://github.com/user-attachments/assets/122fbdc5-4eed-4b59-a533-02b2205275ea)


### Tab 7: Feature Matching

- **Feature Matching**:
  - Detect and match features between two images using algorithms : SSD and NCC
  - Visualize the matched features by drawing lines connecting corresponding points between the images.
![NCC_matching](https://github.com/user-attachments/assets/98388406-d3e8-4eba-b78d-b496595d2308)

### Tab 8: Segmentation

- **Image Segmentation**:
  - Segment images into different regions using techniques like thresholding, region growing, or clustering.
  - Display the segmented regions with distinct colors or boundaries.
    ![image](https://github.com/user-attachments/assets/5d1e8d8e-99bc-4a82-8df1-0107f7a3db2e)


### Tab 9: Face Detection

- **Face Detection**:
  - Detect faces in images using algorithms like Haar Cascades.
  - Draw bounding boxes around detected faces to visualize their locations.
![image](https://github.com/user-attachments/assets/97ab49a7-59a2-4ec9-b3c0-533fa2aad925)


### Tab 10: Face Recognition

- **Face Recognition**:
  - Recognize and identify faces in images using machine learning models or pre-trained neural networks.
  - Display the names or labels of recognized individuals on the images.
![image](https://github.com/user-attachments/assets/1ecb245f-33a4-408b-a003-484f9c653bd3)
![image](https://github.com/user-attachments/assets/110947d1-29ce-4aa5-bc3f-3d342c589a7d)


---

## Usage

1. **Upload Image**: Select and upload the image you want to process. Choose whether it is colored or not from the "colored" checkbox.
2. **Explore your needed Feature**:
   - **Canny Edge Detection**: Choose "Canny" from the "Edge Detection Masks" combobox, specify the low and high thresholds, and click apply.
   - **Shape Detection**: Navigate to the "Hough Transform" tab, select the desired shape from the combobox, specify the threshold, and click apply.
   - **Contouring an Object**: Open the "Active Contour" tab, set the parameters as needed, and click apply.
   - **Corner Detection**: Use the "Harris Operator" tab to detect and visualize corners in your image.
   - **Feature Matching**: Use the "Features Matching" tab to detect and visualize matched features between two images.
   - **Segmentation**: Use the "Segmentation" tab to segment your image and visualize the regions.
   - **Face Detection**: Use the "Face Detection" tab to detect faces in an image and visualize the bounding boxes.
   - **Face Recognition**: Use the "Face Recognition" tab to identify and label faces in an image.
     
## Requirements

- Python 3.x
- Required Python libraries:
  - NumPy
  - OpenCV
  - Matplotlib
  - PyQt5

## Files

- `main.py`: Main script containing the implementation of the tasks.
- `sample_images/`: Directory containing standard grayscale and color images for processing.
- `ui_files/`: Contains the user interface design files.
- `lib/`: Contains all the algorithms implemented files.

## Contributors

- [Basmalah Tarek](https://github.com/BasT13c)
- [Luna Eyad](https://github.com/LunaEyad)
- [Mariam Hatem](https://github.com/Mariam-Hatem)
- [Malak Naser](https://github.com/malaknasser812)
- [Hadeer Faseh](https://github.com/hadeerfasih)

