import cv2
import numpy as np
import timeit

def detect_and_display_faces(image_data, rect_thickness, scale_factor, min_window_size):
    """
    Detect faces in an image, draw rectangles around them, and display the image with the rectangles drawn.

    :param image_data: Input image data
    :param rect_thickness: Thickness of the rectangle to be drawn around the detected faces
    :param scale_factor: Scale factor for face detection
    :param min_window_size: Minimum window size for face detection
    """
    # Convert image data to RGB : done when display image
    image_rgb = image_data

    # Ensure scale_factor is greater than 1.0
    scale_factor = max(scale_factor, 1.2)

    # Calculate function run time
    start_time = timeit.default_timer()

    # Detect faces in the image
    #  The Haar cascade classifier for face detection is loaded using cv2.CascadeClassifier.
    #  Then, detectMultiScale function is used to detect faces in the image.
    #  This function returns a list of rectangles, each representing a detected face.(list of lists)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_data, scaleFactor=scale_factor, minNeighbors=5,
                                           minSize=(min_window_size, min_window_size))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img=image_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=rect_thickness)

    # Calculate function end time
    end_time = timeit.default_timer()

    # Show only 5 digits after the floating point
    elapsed_time = format(end_time - start_time, '.5f')

    # # Display the processed image
    # cv2.imshow('Detected Faces', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(faces)
    print(f"Found {len(faces)} Faces!")

    print(f"Elapsed Time for Face Detection: {elapsed_time} seconds")
    return image_rgb