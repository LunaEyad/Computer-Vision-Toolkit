import timeit

from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QDialog, QButtonGroup
import pyqtgraph as pg
from typing import Optional
import numpy as np
import cv2
import sys
from UI.ui import Ui_MainWindow
from UI.histo import Ui_Form
from UI.image_widget import ImagePlotter
from lib.Noise.gaussian_noise import gaussian_noise
from lib.Noise.uniform_noise import uniform_noise
from lib.Noise.salt_and_papper_noise import salt_pepper_noise
from lib.Shapes_detection.line import line_detection
from lib.Shapes_detection.ellipse import ellipse_detection
from lib.Edge_detection.canny import apply_canny
from lib.Edge_detection.robert import apply_roberts
from lib.Edge_detection.prewitt import apply_prewitt
from lib.Edge_detection.sobel import apply_sobel
from lib.Filters.gaussian_filter import apply_gaussian_channel
from lib.Filters.mean_filter import apply_average_channel
from lib.Filters.median_filter import apply_median_channel
from lib.Filters.low_high_pass_filter import apply_filter_low_high_pass
from lib.Processing.normalization import normalize_image
from lib.Features_extraction import harris
from lib.Features_extraction.SIFT import computeKeypointsAndDescriptors
import lib.Contour.active_contour_greedy as AC
import lib.Contour.active_contour as ACC
from lib.Segmentation.Clustering.kmeans import kmeans, closest_centroid
from lib.Segmentation.Thresholding.otsu_thresholding import globalOtsuThresholding, localOtsuThresholding
from lib.Segmentation.Thresholding.spectral_thresholding import globalSpectralThresholding, localSpectralThresholding
from lib.Segmentation.Thresholding.optimul_thresholding import optimal_local_thresholding, optimal_global_thresholding
from lib.Segmentation.Clustering.kmeans import kmeans
from lib.Segmentation.Clustering.Agglomerative import apply_agglomerative
from lib.Segmentation.Clustering.region_growing import GrowRegion
from lib.Segmentation.Clustering.mean_shift import shift_mean
from lib.Segmentation.rgb_to_luv import RGB_to_Luv


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.connect_signals_slots()

        # additional UI setup
        model = self.ui.comboBox.model()
        model.item(0).setEnabled(False)  # Disable the first item (index 0)

        self.ui.lineEdit_sigma1.setText("5")
        self.ui.lineEdit_sigma2.setText("5")
        
        self.ui.lineEdit_clusters.setText("5")
        # self.ui.lineEdit_clusters.hide()
        # self.ui.label_47.hide()
        self.ui.lineEdit_threshold_clusters.setText("5")
        self.ui.lineEdit_threshold_clusters.hide()
        self.ui.label_48.hide()
        self.ui.checkBox_supervised.hide()
        
        self.buttonGroup_threshold = QButtonGroup(self)
        self.buttonGroup_threshold.addButton(self.ui.radioButton)
        self.buttonGroup_threshold.addButton(self.ui.radioButton_2)

        self.buttonGroup_cluster = QButtonGroup(self)
        self.buttonGroup_cluster.addButton(self.ui.radioButton_connect_4)
        self.buttonGroup_cluster.addButton(self.ui.radioButton_connect_8)
        
        self.ui.radioButton_connect_8.setChecked(True)
        self.ui.radioButton_connect_8.hide()
        self.ui.radioButton_connect_4.hide()

        # Instance Variables
        # self.original_img = None
        self.edited_image = None
        self.hybrid_image_path = None
        self.hybrid_img_paths = {}
        self.mix = False
        self.previous_mix_value = self.mix  # Store the initial value
        self.colored = False # Default image - Grayscale
        self.seeds = [] # stored seed list
        self.supervised = False # Default - Unsupervised
        self.connect = True # Default- 8-connected neighborhood

        # contour spin boxes deafult values (alpha beta and gamma)
        self.ui.doubleSpinBox_alpha.setValue(0.01)
        self.ui.doubleSpinBox_beta.setValue(0.4)
        self.ui.doubleSpinBox_gamma.setValue(99)

        self.ui.label_area.setText("-")
        self.ui.label_perimeter.setText("-")

        self.ui.lineEdit_canny_low.setDisabled(True)
        self.ui.lineEdit_canny_high.setDisabled(True)
        

    def connect_signals_slots(self):
        """
        Connects UI signals (e.g., combo box selections, button clicks) to corresponding slot functions.
        """
        self.ui.checkBox.stateChanged.connect(self.read_img)
        self.ui.pushButton.clicked.connect(self.apply_edge_detection)
        self.ui.comboBox_edges.activated.connect(self.line_enable)
        self.ui.comboBox_filter.activated.connect(self.apply_filter)
        self.ui.comboBox_noise.activated.connect(self.apply_noise)
        self.ui.comboBox_freq.activated.connect(self.frequency_filter)
        self.ui.comboBox.activated.connect(self.histogram_tab)
        self.ui.pushButton_upload.clicked.connect(self.browse_upload)
        self.ui.pushButton_img1.clicked.connect(self.browse_img1)
        self.ui.pushButton_img2.clicked.connect(self.browse_img2)
        self.ui.pushButton_mix.clicked.connect(self.show_current_hybrid)
        self.ui.pushButton_alternate.clicked.connect(self.toggle_mix_flag)
        self.ui.pushButton_histo.clicked.connect(self.open_histo_widget)
        self.ui.pushButton_hough.clicked.connect(self.hough_transform)
        self.ui.pushButton_apply_ac.clicked.connect(self.apply_contour)
        self.ui.pushButton_harris.clicked.connect(self.apply_harris)
        self.ui.pushButton_upload_ref.clicked.connect(self.upload_ref_img)
        self.ui.pushButton_upload_img.clicked.connect(self.upload_sec_img)
        self.ui.pushButton_ncc.clicked.connect(self.apply_NCC)
        self.ui.pushButton_ssd.clicked.connect(self.apply_SSD)
        self.ui.pushButton_clustering.clicked.connect(self.apply_clustering)
        self.ui.comboBox_clustering.activated.connect(self.hide_show_clustering)
        self.ui.pushButton_thresholding.clicked.connect(self.thresholding_methods)
        self.ui.pushButton_2.clicked.connect(self.map_to_luv)

    def hide_show_clustering(self):
        # get method from combobox
        method = self.ui.comboBox_clustering.currentText()
        if self.image_path is not None:
            # update input widget
            self.read_img()
        # Clear output widget
        label = self.ui.widget_seg_output.findChild(QLabel)
        if label is not None:
            # Remove the original label from the layout
            self.ui.verticalLayout_44.removeWidget(label)
            label.deleteLater()
            self.ui.verticalLayout_44.setStretchFactor(self.ui.widget_seg_output, 1)
            self.ui.verticalLayout_44.update()  # Update the layout
        
        if method == "Region Growing":
            self.ui.checkBox_supervised.show()
            self.ui.radioButton_connect_8.show()
            self.ui.radioButton_connect_4.show()
        else: 
            self.ui.checkBox_supervised.hide()
            self.ui.radioButton_connect_8.hide()
            self.ui.radioButton_connect_4.hide()
            
        # Apply the selected clustering method
        if method == 'K-Means':
            self.ui.lineEdit_clusters.show()
            self.ui.label_47.show()
            self.ui.lineEdit_threshold_clusters.hide()
            self.ui.label_48.hide()
            
        elif method == 'Agglomerative Clustering':
            self.ui.lineEdit_clusters.show()
            self.ui.label_47.show()
            self.ui.lineEdit_threshold_clusters.hide()
            self.ui.label_48.hide()
            
        elif method == 'Region Growing':
            self.ui.lineEdit_clusters.hide()
            self.ui.label_47.hide()
            self.ui.lineEdit_threshold_clusters.show()
            self.ui.label_48.show()
            
        elif method == 'Mean-Shift':
            self.ui.lineEdit_clusters.hide()
            self.ui.label_47.hide()
            self.ui.lineEdit_threshold_clusters.show()
            self.ui.label_48.show()
            

    ###################### Histogram dialog ######################

    def open_histo_widget(self):
        """
        Opens a histogram widget dialog and plots the original and edited histograms.

        The method initializes a dialog window, reads an image, calculates histograms based on the selected method,
        and displays the histograms in separate widgets within the dialog.
        """
        # Create a QDialog instance
        self.histo_dialog = QDialog()

        # Set up the UI of the dialog using Ui_Form
        histo_ui = Ui_Form()
        histo_ui.setupUi(self.histo_dialog)

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_widget = histo_ui.widget_1
        edited_widget = histo_ui.widget_2

        # Plot the original Histogram
        original_widget.clear()
        hist_original = cv2.calcHist(image, [0], None, [256], [0, 256])
        hist_original = hist_original.reshape(-1)
        original_widget.plot(hist_original, pen="b")

        # Plot the edited Histogram
        method = self.ui.comboBox.currentText()
        edited_widget.clear()
        if method == "Normalization":
            [n_row, n_col] = image.shape
            total_no_pixels = n_row * n_col
            equalized_image = cv2.equalizeHist(image)
            hist = cv2.calcHist(equalized_image, [0], None, [256], [0, 256])
            hist_normalized = hist / total_no_pixels
            hist_normalized = hist_normalized.reshape(-1)
            edited_widget.plot(hist_normalized, pen="b")

        elif method == "Local Threshold":
            hist_local = cv2.calcHist(self.edited_image, [0], None, [256], [0, 256])
            hist_local = hist_local.reshape(-1)
            edited_widget.plot(hist_local, pen="b")

        elif method == "Global Threshold":
            hist_global = cv2.calcHist(self.edited_image, [0], None, [256], [0, 256])
            hist_global = hist_global.reshape(-1)
            edited_widget.plot(hist_global, pen="b")
            thresh = np.mean(image)
            vertical_line = pg.InfiniteLine(pos=thresh, angle=90, pen="r")
            edited_widget.addItem(vertical_line)
            edited_widget.repaint()

        elif method == "Equalization":
            equalized_image = self.equalize()
            if len(equalized_image.shape) != 2:  # Grayscale image
                equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
            hist_equalized = cv2.calcHist(equalized_image, [0], None, [256], [0, 256])
            hist_equalized = hist_equalized.reshape(-1)
            edited_widget.plot(hist_equalized, pen="b")

        # Show the dialog
        self.histo_dialog.show()

    ###################### Browse ######################

    def browse_upload(self):
        pixmap = self.browse_and_set_image()
        self.ui.label_input_img.setPixmap(pixmap.scaledToWidth(300))
        self.show_image()
        self.show_gray_image()
        self.histogram_distribution()
        image = self.read_img()

        self.label_for_image(image, self.ui.widget_noised, flag=True)
        self.label_for_image(image, self.ui.widget_output, flag=True)

    def browse_img1(self):
        pixmap = self.browse_and_set_image(store_for_hybrid=True)
        self.hybrid_img_paths["img1"] = self.hybrid_image_path
        self.ui.label_img1.setPixmap(pixmap.scaledToWidth(300))

    def browse_img2(self):
        pixmap = self.browse_and_set_image(store_for_hybrid=True)
        self.hybrid_img_paths["img2"] = self.hybrid_image_path
        self.ui.label_img2.setPixmap(pixmap.scaledToWidth(300))

    def browse_and_set_image(self, store_for_hybrid=False):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        pixmap = None
        if file_dialog.exec():
            image_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(image_path)
            if store_for_hybrid:
                self.hybrid_image_path = image_path
            else:
                self.image_path = image_path
        return pixmap

    def read_img(self) -> Optional[np.ndarray]:
        """
        Reads the image file specified by the image path attribute.
        """
        if self.image_path:
            self.colored = self.ui.checkBox.isChecked()
            # Read the image using OpenCV
            if self.colored:
                # read image as RGB
                image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
                self.label_for_colored_image(image, self.ui.widget_hough_input)
                self.label_for_colored_image(image, self.ui.widget_active_input)
                self.label_for_colored_image(image, self.ui.widget_harris_input)
                self.label_for_colored_image(image, self.ui.widget_seg_input)

            else:
                # read image as Grayscale
                image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                self.label_for_image(image, self.ui.widget_hough_input)
                self.label_for_image(image, self.ui.widget_active_input)
                self.label_for_image(image, self.ui.widget_harris_input)
                self.label_for_image(image, self.ui.widget_seg_input)

            return image

    ###################### Hybrid ######################

    def toggle_mix_flag(self) -> None:
        """
        Toggles the mix flag and updates the hybrid image display to an alternate hybrid image combination.
        """
        # Store the previous value of the flag
        self.previous_mix_value = self.mix

        # Update the hybrid image display
        self.create_hybrid_image()

        # Toggle the flag
        self.mix = not self.mix

        print("Alternate")

    def show_current_hybrid(self) -> None:
        """
        Displays an updated version of the current hybrid image combination based on the previous mix flag value.
        """
        # Set flag to the previous value
        self.mix = self.previous_mix_value

        # Update the hybrid image display
        self.create_hybrid_image()

        # Toggle the flag
        self.mix = not self.mix

        print("Update")

    def create_hybrid_image(self) -> None:
        """
        Creates a hybrid image by combining low-pass and high-pass components of two images
        and displays the final result in a label.

        The method reads two input images, applies Gaussian blur to generate low-pass images,
        computes high-pass images by subtracting the low-pass components from the original images,
        and combines them to create a hybrid image. The mix flag determines the components' combination method.
        """
        image1_bgr = cv2.imread(self.hybrid_img_paths["img1"])
        image2_bgr = cv2.imread(self.hybrid_img_paths["img2"])

        # Convert BGR image to RGB
        image1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2RGB)

        # Normalize the images
        image1 = image1.astype(np.float32) / 255.0
        image2 = image2.astype(np.float32) / 255.0

        sigma1 = self.ui.lineEdit_sigma1.text()
        sigma1 = int(sigma1)
        sigma2 = self.ui.lineEdit_sigma2.text()
        sigma2 = int(sigma2)

        # Generate low-pass images using gaussian blur
        blurred_image1 = self.apply_filter_to_channels(
            image1, apply_gaussian_channel, sigma=sigma1
        )
        blurred_image2 = self.apply_filter_to_channels(
            image2, apply_gaussian_channel, sigma=sigma2
        )

        # Generate high-pass image
        high_pass_image1 = image1 - blurred_image1
        high_pass_image2 = image2 - blurred_image2

        # Create the hybrid image by combining low-pass and high-pass components
        if not self.mix:
            hybrid_image = blurred_image2 + high_pass_image1

        else:
            hybrid_image = blurred_image1 + high_pass_image2

        # Convert back to uint8
        hybrid_image_uint8 = (hybrid_image * 255).clip(0, 255).astype("uint8")

        # Create a QImage from the NumPy array
        height, width, channels = hybrid_image_uint8.shape
        qimage = QImage(
            hybrid_image_uint8.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        )

        # Create a QPixmap from the QImage
        pixmap = QPixmap.fromImage(qimage)
        self.ui.label_img_hybrid.setPixmap(pixmap.scaledToWidth(300))

    ###################### Edge detection ######################
    def line_enable(self):
        mask = self.ui.comboBox_edges.currentText()

        if mask in ["Sobel", "Roberts", "Prewitt"]:
            self.ui.lineEdit_canny_low.setDisabled(True)
            self.ui.lineEdit_canny_high.setDisabled(True)

        elif mask == "canny":
            self.ui.lineEdit_canny_low.setDisabled(False)
            self.ui.lineEdit_canny_high.setDisabled(False)

    def apply_edge_detection(self) -> None:
        """
        Reads the input image using the read_img method.
        Determines the selected edge detection method from the UI.
        Applies the selected method:
            - Sobel
            - Roberts
            - Prewitt
            - Canny

        """
        # always convert it to grey whether it is grey or colored
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Get the selected edge detection mask
        mask = self.ui.comboBox_edges.currentText()

        # Apply the selected edge detection mask
        if mask == "Sobel":

            gradient_magnitude, _ = apply_sobel(image)
        elif mask == "Roberts":

            edge_image = apply_roberts(image)
        elif mask == "Prewitt":

            edge_image = apply_prewitt(image)
        elif mask == "canny":

            edge_image = apply_canny(
                image,
                low_threshold=float(self.ui.lineEdit_canny_low.text()),
                high_threshold=float(self.ui.lineEdit_canny_high.text()),
            )

        # Display the resulting edge-detected image
        self.label_for_image(edge_image, self.ui.widget_output)
        # display(edge_image)

    ###################### Filters ######################

    def apply_filter(self) -> None:
        """
        Reads the input image using the read_img method.
        Determines the selected filter method from the UI.
        Applies the selected method:
        - Average
        - Gaussian
        - Median

        """
        image = self.noisy_img

        # Get the selected edge detection method
        method = self.ui.comboBox_filter.currentText()

        # Apply the selected filter method
        if method == "Average":
            filtered_image = self.apply_filter_to_channels(image, apply_average_channel)
        elif method == "Gaussian":
            filtered_image = self.apply_filter_to_channels(
                image, apply_gaussian_channel
            )
        elif method == "Median":
            filtered_image = self.apply_filter_to_channels(image, apply_median_channel)

        # Display the resulting filtered image
        if self.ui.checkBox.isChecked():
            self.label_for_colored_image(filtered_image, self.ui.widget_output)
        else:
            self.label_for_image(filtered_image, self.ui.widget_output)

    def apply_filter_to_channels(self, image: np.ndarray, function, **kwargs) -> np.ndarray:
        if len(image.shape) == 2:  # Grayscale image
            return function(image, **kwargs)
        elif len(image.shape) == 3:  # RGB image
            return np.stack(
                [function(image[:, :, i], **kwargs) for i in range(image.shape[2])],
                axis=-1,
            )

    ###################### Image Settings ######################

    def label_for_image(self, image, widget, flag=False):
        """
        Display the image in QPixmap as Grayscale
        """
        # Normalize the image data to the 0-255 range
        min_val = np.min(image)
        max_val = np.max(image)
        widget_height = widget.height()
        widget_width = widget.width()
        if min_val == max_val:
            # Handle the case where the image is a constant (avoid division by zero)
            if image is not None:
                normalized_image = (image * 255).astype(np.uint8)
            else:
                # Handle the case where image is None
                print("Error: Image is None")
        else:
            normalized_image = ((image - min_val) / (max_val - min_val) * 255).astype(
                np.uint8
            )

        # Convert to QPixmap
        q_img = QImage(
            normalized_image.data,
            normalized_image.shape[1],
            normalized_image.shape[0],
            normalized_image.shape[1],
            QImage.Format_Grayscale8,
        )

        img_pixmap = QPixmap.fromImage(q_img)

        # if widget == self.ui.widget_6:
        #     widget_height = widget.height() * 5

        # Scale the QPixmap to fit the widget size
        scaled_img_pixmap = img_pixmap.scaled(
            widget_width,
            widget_height,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        if flag == True:
            white_image = QImage(widget_width, widget_height, QImage.Format_RGB32)
            white_image.fill(QColor("white"))
            scaled_img_pixmap = QPixmap.fromImage(white_image)
            # print('got here')

        # Create a QLabel and set the QPixmap
        label = QLabel(widget)
        label.setPixmap(scaled_img_pixmap)
        label.setScaledContents(True)
        label.setGeometry(QtCore.QRect(0, 0, widget_width, widget_height))
        label.setAlignment(QtCore.Qt.AlignCenter)
        # print(widget_width, widget_height)
        label.show()

    def label_for_colored_image(self, image, widget, flag=False):
        """
        Display the image in QPixmap as RGB
        """

        # Normalize the image data to the 0-255 range for each color channel
        min_val = np.min(image, axis=(0, 1))
        max_val = np.max(image, axis=(0, 1))

        normalized_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[-1]):
            if min_val[i] == max_val[i]:
                # Handle the case where the image is a constant (avoid division by zero)
                normalized_image[:, :, i] = (image[:, :, i] * 255).astype(np.uint8)
            else:
                normalized_image[:, :, i] = (
                    (image[:, :, i] - min_val[i]) / (max_val[i] - min_val[i]) * 255
                ).astype(np.uint8)

        # Convert to QImage
        height, width, channel = normalized_image.shape
        q_img = QImage(
            normalized_image.data, width, height, channel * width, QImage.Format_RGB888
        )
        q_img = q_img.rgbSwapped()  # Swap RGB channels to BGR

        # Convert QImage to QPixmap
        img_pixmap = QPixmap.fromImage(q_img)

        # Scale the QPixmap to fit the widget size
        scaled_img_pixmap = img_pixmap.scaled(
            widget.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        if flag == True:
            white_image = QImage(widget.size(), QImage.Format_RGB32)
            white_image.fill(QColor("white"))
            scaled_img_pixmap = QPixmap.fromImage(white_image)
            # print('got here')

        # Create a QLabel and set the QPixmap
        label = QLabel(widget)
        label.setPixmap(scaled_img_pixmap)
        label.setScaledContents(True)
        label.setGeometry(QtCore.QRect(0, 0, widget.width(), widget.height()))
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()

    def show_image(self):
        """
        Displays the selected image in a label on the widget.
        The image is converted from BGR to RGB and scaled to fit the widget size.
        """
        img = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        height, width, channel = img_rgb.shape
        q_img = QImage(
            img_rgb.data, width, height, width * channel, QImage.Format_RGB888
        )
        img_pixmap = QPixmap.fromImage(q_img)
        scaled_img_pixmap = img_pixmap.scaled(
            self.ui.widget.width(), self.ui.widget.height()
        )
        label = QLabel(self.ui.widget)
        label.setPixmap(scaled_img_pixmap)
        label.setScaledContents(True)
        label.setGeometry(
            QtCore.QRect(0, 0, self.ui.widget.width(), self.ui.widget.height())
        )
        label.setAlignment(
            QtCore.Qt.AlignCenter
        )  # Align the image in the center of the widget
        label.show()

    def show_gray_image(self):
        """
        Displays the grayscale version of the selected image in a label on the widget.
        The image is converted to grayscale and scaled to fit the widget size.
        """
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        widget = self.ui.widget_6
        self.label_for_image(img, widget)

    ###################### Second tab ######################

    def histogram_tab(self):
        """
        Applies the selected image enhancement technique based on the current combo box selection.
        """
        method = self.ui.comboBox.currentText()

        if method == "Normalization":
            self.normalize()
        elif method == "Local Threshold":
            self.localThresholdingcv()
        elif method == "Global Threshold":
            self.globalThresholding()
        elif method == "Equalization":
            self.equalize_view()

    def normalize(self):
        """
        Normalize the input image.
        """
        # Read the input image
        image = self.read_img()

        # Perform normalization
        normalized_img = image  # Placeholder for the normalization process that will happen in the label_for_image function

        # Store the normalized image
        self.edited_image = normalized_img

        # Set the normalization flag
        self.normalize_flag = True

        # Display the normalized image
        widget = self.ui.widget_6
        self.label_for_colored_image(self.edited_image, widget)

    def globalThresholding(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        threshold = np.mean(img)
        thresholded_image = np.where(img > threshold, 255, 0).astype(np.uint8)
        self.edited_image = thresholded_image
        widget = self.ui.widget_6
        self.label_for_image(self.edited_image, widget)

    def localThresholdingcv(self):
        """
        Apply local thresholding to an input grayscale image.
        """
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # Initialize an empty array to store the thresholded image
        thresholded_image = np.zeros_like(image)
        height, width = image.shape
        block_size = 11
        constant = 2
        half_block = block_size // 2

        # Iterate over each pixel in the image
        for row in range(height):
            for col in range(width):
                # Define the region of interest
                start_row = max(0, row - half_block)
                end_row = min(height, row + half_block + 1)
                start_col = max(0, col - half_block)
                end_col = min(width, col + half_block + 1)

                # Calculate the local mean of the region
                local_mean = np.mean(image[start_row:end_row, start_col:end_col])

                # Apply the threshold
                if image[row, col] > local_mean - constant:
                    thresholded_image[row, col] = 255
                else:
                    thresholded_image[row, col] = 0

        self.edited_image = thresholded_image
        # display(self.edited_image)
        widget = self.ui.widget_6
        self.label_for_image(self.edited_image, widget)

    def histogram_equalization(self, array: np.ndarray) -> np.ndarray:
        """
        Performs histogram equalization on a given input array.

        Histogram equalization enhances the contrast of an image by redistributing pixel intensities.
        The method calculates the histogram of the image/channel array, normalizes it, computes the cumulative distribution function (CDF),
        and maps pixel values to a transformation map to achieve equalization.

        Args:
            array (np.ndarray): Input array (e.g., grayscale image or intensity channel of color image) with intensity values.

        Returns:
            np.ndarray: Equalized array with adjusted more uniform intensity values.
        """
        # Calculate histogram via binning
        histogram_array = np.bincount(array.flatten(), minlength=256)

        # Normalize
        num_pixels = np.sum(histogram_array)
        phistogram_array = histogram_array / num_pixels  # PDF

        # Normalized cumulative histogram (CDF)
        chistogram_array = np.cumsum(phistogram_array)

        # cdf * max grey level value (for 8-bit grayscale image - value is 255)
        transform_map = np.round(255 * chistogram_array).astype(np.uint8)

        # Transform pixel values to equalize while maintaining shape of img_array
        eq_array = transform_map[array]

        return eq_array

    def equalize(self) -> np.ndarray:
        """
        Performs histogram equalization on the loaded image.

        The method checks the image format (grayscale or RGB) and applies histogram equalization accordingly.
        For RGB images, it converts to the YCrCb color space, equalizes the Y channel, and converts back to RGB.
        For grayscale images, it directly applies histogram equalization.

        Raises:
            ValueError: If the image format is unsupported (i.e. not RGB/BGR or Grayscale).

        Returns:
            np.ndarray: Equalized image array.
        """
        # Load the image
        image_array = self.read_img()

        if (
            len(image_array.shape) == 3
            and image_array.shape[2] == 3
            and self.ui.checkBox.isChecked()
        ):
            # RGB image: Convert to YCrCb color space and equalize the Y channel
            ycrcb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2YCR_CB)
            y_channel = ycrcb_img[:, :, 0]
            y_equalized = self.histogram_equalization(y_channel)
            ycrcb_img[:, :, 0] = y_equalized
            eq_rgb_img_array = cv2.cvtColor(
                ycrcb_img, cv2.COLOR_YCrCb2BGR
            )  # Convert back to RGB
            return eq_rgb_img_array

        elif len(image_array.shape) == 2 and not self.ui.checkBox.isChecked():
            # Grayscale image: Apply histogram equalization directly
            eq_img_array = self.histogram_equalization(image_array)
            return eq_img_array
        else:
            raise ValueError("Unsupported image format")

    def equalize_view(self) -> None:
        """
        Displays the equalized image in the specified widget.

        The method calls the `equalize` method to obtain the equalized image. If the checkbox for colored images
        is checked, it displays the equalized RGB image; otherwise, it displays the equalized grayscale image.
        """
        widget = self.ui.widget_6
        self.edited_image = self.equalize()
        if self.ui.checkBox.isChecked():
            self.label_for_colored_image(self.edited_image, widget)
        else:
            self.label_for_image(self.edited_image, widget)

    def histogram_distribution(self):
        """
        Displays the histograms and cumulative curves of the selected image.
        Calculates and plots the grayscale and color histograms using the histo function.
        Plots the cumulative curves using the distribution_func function.
        """
        img = cv2.imread(self.image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.original_img = gray_img
        self.distribution_func(
            self.histo(gray_img, self.ui.widget_2, color="w"),
            self.ui.widget_7,
            color="w",
        )

        b, g, r = cv2.split(img)
        self.distribution_func(
            self.histo(b, self.ui.widget_3, color="b"), self.ui.widget_8, color="b"
        )
        self.distribution_func(
            self.histo(g, self.ui.widget_4, color="g"), self.ui.widget_9, color="g"
        )
        self.distribution_func(
            self.histo(r, self.ui.widget_5, color="r"), self.ui.widget_10, color="r"
        )

    def histo(self, img, widget, color):
        """
        Calculates and plots the histogram of an input image on the given widget.
        :param img: The input image to calculate the histogram from.
        :param widget: The widget to display the histogram plot.
        :param color: The color of the histogram plot.
        :return hist: The calculated histogram values.
        """
        widget.clear()
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])
        widget.plot(bins, hist, stepMode=True, fillLevel=0, brush=color, pen=color)
        return hist

    def distribution_func(self, hist, widget, color):
        """
        Plots the normalized cumulative curve of an input histogram on the given widget.
        :param hist: The histogram values to calculate the cumulative curve from.
        :param widget: The widget to display the cumulative curve plot.
        :param color: The color of the cumulative curve plot.
        """
        widget.clear()
        widget.plot(np.cumsum(hist) / np.sum(hist), pen=color)

    ###################### Noise  ######################

    def apply_noise(self) -> None:
        """
        call the function to apply the selected noise type
        """
        # Get the selected noise type
        method = self.ui.comboBox_noise.currentText()
        image = self.read_img()
        self.colored = self.ui.checkBox.isChecked()

        # Apply the selected noise
        if method == "Gaussian":
            self.noisy_img = gaussian_noise(image, self.colored)
        elif method == "Uniform":
            self.noisy_img = uniform_noise(image, self.colored)
        elif method == "Salt & Pepper":
            self.noisy_img = salt_pepper_noise(image, self.colored)

        if self.colored:
            self.label_for_colored_image(self.noisy_img, self.ui.widget_noised)
        else:
            self.label_for_image(self.noisy_img, self.ui.widget_noised)

    ##################### Freq filters ######################
    def label_for_image_freq(self, image, widget):
        """
        Display the image with low or high pass filter
        """
        if len(image.shape) == 2:
            # Grayscale image
            normalized_image = normalize_image(image)
            q_img = QImage(
                normalized_image.data.tobytes(),
                normalized_image.shape[1],
                normalized_image.shape[0],
                normalized_image.shape[1],
                QImage.Format_Grayscale8,
            )
        elif len(image.shape) == 3:
            # Colored image
            normalized_image = normalize_image(image)
            height, width, channel = normalized_image.shape
            q_img = QImage(
                normalized_image.data.tobytes(),
                width,
                height,
                channel * width,
                QImage.Format_RGB888,
            )
            q_img = q_img.rgbSwapped()  # Swap RGB channels to BGR

        # Convert QImage to QPixmap
        img_pixmap = QPixmap.fromImage(q_img)

        # Scale the QPixmap to fit the widget size
        scaled_img_pixmap = img_pixmap.scaled(
            widget.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        # Create a QLabel and set the QPixmap
        label = QLabel(widget)
        label.setPixmap(scaled_img_pixmap)
        label.setScaledContents(True)
        label.setGeometry(QtCore.QRect(0, 0, widget.width(), widget.height()))
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()

    def frequency_filter(self):
        original_img = self.read_img()
        # Check if the checkbox is checked
        if self.colored:
            # Apply filters to color image
            filtered_image_low_pass = apply_filter_low_high_pass(
                original_img, "Low pass"
            )
            filtered_image_high_pass = apply_filter_low_high_pass(
                original_img, "High pass"
            )
        else:
            # Apply filters to grayscale image
            filtered_image_low_pass = apply_filter_low_high_pass(
                original_img, "Low pass"
            )
            filtered_image_high_pass = apply_filter_low_high_pass(
                original_img, "High pass"
            )

        # Display the filtered images
        if self.ui.comboBox_freq.currentText() == "High pass":
            widget = self.ui.widget_output
            self.label_for_image_freq(filtered_image_high_pass, widget)
        elif self.ui.comboBox_freq.currentText() == "Low pass":
            widget = self.ui.widget_output
            self.label_for_image_freq(filtered_image_low_pass, widget)

    ############### TASK 2 ################ TASK 2 ################# TASK 2 ############# TASK 2 ############### TASK 2

    def hough_transform(self):
        method = self.ui.comboBox_2.currentText()
        if method == "Line":
            self.line_hough_transform()
        if method == "Circle":
            self.circle_hough_transform()
        if method == "Ellipse":
            self.ellipse_hough_transform()

    def line_hough_transform(self):
        print("line")
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the threshold value from the line edit
        threshold = int(self.ui.lineEdit_threshold.text())

        # Perform Hough Line Transform
        lines = line_detection(gray_image, threshold)

        # Draw the lines on the colored image
        for line in lines:
            r, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # Display the image with lines in the widget
        self.label_for_colored_image(image, self.ui.widget_hough_output)

    def circle_hough_transform(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get the threshold value from the line edit
        threshold = int(self.ui.lineEdit_threshold.text())
        # Resize the image to reduce its size
        resized_image = cv2.resize(
            gray_image, (0, 0), fx=0.5, fy=0.5
        )  # Change the scaling factor as needed

        # Extract image edges using Canny detector
        edges = apply_canny(resized_image, low_threshold=0.1, high_threshold=0.3)
        height, width = edges.shape[:2]
        max_radius = min(height, width) // 2  # Set max_radius based on image dimensions

        # Create accumulator array
        accumulator = np.zeros((height, width, max_radius), dtype=np.uint64)

        # Precompute cosine and sine values for all angles
        angles = np.deg2rad(np.arange(0, 360, 1))  # Use step_angle
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Get indices of edge pixels
        y_idxs, x_idxs = np.nonzero(edges)

        # Iterate over edge pixels and radii
        for x, y in zip(x_idxs, y_idxs):
            for radius in range(1, max_radius):  # Exclude radius 0 as it's meaningless
                # Calculate (x - a)^2 + (y - b)^2 = r^2 for all angles
                a_values = np.round(x - radius * cos_angles).astype(int)
                b_values = np.round(y - radius * sin_angles).astype(int)

                # Filter valid indices
                valid_indices = (
                    (a_values >= 0)
                    & (a_values < width)
                    & (b_values >= 0)
                    & (b_values < height)
                )
                a_values = a_values[valid_indices]
                b_values = b_values[valid_indices]

                # Increment accumulator for valid indices
                accumulator[b_values, a_values, radius - 1] += 1

        # Find circles with accumulator values above threshold
        circle_indices = np.where(accumulator >= threshold)
        circles = [(x, y, radius) for y, x, radius in zip(*circle_indices)]
        for circle in circles:
            center = (circle[0] * 2, circle[1] * 2)  # Scale back to original size
            radius = circle[2] * 2  # Scale back to original size
            cv2.circle(image, center, radius, (0, 255, 0), 2)
        # Display the image with lines in the widget
        self.label_for_colored_image(image, self.ui.widget_hough_output)
        print("circle")

    def ellipse_hough_transform(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize the image to reduce its size
        resized_image = cv2.resize(gray_image, (0, 0), fx=0.5, fy=0.5)
        # Extract image edges using Canny detector
        edges = apply_canny(resized_image, low_threshold=0.1, high_threshold=0.3)
        # Get the threshold value from the line edit
        threshold = int(self.ui.lineEdit_threshold.text())
        ellipses = ellipse_detection(edges, threshold, step_angle=1)

        # Draw detected ellipses on original image
        output_image = image.copy()
        for ellipse in ellipses:
            center = (ellipse[0] * 2, ellipse[1] * 2)  # Scale back to original size
            axis_a = ellipse[2] * 2  # Scale back to original size
            axis_b = ellipse[3] * 2  # Scale back to original size
            cv2.ellipse(
                output_image, center, (axis_a, axis_b), 0, 0, 360, (0, 255, 0), 2
            )

        # Display the image with ellipses in the widget
        self.label_for_colored_image(output_image, self.ui.widget_hough_output)
        print("ellipse")

    def apply_contour(self):
        self.ui.checkBox.setChecked(True)

        img = self.read_img()
        # print(img.shape)
        height, width, _ = img.shape
        # alpha = 0.001
        # beta = 0.4
        # gamma = 100
        alpha = self.ui.doubleSpinBox_alpha.value()
        beta = self.ui.doubleSpinBox_beta.value()
        gamma = self.ui.doubleSpinBox_gamma.value()

        contour = AC.apply_snake(img, 0.5, 0.5, width, height, alpha, beta, gamma)
        chain_code = AC.freeman_chain_code(contour)
        area = AC.calculate_area(chain_code)
        primeter = AC.calculate_perimeter(chain_code)

        contoured_img = ACC.apply_contour(img)

        widget = self.ui.widget_active_output

        # self.label_for_image(contoured_img, widget)

        self.label_for_colored_image(contoured_img, self.ui.widget_active_output)

        print(chain_code)
        self.ui.label_area.setText(str(area))
        self.ui.label_perimeter.setText(str(primeter))

    ############### TASK 3 ################ TASK 3 ################# TASK 3 ############# TASK 3 ############### TASK 3
    
    def apply_harris(self):
        image = self.read_img()
        # Get the selected operator
        select = self.ui.comboBox_harris.currentText()
        if select == "Harris":
            corner_image = harris.harris(image, harris_flag=True, window_size=5, k=0.04,
                                         threshold=float(self.ui.lineEdit_threshold_harris.text()),
                                         colored_flag=self.ui.checkBox.isChecked())
        elif select == "Lambda":
            corner_image = harris.harris(image, harris_flag=False, window_size=5, k=0.04,
                                         threshold=float(self.ui.lineEdit_threshold_harris.text()),
                                         colored_flag=self.ui.checkBox.isChecked())
        else:
            # Handle other cases or provide a default value for corner_image
            corner_image = None

        # Check if corner_image is None before accessing it
        if corner_image is not None:
            if self.ui.checkBox.isChecked():
                self.label_for_colored_image(corner_image, self.ui.widget_harris_output)
            else:
                self.label_for_image(corner_image, self.ui.widget_harris_output)

    def upload_ref_img(self):
        """
        Uploads a reference image for matching.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            self.ref_image_path = file_dialog.selectedFiles()[0]
            self.ref_image = cv2.imread(self.ref_image_path, cv2.IMREAD_COLOR)
            self.label_for_colored_image(self.ref_image, self.ui.widget_sift_ref)

    def upload_sec_img(self):
        """
        Uploads a secondary image for matching.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            self.sec_image_path = file_dialog.selectedFiles()[0]
            self.sec_image = cv2.imread(self.sec_image_path, cv2.IMREAD_COLOR)
            self.label_for_colored_image(self.sec_image, self.ui.widget_sift_img)

    def apply_NCC (self):
        self.select = 'NCC'
        self.apply_matching()

    def apply_SSD (self):
        self.select = 'SSD'
        self.apply_matching()

    def apply_matching(self):
        """
        Applies normalized cross-correlation (NCC) matching between a reference image and a secondary image.
        """
        img1 = cv2.imread(self.ref_image_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.sec_image_path, cv2.IMREAD_GRAYSCALE)
        # Compute keypoints and descriptors using SIFT
        keypoints1, descriptors1 = computeKeypointsAndDescriptors(img1)
        keypoints2, descriptors2 = computeKeypointsAndDescriptors(img2)
        matches = []
        threshold = float(self.ui.lineEdit_threshold_ssd.text())
        for i, descriptor1 in enumerate(descriptors1):
            best_match = None
            best_score = -1.0
            for j, descriptor2 in enumerate(descriptors2):
                if self.select == 'SSD':
                    score = np.mean(self.sum_of_squared_differences(descriptor1, descriptor2))
                if self.select == 'NCC':
                    score = np.mean(self.normalized_cross_correlation(descriptor1, descriptor2))
                if score > best_score:
                    best_match = j
                    best_score = score
            if best_score > threshold:
                matches.append(cv2.DMatch(i, best_match, 0))

        # Draw lines between matched keypoints
        matched_img = cv2.drawMatches(self.ref_image, keypoints1, self.sec_image, keypoints2, matches, None)
        self.label_for_colored_image(matched_img, self.ui.widget_sift_output)

    def normalized_cross_correlation(self, ref_img, sec_img):
        """
        Calculates the NCC score and the position of the best match between a reference image and a secondary image.
        """
        ref_mean = np.mean(ref_img)
        ref_std = np.std(ref_img)
        sec_mean = np.mean(sec_img)
        sec_std = np.std(sec_img)
        correlation = np.correlate((ref_img - ref_mean), (sec_img - sec_mean), mode='valid')
        return correlation / (ref_std * sec_std)

    def sum_of_squared_differences(self, ref_img, sec_img):
        """
        Calculates the SSD score between a reference image and a secondary image.
        """
        ssd = np.sum((ref_img - sec_img) ** 2)
        return ssd

    ############### TASK 4 ################ TASK 4 ################# TASK 4 ############# TASK 4 ############### TASK 4
    def apply_clustering(self) -> None:
        """Apply the clustering method chosen from the clustering combobox:
            - k-Means
            - Agglomerative Clustering
            - Region Growing
            - Mean-Shift
        """
        
        # get method from combobox
        method = self.ui.comboBox_clustering.currentText()
        # get image and display as input
        image = self.read_img()
        # if colored (RGB) checkbox is checked
        self.colored = self.ui.checkBox.isChecked()
        # Number of clusters
        k = int(self.ui.lineEdit_clusters.text())
        # Homogeneity Threshold 
        threshold = int(self.ui.lineEdit_threshold_clusters.text())

        # Calculate function run time
        start_time = timeit.default_timer()


        # Apply the selected clustering method
        if method == 'K-Means':
            # Reshape the image into a 2D array of pixels and 3 color values (RGB)
            pixel_vals = image.reshape((-1, 3))
            # Convert to float type
            pixel_vals = np.float32(pixel_vals)
            
            centroids = kmeans(pixel_vals)
            # Convert centroids to uint8
            centers = np.uint8(centroids)

            # Assign each pixel to the closest centroid
            labels = np.array([closest_centroid(pixel, centroids) for pixel in pixel_vals])

            # Convert data into 8-bit values
            segmented_data = centers[labels.flatten()]

            # Reshape data into the original image dimensions
            self.segmented_image = segmented_data.reshape((image.shape))
            # self.segmented_image = cv2.cvtColor(self.segmented_image, cv2.COLOR_BGR2RGB)

        elif method == 'Agglomerative Clustering':
            self.segmented_image = apply_agglomerative(image, clusters_numbers = k)
        
        elif method == 'Region Growing':
            # if supervised checkbox is checked
            self.supervised = self.ui.checkBox_supervised.isChecked()
            self.seeds = [] # clear seeds list

            if self.ui.radioButton_connect_8.isChecked():
                self.connect = True
            elif self.ui.radioButton_connect_4.isChecked():
                self.connect = False

            # Create GrowRegion Instance
            segments = GrowRegion(self.image_path, threshold, self.colored, self.connect)
            
            if self.supervised:
                # select seeds
                self.seeds = segments.selectSeeds(image)
            else: 
                # Unsupervised Region Growing
                # Create seed list based on image's local minima
                self.seeds = segments.generateSeeds()
            
            # check if seed list is not empty
            if self.seeds is not None:
                # plot seeds on input image
                self.plot_seeds()
                print(f"Seeds number: {len(self.seeds)}")
                # Apply region growing
                self.segmented_image = segments.applyRegionGrow(self.seeds)          
               
        elif method == 'Mean-Shift':
            self.segmented_image = shift_mean(image)

        end_time = timeit.default_timer()

        # Show only 5 digits after floating point
        elapsed_time = format(end_time - start_time, '.5f')
        print(elapsed_time)
        try:
            if self.colored or method == "Region Growing":
                self.label_for_colored_image(self.segmented_image, self.ui.widget_seg_output)
            else:
                self.label_for_image(self.segmented_image, self.ui.widget_seg_output)
        except TypeError:
            print("Cannot display Segmented Image")
            
    def plot_seeds(self):
        """
        Plots the seed points onto the input Region Growing image.
        
        Removes the original widget from the layout, reinitializes it with the ImagePlotter widget,
        adds it back to the layout, sets its stretch factor, and updates the layout.
        """
        # Remove the original widget from the layout
        self.ui.verticalLayout_44.removeWidget(self.ui.widget_seg_input)
        self.ui.widget_seg_input.deleteLater()

        # Reinitialize the original widget with the ImagePlotter widget
        self.ui.widget_seg_input = ImagePlotter(self.seeds, self.image_path, self.colored)
        self.ui.verticalLayout_44.addWidget(self.ui.widget_seg_input)
        self.ui.verticalLayout_44.setStretchFactor(self.ui.widget_seg_input, 1)
        self.ui.verticalLayout_44.update()  # Update the layout

    def thresholding_methods(self):
        """
        Applies the selected thresholding method to the grayscale image.

        Reads a grayscale image and applies the chosen thresholding method based on the UI selection.
        The resulting thresholded image is displayed in the UI widget.

        Supported methods: Optimal Thresholding, Otsu Thresholding, Spectral Thresholding.
        """
        method = self.ui.comboBox_thresholding.currentText()
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        if method == "Optimal Thresholding":
            if self.ui.radioButton_2.isChecked():
                threshold_img = optimal_global_thresholding(img)
                print("hello")
            elif self.ui.radioButton.isChecked():
                threshold_img = optimal_local_thresholding(img, block_size=90) 

        elif method == "Otsu Thresholding":
            if self.ui.radioButton_2.isChecked():
                threshold_img = globalOtsuThresholding(img)
            elif self.ui.radioButton.isChecked():
                threshold_img = localOtsuThresholding(img, block_size=181)

        elif method == "Spectral Thresholding ":
            if self.ui.radioButton_2.isChecked():
                threshold_img = globalSpectralThresholding(img)
            elif self.ui.radioButton.isChecked():
                threshold_img = localSpectralThresholding(img, block_size=50)

        self.label_for_image(threshold_img, self.ui.widget_seg_output)

    def map_to_luv(self):
        image = self.read_img()
        height, width, channels = image.shape

        # Create a new blank image for the mapped L*u*v* values
        luv_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Iterate through each pixel of the image
        for y in range(height):
            for x in range(width):
                # Get the RGB values of the current pixel
                R, G, B = image[y, x]

                # Convert RGB to L*u*v*
                L, u, v = RGB_to_Luv(R, G, B)

                # Store the L*u*v* values in the new image
                luv_image[y, x] = [L, u, v]


        self.label_for_colored_image(luv_image, self.ui.widget_output)

    ############### TASK 5 ################ TASK 5 ################# TASK 5 ############# TASK 5 ############### TASK 5


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setQuitOnLastWindowClosed(False)
    mainWindow = MyMainWindow()
    mainWindow.setWindowTitle("Computer vision tools")
    mainWindow.show()
    sys.exit(app.exec())
