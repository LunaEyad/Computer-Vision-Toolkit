import cv2
from PyQt5 import QtWidgets, QtGui

class ImagePlotter(QtWidgets.QWidget):
    def __init__(self, points, img_path, flag):
        QtWidgets.QWidget.__init__(self)
        # Load the image with OpenCV and draw the points
        # flag - (0 for grayscale, 1 for color)
        img = cv2.imread(img_path, int(flag))
        if points is not None:
            for point in points:
                x, y = point  # Unpack the point
                cv2.circle(img, (y, x), 5, (0, 0, 255), -1)  # Draw a red circle at each point
        # Convert the OpenCV image to QImage
        if len(img.shape) == 3:  # RGB image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            format = QtGui.QImage.Format_RGB888
        elif len(img.shape) == 2:  # Grayscale image
            img_rgb = img
            h, w = img_rgb.shape
            bytes_per_line = w
            format = QtGui.QImage.Format_Grayscale8
        self._image = QtGui.QPixmap(QtGui.QImage(img_rgb.data, w, h, bytes_per_line, format))

    def paintEvent(self, paint_event):
        painter = QtGui.QPainter(self)
        painter.drawPixmap(self.rect(), self._image)

