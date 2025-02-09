import sys
import os
import cv2
import numpy as np
import onnxruntime as ort
from rembg import remove, new_session
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QGraphicsView, \
    QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QImage, QPainter
from PyQt6.QtCore import Qt, QPoint


def check_onnx_runtime():
    """Check if ONNX Runtime is properly initialized."""
    try:
        device = ort.get_device()
        print(f"ONNX Runtime is using: {device}")
    except Exception as e:
        print(f"Error checking ONNX Runtime: {e}")


def segment_human(image_path):
    """Segment the human from the image using U²-Net."""
    check_onnx_runtime()
    model_path = "u2net.onnx"
    if not os.path.exists(model_path):
        print("Error: Model not found. Please download it first.")
        return None, None
    session = new_session(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None, None

    image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    human_rgba = remove(image_rgba, session=session)
    return image_rgba, human_rgba


class ImageEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.background = None
        self.human = None
        self.human_pos = QPoint(0, 0)

    def initUI(self):
        self.setWindowTitle("Human Segmentation & Layer Editing")
        self.setGeometry(100, 100, 800, 600)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)

        self.loadButton = QPushButton("Load Image", self)
        self.loadButton.clicked.connect(self.load_image)

        self.flattenButton = QPushButton("Flatten Image", self)
        self.flattenButton.clicked.connect(self.flatten_image)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.loadButton)
        layout.addWidget(self.flattenButton)
        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.background, self.human = segment_human(file_path)
            if self.background is not None and self.human is not None:
                self.update_scene()

    def update_scene(self):
        self.scene.clear()
        bg_pixmap = self.convert_cv_qt(self.background)
        self.bg_item = QGraphicsPixmapItem(bg_pixmap)
        self.scene.addItem(self.bg_item)

        human_pixmap = self.convert_cv_qt(self.human)
        self.human_item = QGraphicsPixmapItem(human_pixmap)
        self.human_item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable)
        self.scene.addItem(self.human_item)

    def flatten_image(self):
        if self.background is None or self.human is None:
            return

        h, w, _ = self.background.shape
        final_image = self.background.copy()

        human_x = int(self.human_item.x())
        human_y = int(self.human_item.y())
        human_h, human_w, _ = self.human.shape

        x = max(0, min(human_x, w - human_w))
        y = max(0, min(human_y, h - human_h))

        alpha_human = self.human[:, :, 3] / 255.0
        for c in range(3):
            final_image[y:y + human_h, x:x + human_w, c] = (
                    alpha_human * self.human[:, :, c] + (1 - alpha_human) * final_image[y:y + human_h, x:x + human_w, c]
            )

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "output.png", "Images (*.png)")
        if save_path:
            cv2.imwrite(save_path, final_image)
            print(f"Image saved to {save_path}")

    def convert_cv_qt(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qt_img)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)  # Create PyQt application
    editor = ImageEditor()  # Initialize your GUI
    editor.show()  # Show the window
    sys.exit(app.exec())  # Run the event loop