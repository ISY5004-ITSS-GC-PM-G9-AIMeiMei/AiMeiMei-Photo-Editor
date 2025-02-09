import sys
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt6.QtCore import Qt, QPoint
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from controlnet_aux import OpenposeDetector  # ControlNet for reference-based inpainting
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from torchmetrics.image.fid import FrechetInceptionDistance


class InpaintingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Inpainting with ControlNet and FID")
        self.setGeometry(100, 100, 800, 600)

        self.image = None
        self.mask = None
        self.reference_images = []
        self.drawing = False
        self.last_point = QPoint()

        self.fid_metric = FrechetInceptionDistance(feature=2048)

        self.initUI()
        self.load_model()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_button)

        self.load_reference_button = QPushButton("Load Reference Images", self)
        self.load_reference_button.clicked.connect(self.load_reference_images)
        self.layout.addWidget(self.load_reference_button)

        self.inpaint_button = QPushButton("Apply AI Inpainting", self)
        self.inpaint_button.clicked.connect(self.apply_inpainting)
        self.layout.addWidget(self.inpaint_button)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

    def load_model(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        print("ControlNet model loaded successfully!")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png)")
        if file_path:
            self.image = Image.open(file_path).convert("RGBA")
            self.mask = Image.new("L", self.image.size, 0)
            self.display_image()

    def load_reference_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Reference Images", "", "Images (*.png)")
        if file_paths:
            self.reference_images = [Image.open(f).convert("RGB") for f in file_paths]
            print(f"{len(self.reference_images)} reference images loaded!")

    def display_image(self):
        if self.image:
            qt_image = self.pil_to_qimage(self.image)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def pil_to_qimage(self, pil_image):
        img_array = np.array(pil_image)
        height, width, channel = img_array.shape
        bytes_per_line = 4 * width
        return QImage(img_array.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)

    def compute_fid(self, generated_image):
        if not self.reference_images:
            print("No reference images loaded for FID calculation.")
            return

        ref_images_np = [np.array(ref).astype(np.uint8).transpose(2, 0, 1) for ref in self.reference_images]
        generated_image_np = np.array(generated_image).astype(np.uint8).transpose(2, 0, 1)

        self.fid_metric.update(torch.tensor([generated_image_np]), real=False)
        self.fid_metric.update(torch.tensor(ref_images_np), real=True)
        fid_score = self.fid_metric.compute().item()
        print(f"FID Score: {fid_score}")

    def apply_inpainting(self):
        if self.image:
            image_np = np.array(self.image)
            mask_np = (image_np[:, :, 3] == 0).astype(np.uint8) * 255  # Transparent areas as mask
            self.mask = Image.fromarray(mask_np, mode='L')

            if self.reference_images:
                controlnet_processor = OpenposeDetector()
                combined_edges = [controlnet_processor(ref) for ref in self.reference_images]

                prompt = "Fill the missing parts of the image using the reference structures."
                image_input = self.image.convert("RGB")

                result = self.pipe(
                    prompt=prompt,
                    image=image_input,
                    mask_image=self.mask,
                    control_image=combined_edges[0]  # Use first reference edges (Can be improved to blend)
                ).images[0]
            else:
                prompt = "Fill the missing areas naturally."
                result = self.pipe(
                    prompt=prompt,
                    image=self.image.convert("RGB"),
                    mask_image=self.mask
                ).images[0]

            self.image = result.convert("RGBA")
            self.display_image()
            self.compute_fid(self.image)
            print("Inpainting complete!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InpaintingApp()
    window.show()
    sys.exit(app.exec())