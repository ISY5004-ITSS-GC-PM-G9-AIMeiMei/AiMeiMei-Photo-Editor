# main.py
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

import torch
import torchvision


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
