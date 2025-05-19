# main.py
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow
import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/qt/plugins"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
