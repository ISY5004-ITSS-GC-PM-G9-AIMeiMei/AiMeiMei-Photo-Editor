from turtle import width
import PyQt6
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QSlider,
    QToolBar,
    QToolButton,
    QFileDialog,
    QStatusBar
)
from PyQt6.QtGui import QPixmap, QAction
import sys

from QImageViewer import QtImageViewer
from PyQt6.QtGui import QKeySequence
import pyqtgraph as pg
from QColorPicker import QColorPicker
import os
from QFlowLayout import QFlowLayout
from PIL import Image, ImageEnhance, ImageFilter
import QCurveWidget


def free_gpu_cache():
    import torch
    from GPUtil import showUtilization as gpu_usage

    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    print("GPU Usage after emptying the cache")
    gpu_usage()


def importLibraries():
    import torch
    import numpy as np
    import cv2
    import PIL
    print("Torch version", torch.__version__)
    print("Torch CUDA available?", "YES" if torch.cuda.is_available() else "NO")
    print("cv2 version", cv2.__version__)
    print("numpy version", np.__version__)
    print("PIl version", PIL.__version__)


class Gui(QtWidgets.QMainWindow):
    sliderChangeSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)
        self.setWindowTitle('PhotoLab')
        self.setMinimumHeight(850)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        ##############################################################################################
        ##############################################################################################
        # Create Histogram
        ##############################################################################################
        ##############################################################################################

        # Compute image histogram
        r_histogram = []
        g_histogram = []
        b_histogram = []

        # ITU-R 601-2 luma transform:
        luma_histogram = []

        # Create histogram plot
        self.ImageHistogramPlot = pg.plot()
        x = list(range(len(r_histogram)))
        self.ImageHistogramGraphRed = pg.PlotCurveItem(x=x, y=r_histogram, fillLevel=2, width=1.0,
                                                       brush=(255, 0, 0, 80))
        self.ImageHistogramGraphGreen = pg.PlotCurveItem(x=x, y=g_histogram, fillLevel=2, width=1.0,
                                                         brush=(0, 255, 0, 80))
        self.ImageHistogramGraphBlue = pg.PlotCurveItem(x=x, y=b_histogram, fillLevel=2, width=1.0,
                                                        brush=(0, 0, 255, 80))
        self.ImageHistogramGraphLuma = pg.PlotCurveItem(x=x, y=luma_histogram, fillLevel=2, width=1.0,
                                                        brush=(255, 255, 255, 80))
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphRed)
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphGreen)
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphBlue)
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphLuma)
        self.HistogramContent = None
        self.ImageHistogramPlot.hide()

        ##############################################################################################
        ##############################################################################################
        # Color Picker
        ##############################################################################################
        ##############################################################################################
        self.color_picker = None

        ##############################################################################################
        ##############################################################################################
        # Adjustment Sliders
        ##############################################################################################
        ##############################################################################################

        # State of enhance sliders
        self.RedFactor = 100
        self.GreenFactor = 100
        self.BlueFactor = 100
        self.Temperature = 6000  # Kelvin, maps to (255,255,255), direct sunlight
        self.Color = 100
        self.Brightness = 100
        self.Contrast = 100
        self.Sharpness = 100

        # State of filter sliders
        self.GaussianBlurRadius = 0

        self.timer_id = -1
        self.sliderExplanationOfChange = None
        self.sliderTypeOfChange = None
        self.sliderValueOfChange = None
        self.sliderObjectOfChange = None

        
        self.PaintToolButton = QToolButton(self)
        self.PaintToolButton.setText("&Paint")
        self.PaintToolButton.setToolTip("Paint")
        self.setIconPixmapWithColor(self.PaintToolButton, "icons/paint.svg")
        self.PaintToolButton.setCheckable(True)
        self.PaintToolButton.toggled.connect(self.OnPaintToolButton)

        ##############################################################################################
        ##############################################################################################
        # Prompt Selection Mode
        ##############################################################################################
        ##############################################################################################

        self.promptSelectToolButton = QToolButton(self)
        self.promptSelectToolButton.setText("&Prompt Selection")
        self.promptSelectToolButton.setToolTip("Prompt Selection")
        #self.setIconPixmapWithColor(self.promptSelectToolButton, "")
        self.promptSelectToolButton.setCheckable(True)
        self.promptSelectToolButton.toggled.connect(self.OnPromptSelectToolButton)

        ##############################################################################################
        ##############################################################################################
        # Auto Selection Mode
        ##############################################################################################
        ##############################################################################################

        self.autoSelectToolButton = QToolButton(self)
        self.autoSelectToolButton.setText("&Auto Selection")
        self.autoSelectToolButton.setToolTip("Auto Selection")
        #self.setIconPixmapWithColor(self.autoSelectToolButton, "")
        self.autoSelectToolButton.setCheckable(True)
        self.autoSelectToolButton.toggled.connect(self.OnAutoSelectToolButton)

        
        ##############################################################################################
        ##############################################################################################
        # Transform Selection Mode
        ##############################################################################################
        ##############################################################################################

        self.transfromSelectToolButton = QToolButton(self)
        self.transfromSelectToolButton.setText("&Transform Selection")
        self.transfromSelectToolButton.setToolTip("Transform Selection")
        #self.setIconPixmapWithColor(self.autoSelectToolButton, "")
        self.transfromSelectToolButton.setCheckable(True)
        self.transfromSelectToolButton.toggled.connect(self.OnTransfromSelectToolButton)
        
        
           
        ##############################################################################################
        ##############################################################################################
        # Crop Tool
        ##############################################################################################
        ##############################################################################################

        self.CropToolButton = QToolButton(self)
        self.CropToolButton.setText("&Crop")
        self.setIconPixmapWithColor(self.CropToolButton, "icons/crop.svg")
        self.CropToolButton.setToolTip("Crop")
        self.CropToolButton.setCheckable(True)
        self.CropToolButton.toggled.connect(self.OnCropToolButton)

        self.CropToolShortcut = QtGui.QShortcut(QKeySequence("Ctrl+Shift+Alt+K"), self)
        self.CropToolShortcut.activated.connect(lambda: self.CropToolButton.toggle())

        ##############################################################################################
        ##############################################################################################
        # Super-Resolution Tool
        ##############################################################################################
        ##############################################################################################

        self.SuperResolutionToolButton = QToolButton(self)
        self.SuperResolutionToolButton.setText("&Super Resolution")
        self.SuperResolutionToolButton.setToolTip("Super-Resolution")
        self.setIconPixmapWithColor(self.SuperResolutionToolButton, "icons/super_resolution.svg")
        self.SuperResolutionToolButton.setCheckable(True)
        self.SuperResolutionToolButton.toggled.connect(self.OnSuperResolutionToolButton)

        ##############################################################################################
        ##############################################################################################
        # White Balance Tool
        # https://github.com/mahmoudnafifi/WB_sRGB
        ##############################################################################################
        ##############################################################################################

        self.WhiteBalanceToolButton = QToolButton(self)
        self.WhiteBalanceToolButton.setText("&White Balance")
        self.WhiteBalanceToolButton.setToolTip("White Balance")
        self.setIconPixmapWithColor(self.WhiteBalanceToolButton, "icons/white_balance.svg")
        self.WhiteBalanceToolButton.setCheckable(True)
        self.WhiteBalanceToolButton.toggled.connect(self.OnWhiteBalanceToolButton)

        ##############################################################################################
        ##############################################################################################
        # Eraser Tool
        ##############################################################################################
        ##############################################################################################

        self.EraserToolButton = QToolButton(self)
        self.EraserToolButton.setText("&Eraser")
        self.EraserToolButton.setToolTip("Eraser")
        self.setIconPixmapWithColor(self.EraserToolButton, "icons/eraser.svg")
        self.EraserToolButton.setCheckable(True)
        self.EraserToolButton.toggled.connect(self.OnEraserToolButton)

        ##############################################################################################
        ##############################################################################################
        # Instagram Filters Tool
        ##############################################################################################
        ##############################################################################################

        self.InstagramFiltersToolButton = QToolButton(self)
        self.InstagramFiltersToolButton.setText("&Instagram Filters")
        self.InstagramFiltersToolButton.setToolTip("Instagram Filters")
        self.setIconPixmapWithColor(self.InstagramFiltersToolButton, "icons/instagram.svg")
        self.InstagramFiltersToolButton.setCheckable(True)
        self.InstagramFiltersToolButton.toggled.connect(self.OnInstagramFiltersToolButton)

        ##############################################################################################
        ##############################################################################################
        # Apply Tranformation
        ##############################################################################################
        ##############################################################################################

        self.ApplyToolButton = QToolButton(self)
        self.ApplyToolButton.setText("&Apply")
        self.ApplyToolButton.setToolTip("Apply")
        self.setIconPixmapWithColor(self.PaintToolButton, "icons/apply.svg")
        self.ApplyToolButton.setCheckable(True)
        self.ApplyToolButton.toggled.connect(self.OnApplyToolButton)

        # Toolbar
        ##############################################################################################
        ##############################################################################################

        self.tools = {           
            "paint": {
                "tool": "PaintToolButton",
                "var": '_isPainting'
            },           
            "prompt_select": {
                "tool": "promptSelectToolButton",
                "var": '_isSelectingRect',
                "destructor": 'exitSelectRect'
            }, 
            "auto_select": {
                "tool": "autoSelectToolButton",
                "var": '_isSelectingRect',
                "destructor": 'exitSelectRect'
            },  
            "transform_select": {
                "tool": "transfromSelectToolButton",
                "var": '_isSelectingRect',
                "destructor": 'exitSelectRect'
            },          
            "crop": {
                "tool": "CropToolButton",
                "var": '_isCropping'
            },
           "eraser": {
                "tool": "EraserToolButton",
                "var": '_isErasing'
            },
           "instagram_filters": {
                "tool": "InstagramFiltersToolButton",
                "var": '_isApplyingFilter'
            },
             "Apply": {
                "tool": "InstagramFiltersToolButton",
                "var": '_isApplyingFilter'
            },
        }

        self.ToolbarDockWidget = QtWidgets.QDockWidget("Tools")
        self.ToolbarDockWidget.setTitleBarWidget(QtWidgets.QWidget())
        ToolbarContent = QtWidgets.QWidget()
        ToolbarLayout = QFlowLayout(ToolbarContent)
        ToolbarLayout.setSpacing(0)

        self.ToolButtons = [
            self.PaintToolButton, self.EraserToolButton,
            self.promptSelectToolButton, self.CropToolButton,
            self.InstagramFiltersToolButton,
            self.WhiteBalanceToolButton,
           self.SuperResolutionToolButton,
           self.ApplyToolButton,
           self.autoSelectToolButton,
           self.transfromSelectToolButton
        ]

        for button in self.ToolButtons:
            button.setIconSize(QtCore.QSize(20, 20))
            button.setEnabled(False)
            button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
            ToolbarLayout.addWidget(button)

        ToolbarContent.setLayout(ToolbarLayout)
        self.ToolbarDockWidget.setWidget(ToolbarContent)

        ##############################################################################################
        ##############################################################################################
        # Right Dock
        ##############################################################################################
        ##############################################################################################

        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.ToolbarDockWidget)
        self.ToolbarDockWidget.setFloating(True)
        self.ToolbarDockWidget.setGeometry(QtCore.QRect(20, 20, 90, 600))

        ##############################################################################################
        ##############################################################################################
        # Show Window
        ##############################################################################################
        ##############################################################################################

        self.initImageViewer()
        self.ToolbarDockWidget.setParent(self.image_viewer)
        self.showMaximized()

        self.threadpool = QtCore.QThreadPool()
        self.sliderChangedPixmap = None
        self.sliderExplanationOfChange = None
        self.sliderTypeOfChange = None
        self.sliderValueOfChange = None
        self.sliderObjectOfChange = None
        self.sliderChangeSignal.connect(self.onUpdateImageCompleted)
        self.sliderWorkers = []

        self.resizeDockWidgets()
        self.createMenu()

    def createMenu(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")

        openAction = QAction("&Open File", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.OnOpen)
        fileMenu.addAction(openAction)

        saveAction = QAction("&Save File", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.OnSave)
        fileMenu.addAction(saveAction)

        saveAsAction = QAction("&Save File As", self)
        saveAsAction.setShortcut("Ctrl+Shift+S")
        saveAsAction.triggered.connect(self.OnSaveAs)
        fileMenu.addAction(saveAsAction)

        editMenu = menubar.addMenu("&Edit")

        undoAction = QAction("&Undo", self)
        undoAction.setShortcut("Ctrl+Z")
        undoAction.triggered.connect(self.OnUndo)
        editMenu.addAction(undoAction)

        pasteAction = QAction("&Paste", self)
        pasteAction.setShortcut("Ctrl+v")
        pasteAction.triggered.connect(self.OnPaste)
        editMenu.addAction(undoAction)

    def setIconPixmapWithColor(self, button, filename, findColor='black', newColor='white'):
        pixmap = QPixmap(filename)
        mask = pixmap.createMaskFromColor(QtGui.QColor(findColor), Qt.MaskMode.MaskOutColor)
        pixmap.fill((QtGui.QColor(newColor)))
        pixmap.setMask(mask)
        button.setIcon(QtGui.QIcon(pixmap))

    def setToolButtonStyleChecked(self, button):
        button.setStyleSheet('''
            border-color: rgb(22, 22, 22);
            background-color: rgb(22, 22, 22);
            border-style: solid;
        ''')

    def setToolButtonStyleUnchecked(self, button):
        button.setStyleSheet("")

    def resizeDockWidgets(self):
        pass
        # self.resizeDocks([self.ToolbarDockWidget], [200], Qt.Orientation.Vertical)

    @QtCore.pyqtSlot(int, str)
    def updateProgressBar(self, e, label):
        self.progressBar.setValue(e)
        self.progressBarLabel.setText(label)

    def initImageViewer(self):
        self.image_viewer = QtImageViewer(self)
        self.layerListDock = None
        self.CurvesDock = None

        # Set viewer's aspect ratio mode.
        # !!! ONLY applies to full image view.
        # !!! Aspect ratio always ignored when zoomed.
        #   Qt.AspectRatioMode.IgnoreAspectRatio: Fit to viewport.
        #   Qt.AspectRatioMode.KeepAspectRatio: Fit in viewport using aspect ratio.
        #   Qt.AspectRatioMode.KeepAspectRatioByExpanding: Fill viewport using aspect ratio.
        self.image_viewer.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio

        # Set the viewer's scroll bar behaviour.
        #   Qt.ScrollBarPolicy.ScrollBarAlwaysOff: Never show scroll bar.
        #   Qt.ScrollBarPolicy.ScrollBarAlwaysOn: Always show scroll bar.
        #   Qt.ScrollBarPolicy.ScrollBarAsNeeded: Show scroll bar only when zoomed.
        self.image_viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_viewer.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Allow zooming by draggin a zoom box with the left mouse button.
        # !!! This will still emit a leftMouseButtonReleased signal if no dragging occured,
        #     so you can still handle left mouse button clicks in this way.
        #     If you absolutely need to handle a left click upon press, then
        #     either disable region zooming or set it to the middle or right button.
        self.image_viewer.regionZoomButton = Qt.MouseButton.LeftButton  # set to None to disable

        # Pop end of zoom stack (double click clears zoom stack).
        self.image_viewer.zoomOutButton = Qt.MouseButton.RightButton  # set to None to disable

        # Mouse wheel zooming.
        self.image_viewer.wheelZoomFactor = 1.25  # Set to None or 1 to disable

        # Allow panning with the middle mouse button.
        self.image_viewer.panButton = Qt.MouseButton.MiddleButton  # set to None to disable

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(self.image_viewer)

    def resetSliderValues(self):
        # State of enhance sliders
        self.RedFactor = 100
        self.BlueFactor = 100
        self.GreenFactor = 100
        self.Temperature = 6000
        self.Color = 100
        self.Brightness = 100
        self.Contrast = 100
        self.Sharpness = 100
        self.GaussianBlurRadius = 0

        self.RedColorSlider.setValue(self.RedFactor)
        self.GreenColorSlider.setValue(self.GreenFactor)
        self.BlueColorSlider.setValue(self.BlueFactor)
        self.TemperatureSlider.setValue(self.Temperature)
        self.ColorSlider.setValue(self.Color)
        self.BrightnessSlider.setValue(self.Brightness)
        self.ContrastSlider.setValue(self.Contrast)
        self.SharpnessSlider.setValue(self.Sharpness)
        self.GaussianBlurSlider.setValue(self.GaussianBlurRadius)

    def getCurrentLayerLatestPixmap(self):
        return self.image_viewer.getCurrentLayerLatestPixmap()

    def processSliderChange(self, explanationOfChange, typeOfChange, valueOfChange, objectOfChange):
        self.sliderExplanationOfChange = explanationOfChange
        self.sliderTypeOfChange = typeOfChange
        self.sliderValueOfChange = valueOfChange
        self.sliderObjectOfChange = objectOfChange

        if self.timer_id != -1:
            self.killTimer(self.timer_id)

        self.timer_id = self.startTimer(500)

    def QPixmapToImage(self, pixmap):
        width = pixmap.width()
        height = pixmap.height()
        image = pixmap.toImage()

        byteCount = image.bytesPerLine() * height
        data = image.constBits().asstring(byteCount)
        return Image.frombuffer('RGBA', (width, height), data, 'raw', 'BGRA', 0, 1)

    def ImageToQPixmap(self, image):
        from PIL.ImageQt import ImageQt
        return QPixmap.fromImage(ImageQt(image))

    def EnhanceImage(self, Pixmap, Property, value):
        CurrentImage = self.QPixmapToImage(Pixmap)
        AdjustedImage = Property(CurrentImage).enhance(float(value) / 100)
        return self.ImageToQPixmap(AdjustedImage)

    def ApplyGaussianBlur(self, Pixmap, value):
        CurrentImage = self.QPixmapToImage(Pixmap)
        AdjustedImage = CurrentImage.filter(ImageFilter.GaussianBlur(radius=value))
        return self.ImageToQPixmap(AdjustedImage)

    def UpdateReds(self, Pixmap, value):
        CurrentImage = self.QPixmapToImage(Pixmap)

        # Split into channels
        r, g, b, a = CurrentImage.split()

        # Increase Reds
        r = r.point(lambda i: i * value)

        # Recombine back to RGB image
        AdjustedImage = Image.merge('RGBA', (r, g, b, a))

        return self.ImageToQPixmap(AdjustedImage)

    def AddRedColorSlider(self, layout):
        self.RedColorSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.RedColorSlider.setRange(0, 200)  # 1 is original image, 0 is black image
        layout.addRow("Red", self.RedColorSlider)

        # Default value of the Color slider
        self.RedColorSlider.setValue(100)

        self.RedColorSlider.valueChanged.connect(self.OnRedColorChanged)

    def OnRedColorChanged(self, value):
        self.RedFactor = value
        self.processSliderChange("Red", "Slider", value, "RedColorSlider")

    def UpdateGreens(self, Pixmap, value):
        CurrentImage = self.QPixmapToImage(Pixmap)

        # Split into channels
        r, g, b, a = CurrentImage.split()

        # Increase Greens
        g = g.point(lambda i: i * value)

        # Recombine back to RGB image
        AdjustedImage = Image.merge('RGBA', (r, g, b, a))

        return self.ImageToQPixmap(AdjustedImage)

    def AddGreenColorSlider(self, layout):
        self.GreenColorSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.GreenColorSlider.setRange(0, 200)  # 1 is original image, 0 is black image
        layout.addRow("Green", self.GreenColorSlider)

        # Default value of the Color slider
        self.GreenColorSlider.setValue(100)

        self.GreenColorSlider.valueChanged.connect(self.OnGreenColorChanged)

    def OnGreenColorChanged(self, value):
        self.GreenFactor = value
        self.processSliderChange("Green", "Slider", value, "GreenColorSlider")

    def UpdateBlues(self, Pixmap, value):
        CurrentImage = self.QPixmapToImage(Pixmap)

        # Split into channels
        r, g, b, a = CurrentImage.split()

        # Increase Blues
        b = b.point(lambda i: i * value)

        # Recombine back to RGB image
        AdjustedImage = Image.merge('RGBA', (r, g, b, a))

        return self.ImageToQPixmap(AdjustedImage)

    def AddBlueColorSlider(self, layout):
        self.BlueColorSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.BlueColorSlider.setRange(0, 200)  # 1 is original image, 0 is black image
        layout.addRow("Blue", self.BlueColorSlider)

        # Default value of the Color slider
        self.BlueColorSlider.setValue(100)

        self.BlueColorSlider.valueChanged.connect(self.OnBlueColorChanged)

    def OnBlueColorChanged(self, value):
        self.BlueFactor = value
        self.processSliderChange("Blue", "Slider", value, "BlueColorSlider")

    def AddTemperatureSlider(self, layout):
        self.TemperatureSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.TemperatureSlider.setRange(0, 12000)
        layout.addRow("Temperature", self.TemperatureSlider)

        # Default value of the Temperature slider
        self.TemperatureSlider.setValue(6000)

        self.TemperatureSlider.valueChanged.connect(self.OnTemperatureChanged)

    def OnTemperatureChanged(self, value):
        self.Temperature = value
        self.processSliderChange("Temperature", "Slider", value, "TemperatureSlider")

    def AddColorSlider(self, layout):
        self.ColorSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ColorSlider.setRange(0, 200)  # 1 is original image, 0 is black image
        layout.addRow("Saturation", self.ColorSlider)

        # Default value of the Color slider
        self.ColorSlider.setValue(100)

        self.ColorSlider.valueChanged.connect(self.OnColorChanged)

    def OnColorChanged(self, value):
        self.Color = value
        self.processSliderChange("Saturation", "Slider", value, "ColorSlider")

    def AddBrightnessSlider(self, layout):
        self.BrightnessSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.BrightnessSlider.setRange(0, 200)  # 1 is original image, 0 is black image
        layout.addRow("Brightness", self.BrightnessSlider)

        # Default value of the brightness slider
        self.BrightnessSlider.setValue(100)

        self.BrightnessSlider.valueChanged.connect(self.OnBrightnessChanged)

    def OnBrightnessChanged(self, value):
        self.Brightness = value
        self.processSliderChange("Brightness", "Slider", value, "BrightnessSlider")

    def AddContrastSlider(self, layout):
        self.ContrastSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ContrastSlider.setRange(0, 200)  # 1 is original image, 0 is a solid grey image
        layout.addRow("Contrast", self.ContrastSlider)

        # Default value of the brightness slider
        self.ContrastSlider.setValue(100)

        self.ContrastSlider.valueChanged.connect(self.OnContrastChanged)

    def OnContrastChanged(self, value):
        self.Contrast = value
        self.processSliderChange("Contrast", "Slider", value, "ContrastSlider")

    def AddSharpnessSlider(self, layout):
        self.SharpnessSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.SharpnessSlider.setRange(0, 200)  # 1 is original image, 0 is black image
        layout.addRow("Sharpness", self.SharpnessSlider)

        # Default value of the Sharpness slider
        self.SharpnessSlider.setValue(100)

        self.SharpnessSlider.valueChanged.connect(self.OnSharpnessChanged)

    def OnSharpnessChanged(self, value):
        self.Sharpness = value
        self.processSliderChange("Sharpness", "Slider", value, "SharpnessSlider")

    def AddGaussianBlurSlider(self, layout):
        self.GaussianBlurSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.GaussianBlurSlider.setRange(0, 2000)
        layout.addRow("Gaussian Blur", self.GaussianBlurSlider)
        self.GaussianBlurSlider.valueChanged.connect(self.OnGaussianBlurChanged)

    def OnGaussianBlurChanged(self, value):
        self.GaussianBlurRadius = value
        self.processSliderChange("Gaussian Blur", "Slider", value, "GaussianBlurSlider")

    def UpdateHistogramPlot(self):
        # Compute image histogram
        img = self.QPixmapToImage(self.image_viewer.pixmap())
        r, g, b, a = img.split()
        r_histogram = r.histogram()
        g_histogram = g.histogram()
        b_histogram = b.histogram()

        # ITU-R 601-2 luma transform:
        luma_histogram = [sum(x) for x in zip([item * float(299 / 1000) for item in r_histogram],
                                              [item * float(587 / 1000) for item in g_histogram],
                                              [item * float(114 / 1000) for item in b_histogram])]

        # Update histogram plot
        self.ImageHistogramGraphRed.setData(y=r_histogram)
        self.ImageHistogramGraphGreen.setData(y=g_histogram)
        self.ImageHistogramGraphBlue.setData(y=b_histogram)
        self.ImageHistogramGraphLuma.setData(y=luma_histogram)

    @QtCore.pyqtSlot()
    def onUpdateImageCompleted(self):
        if self.sliderChangedPixmap:
            self.image_viewer.setImage(self.sliderChangedPixmap, False, self.sliderExplanationOfChange,
                                       self.sliderTypeOfChange, self.sliderValueOfChange, self.sliderObjectOfChange)
            self.UpdateHistogramPlot()

    def RemoveRenderedCursor(self):
        # The cursor overlay is being rendered in the view
        # Remove it
        if any([self.image_viewer._isBlurring, self.image_viewer._isRemovingSpots]):
            pixmap = self.getCurrentLayerLatestPixmap()
            self.image_viewer.setImage(pixmap, False)

    def InitTool(self):
        self.RemoveRenderedCursor()

    def OnPaintToolButton(self, checked):
        if  self.image_viewer._isRectangleSelectPreseed:
            self.image_viewer.clearImage()
            self.image_viewer.open(self.image_viewer._current_filename)
        if checked:
            self.InitTool()

            class ColorPickerWidget(QtWidgets.QWidget):
                def __init__(self, parent, mainWindow):
                    QtWidgets.QWidget.__init__(self, parent)
                    self.parent = parent
                    self.closed = False
                    self.mainWindow = mainWindow

                def closeEvent(self, event):
                    self.destroyed.emit()
                    event.accept()
                    self.closed = True
                    self.mainWindow.DisableTool("paint")

            self.PaintContent = ColorPickerWidget(None, self)
            ColorPickerLayout = QtWidgets.QVBoxLayout(self.PaintContent)
            self.color_picker = QColorPicker(self.PaintContent, rgb=(173, 36, 207))
            self.image_viewer.ColorPicker = self.color_picker
            ColorPickerLayout.addWidget(self.color_picker)
            self.EnableTool("paint") if checked else self.DisableTool("paint")

            self.PaintContent.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
            self.PaintContent.show()
            # Create a local event loop for this widget
            loop = QtCore.QEventLoop()
            self.PaintContent.destroyed.connect(loop.quit)
            loop.exec()  # wait
        else:
            self.DisableTool("paint")
            self.PaintContent.hide()

    def OnCropToolButton(self, checked):
        if  self.image_viewer._isRectangleSelectPreseed:
            self.image_viewer.clearImage()
            self.image_viewer.open(self.image_viewer._current_filename)
            self.image_viewer._isRectangleSelectPreseed = False

        if checked:
            self.InitTool()
            self.image_viewer._isCropping = True    

    def OnPromptSelectToolButton(self, checked):
        if(checked):
            self.image_viewer.set_mode("selection")
            self.InitTool()
            self.EnableTool("prompt_select") if checked else self.DisableTool("prompt_select")
        else:
            self.image_viewer.set_mode("")

    def OnAutoSelectToolButton(self, checked):
        if(checked):
            self.image_viewer.set_mode("auto")
            self.InitTool()
            self.EnableTool("auto_select") if checked else self.DisableTool("auto_select")
        else:
            self.image_viewer.set_mode("")
    
    def OnTransfromSelectToolButton(self, checked):
        if(checked):
            self.image_viewer.set_mode("transform")
            self.InitTool()
            self.EnableTool("transfrom_select") if checked else self.DisableTool("transfrom_select")
        else:
            self.image_viewer.set_mode("")
           
    def onSuperResolutionCompleted(self, tool):
        if  self.image_viewer._isRectangleSelectPreseed:
            self.image_viewer.clearImage()
            self.image_viewer.open(self.image_viewer._current_filename)
            self.image_viewer._isRectangleSelectPreseed = False
        output = tool.output
        if output is not None:
            # Save new pixmap
            output = Image.fromarray(output)
            updatedPixmap = self.ImageToQPixmap(output)
            self.image_viewer.setImage(updatedPixmap, True, "Super Resolution")

        self.SuperResolutionToolButton.setChecked(False)
        del tool
        tool = None

    def OnSuperResolutionToolButton(self, checked):
        if  self.image_viewer._isRectangleSelectPreseed:
            self.image_viewer.clearImage()
            self.image_viewer.open(self.image_viewer._current_filename)
            self.image_viewer._isRectangleSelectPreseed = False
        if checked:
            self.InitTool()
            currentPixmap = self.getCurrentLayerLatestPixmap()
            image = self.QPixmapToImage(currentPixmap)

            from QToolSuperResolution import QToolSuperResolution
            widget = QToolSuperResolution(None, image, self.onSuperResolutionCompleted)
            widget.show()

    @QtCore.pyqtSlot()
    def onWhiteBalanceCompleted(self, tool):
        output = tool.output
        if output is not None:
            # Save new pixmap
            updatedPixmap = self.ImageToQPixmap(output)
            self.image_viewer.setImage(updatedPixmap, True, "White Balance")

        self.WhiteBalanceToolButton.setChecked(False)
        del tool
        tool = None

    def OnWhiteBalanceToolButton(self, checked):
        if  self.image_viewer._isRectangleSelectPreseed:
            self.image_viewer.clearImage()
            self.image_viewer.open(self.image_viewer._current_filename)
            self.image_viewer._isRectangleSelectPreseed = False
        if checked:
            self.InitTool()
            currentPixmap = self.getCurrentLayerLatestPixmap()
            image = self.QPixmapToImage(currentPixmap)

            from QToolWhiteBalance import QToolWhiteBalance
            widget = QToolWhiteBalance(None, image, self.onWhiteBalanceCompleted)
            widget.show()

    def OnApplyToolButton(self,checked):
        self.image_viewer.apply_merge()

    def OnInstagramFiltersToolButton(self, checked):
        if checked:
            self.InitTool()

            class QInstagramToolDockWidget(QtWidgets.QDockWidget):
                def __init__(self, parent, mainWindow):
                    QtWidgets.QDockWidget.__init__(self, parent)
                    self.parent = parent
                    self.closed = False
                    self.mainWindow = mainWindow
                    self.setWindowTitle("Filters")

                def closeEvent(self, event):
                    self.destroyed.emit()
                    event.accept()
                    self.closed = True
                    self.mainWindow.InstagramFiltersToolButton.setChecked(False)
                    self.mainWindow.image_viewer.setImage(self.mainWindow.image_viewer.pixmap(), True,
                                                          "Instagram Filters")

            self.EnableTool("instagram_filters") if checked else self.DisableTool("instagram_filters")
            currentPixmap = self.getCurrentLayerLatestPixmap()
            image = self.QPixmapToImage(currentPixmap)

            from QToolInstagramFilters import QToolInstagramFilters
            tool = QToolInstagramFilters(self, image)
            self.filtersDock = QInstagramToolDockWidget(None, self)
            self.filtersDock.setWidget(tool)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.filtersDock)

            widget = self.filtersDock

            widget.show()

            # Create a local event loop for this widget
            loop = QtCore.QEventLoop()
            self.filtersDock.destroyed.connect(loop.quit)
            tool.destroyed.connect(loop.quit)
            loop.exec()  # wait
        else:
            self.DisableTool("instagram_filters")
            self.filtersDock.hide()

    def OnEraserToolButton(self, checked):
        if  self.image_viewer._isRectangleSelectPreseed:
            self.image_viewer.clearImage()
            self.image_viewer.open(self.image_viewer._current_filename)
            self.image_viewer._isRectangleSelectPreseed = False
        if checked:
            self.InitTool()
            self.EnableTool("eraser") if checked else self.DisableTool("eraser")

    def EnableTool(self, tool):
        for key, value in self.tools.items():
            if key == tool:
                button = getattr(self, value["tool"])
                button.setChecked(True)
                self.setToolButtonStyleChecked(button)
                setattr(self.image_viewer, value["var"], True)
            else:
                # Disable the other tools
                button = getattr(self, value["tool"])
                button.setChecked(False)
                self.setToolButtonStyleUnchecked(button)
                setattr(self.image_viewer, value["var"], False)
                if "destructor" in value:
                    getattr(self.image_viewer, value["destructor"])()

    def DisableTool(self, tool):
        value = self.tools[tool]
        button = getattr(self, value["tool"])
        button.setChecked(False)
        self.setToolButtonStyleUnchecked(button)
        setattr(self.image_viewer, value["var"], False)
        if "destructor" in value:
            getattr(self.image_viewer, value["destructor"])()

        if tool in ["blur", "spot_removal"]:
            # The cursor overlay is being rendered in the view
            # Remove it
            pixmap = self.getCurrentLayerLatestPixmap()
            self.image_viewer.setImage(pixmap, False)

    def DisableAllTools(self):
        for _, value in self.tools.items():
            getattr(self, value["tool"]).setChecked(False)
            setattr(self.image_viewer, value["var"], False)
            if "destructor" in value:
                getattr(self.image_viewer, value["destructor"])()

    def updateHistogram(self):
        # Update Histogram

        # Compute image histogram
        img = self.QPixmapToImage(self.getCurrentLayerLatestPixmap())
        r, g, b, a = img.split()
        r_histogram = r.histogram()
        g_histogram = g.histogram()
        b_histogram = b.histogram()

        # ITU-R 601-2 luma transform:
        luma_histogram = [sum(x) for x in zip([item * float(299 / 1000) for item in r_histogram],
                                              [item * float(587 / 1000) for item in g_histogram],
                                              [item * float(114 / 1000) for item in b_histogram])]

        # Create histogram plot
        x = list(range(len(r_histogram)))
        self.ImageHistogramPlot.removeItem(self.ImageHistogramGraphRed)
        self.ImageHistogramPlot.removeItem(self.ImageHistogramGraphGreen)
        self.ImageHistogramPlot.removeItem(self.ImageHistogramGraphBlue)
        self.ImageHistogramPlot.removeItem(self.ImageHistogramGraphLuma)
        self.ImageHistogramGraphRed = pg.PlotCurveItem(x=x, y=r_histogram, fillLevel=2, width=1.0,
                                                       brush=(255, 0, 0, 80))
        self.ImageHistogramGraphGreen = pg.PlotCurveItem(x=x, y=g_histogram, fillLevel=2, width=1.0,
                                                         brush=(0, 255, 0, 80))
        self.ImageHistogramGraphBlue = pg.PlotCurveItem(x=x, y=b_histogram, fillLevel=2, width=1.0,
                                                        brush=(0, 0, 255, 80))
        self.ImageHistogramGraphLuma = pg.PlotCurveItem(x=x, y=luma_histogram, fillLevel=2, width=1.0,
                                                        brush=(255, 255, 255, 80))
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphRed)
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphGreen)
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphBlue)
        self.ImageHistogramPlot.addItem(self.ImageHistogramGraphLuma)

    def OnOpen(self):
        # Load an image file to be displayed (will popup a file dialog).
        self.image_viewer.numLayersCreated = 1
        self.image_viewer.currentLayer = 0
        self.image_viewer.layerHistory = {
            0: []
        }
        self.image_viewer.open()
        if self.image_viewer._current_filename != None:
            size = self.image_viewer.currentPixmapSize()
            if size:
                w, h = size.width(), size.height()
                self.statusBar.showMessage(str(w) + "x" + str(h))
            self.InitTool()
            self.DisableAllTools()
            filename = self.image_viewer._current_filename
            filename = os.path.basename(filename)
            # self.image_viewer.OriginalImage = self.image_viewer.pixmap()
            self.updateHistogram()
            self.createLayersDock()
            for button in self.ToolButtons:
                button.setEnabled(True)

    def createLayersDock(self):
        if self.layerListDock:
            self.removeDockWidget(self.layerListDock)
            self.layerListDock.layerButtons = []
            self.layerListDock.numLayers = 1
            self.layerListDock.currentLayer = 0

        from QLayerList import QLayerList
        self.layerListDock = QLayerList("Layers", self)
        self.layerListDock.setTitleBarWidget(QtWidgets.QWidget())
        self.layerListDock.setFixedWidth(170)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.layerListDock)
        self.image_viewer.layerListDock = self.layerListDock

    def OnSave(self):
        if self.image_viewer._current_filename.lower().endswith(".nef"):
            # Cannot save pixmap as .NEF (yet)
            # so open SaveAs menu to export as PNG instead
            self.OnSaveAs()
        else:
            self.image_viewer.save()

    def OnSaveAs(self):
        name, ext = os.path.splitext(self.image_viewer._current_filename)
        dialog = QFileDialog()
        dialog.setDefaultSuffix("png")
        extension_filter = "Default (*.png);;BMP (*.bmp);;Icon (*.ico);;JPEG (*.jpeg *.jpg);;PBM (*.pbm);;PGM (*.pgm);;PNG (*.png);;PPM (*.ppm);;TIF (*.tif *.tiff);;WBMP (*.wbmp);;XBM (*.xbm);;XPM (*.xpm)"
        name = dialog.getSaveFileName(self, 'Save File', name + ".png", extension_filter)
        self.image_viewer.save(name[0])
        filename = self.image_viewer._current_filename
        filename = os.path.basename(filename)

    def OnUndo(self):
        self.image_viewer.undoCurrentLayerLatestChange()

    def OnPaste(self):
        cb = QApplication.clipboard()
        md = cb.mimeData()
        if md.hasImage():
            img = cb.image()
            self.initImageViewer()
            self.image_viewer._current_filename = "Untitled.png"
            self.image_viewer.setImage(img, True, "Paste")
            filename = self.image_viewer._current_filename
            # self.image_viewer.OriginalImage = self.image_viewer.pixmap()

            self.updateHistogram()
            self.createLayersDock()
            for button in self.ToolButtons:
                button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    gui = Gui()
    app.setStyleSheet('''
    QWidget {
        background-color: rgb(44, 44, 44);
        color: white;
    }
    QMainWindow { 
        background-color: rgb(44, 44, 44); 
    }
    QGraphicsView { 
        background-color: rgb(22, 22, 22); 
    }
    QDockWidget { 
        background-color: rgb(44, 44, 44); 
    }
    QToolButton {
        border: none;
        color: white;
        background-color: rgb(44, 44, 44);
    }
    QToolButton:pressed {
        border-width: 1px;
        border-color: rgb(22, 22, 22);
        background-color: rgb(22, 22, 22);
        border-style: solid;
    }
    QPushButton {
        border: none;
        color: white;
        background-color: rgb(44, 44, 44);
    }
    QPushButton:pressed {
        border-width: 1px;
        border-color: rgb(22, 22, 22);
        background-color: rgb(22, 22, 22);
        border-style: solid;
    }
    QLabel {
        background-color: rgb(22, 22, 22);
        color: white;
    }
    ''');
    app.setWindowIcon(QtGui.QIcon("icons/logo.png"))
    sys.exit(app.exec())


if __name__ == '__main__':
    # https://stackoverflow.com/questions/71458968/pyqt6-how-to-set-allocation-limit-in-qimagereader
    os.environ['QT_IMAGEIO_MAXALLOC'] = "1024"
    # QtGui.QImageReader.setAllocationLimit(0)

    main()