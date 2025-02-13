
## Quick Start

Download the requirements file first inside the PhotoLab, and also download some extra packages:
Python 3.12
pip install requirement
```console

pip install PyQt6
pip install pyqtgraph
pip install panda3d
```

Download the pretrained models by running the included download script:

```console
foo:bar$ python download_models.py
```

Start the editor by running:

```console
foo:bar$ python src/main.py
```

If you face any issue relate to "qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem", here is several methods you could try to figure it out:

First, try reinstall PyQt6 PyQt6-Qt6 packages:

```console
pip install --force-reinstall PyQt6 PyQt6-Qt6
```

Second, set the export route for plugin:

```console
export QT_PLUGIN_PATH="/Users/liqi/anaconda3/lib/python3.12/site-packages/PyQt6/Qt6/plugins"
```

After all these steps, try rerun the "python src/main.py" in your terminal.

## Features
- NIMA Score (Need to Impleemnt)
- Instagram Filters
- Super-Resolution
- Human Segmentation (Need to Intergrate)
- Inpainting (Need to Implement)


