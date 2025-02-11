import os
import subprocess
import cv2
import numpy as np
from gi.repository import Gimp, GObject, GLib


def external_ai_segmentation(input_path, output_mask_path):
    """Run AI segmentation externally using a relative path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current plugin directory
    ai_script_path = os.path.join(script_dir, "selfie_segment_selection.py")  # Relative path to AI model

    command = ["python", ai_script_path, input_path, output_mask_path]
    subprocess.run(command, check=True)


def load_selection_mask(image, mask_path):
    """Load the selection mask into GIMP."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Error loading segmentation mask.")

    width, height = image.width, image.height
    mask = cv2.resize(mask, (width, height))  # Resize mask to match image size

    # Convert to GIMP selection
    pdb = Gimp.get_pdb()
    selection = pdb.run_procedure("gimp_selection_layer_alpha", [image])

    # Apply mask as selection
    region = image.get_active_layer().get_pixel_rgn(0, 0, width, height, True, False)
    region[:, :] = mask  # Apply mask
    image.get_active_layer().flush()
    image.get_active_layer().update(0, 0, width, height)


def plugin_main(procedure, run_mode, image, drawable, args, data):
    """Main function for the GIMP AI selection plugin."""
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the plugin's directory
    input_path = os.path.join(script_dir, "gimp_input.png")  # Temporary input image
    output_mask_path = os.path.join(script_dir, "gimp_mask.png")  # Temporary output mask

    # Export active layer
    pdb = Gimp.get_pdb()
    pdb.run_procedure("file-png-save", [run_mode, image, drawable, input_path, input_path, 0, 9, 1, 1, 1, 1, 1])

    # Run AI segmentation
    external_ai_segmentation(input_path, output_mask_path)

    # Load selection mask into GIMP
    load_selection_mask(image, output_mask_path)

    # Cleanup
    os.remove(input_path)
    os.remove(output_mask_path)

    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


# Register the plugin in GIMP
Gimp.main(
    Gimp.Procedure.new(
        "python-fu-ai-selfie-selection",  # Internal unique ID
        Gimp.PDBProcType.PLUGIN,
        "AI-based selfie selection",
        "Runs AI segmentation to select humans in the image",
        "Your Name",
        "Your Name",
        "2025",
        "_AI Selfie Selection...",  # This is what appears in GIMP's menu
        "RGB*, GRAY*",
        Gimp.PDBType.IMAGE | Gimp.PDBType.DRAWABLE
    ).set_attribution(plugin_main)
)