import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, redirect, url_for
import torch
from torch.utils.data import DataLoader
from unet_model import UNet
from dataset_fixed import BrainTumor2DSliceDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

app = Flask(__name__)

# Paths for validation data
VAL_IMAGES_PATH = "Brats_2D_Unet_Project/data/2d/val/images"
VAL_MASKS_PATH = "Brats_2D_Unet_Project/data/2d/val/masks"

# Load dataset
val_dataset = BrainTumor2DSliceDataset(VAL_IMAGES_PATH, VAL_MASKS_PATH)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("unet_model_brats.pth", map_location=torch.device('cpu')))
model.eval()

def tensor_to_base64_img(tensor, cmap='gray', vmin=None, vmax=None):
    """Convert a torch tensor image to base64 encoded PNG for HTML display."""
    np_img = tensor.cpu().numpy()
    if np_img.ndim == 3:
        np_img = np_img[0]  # For single channel images
    plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(np_img, cmap=cmap, vmin=vmin, vmax=vmax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_bytes = buf.getvalue()
    base64_img = base64.b64encode(img_bytes).decode('utf-8')
    return base64_img

@app.route('/', methods=['GET', 'POST'])
def index():
    description = """
    This project implements a UNet model for brain tumor segmentation on 2D MRI slices.
    The model takes 3-channel input images and outputs a binary mask predicting tumor regions.
    """
    loss = "Binary Cross Entropy (BCE) Loss (placeholder)"
    accuracy = "Accuracy: 85% (placeholder)"

    selected_index = 0
    if request.method == 'POST':
        selected_index = int(request.form.get('slice_index', 0))
        if selected_index < 0 or selected_index >= len(val_dataset):
            selected_index = 0

    # Get image and mask tensors
    image, mask = val_dataset[selected_index]
    image_batch = image.unsqueeze(0)
    with torch.no_grad():
        pred = model(image_batch)
        pred = torch.sigmoid(pred)
        pred_thresh = (pred > 0.5).float()

    # Binarize ground truth mask
    mask_bin = (mask > 0).float()

    # Convert tensors to base64 images
    image_b64 = tensor_to_base64_img(image[0], cmap='gray', vmin=0, vmax=1)
    mask_b64 = tensor_to_base64_img(mask_bin, cmap='gray', vmin=0, vmax=1)
    pred_b64 = tensor_to_base64_img(pred_thresh[0][0], cmap='gray', vmin=0, vmax=1)

    return render_template('index.html',
                           description=description,
                           loss=loss,
                           accuracy=accuracy,
                           image=image_b64,
                           mask=mask_b64,
                           prediction=pred_b64,
                           total_slices=len(val_dataset),
                           selected_index=selected_index)

if __name__ == '__main__':
    app.run(debug=True)
