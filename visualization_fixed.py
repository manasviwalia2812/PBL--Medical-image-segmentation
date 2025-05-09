import torch

from torch.utils.data import DataLoader

from unet_model import UNet

from dataset import BrainTumor2DSliceDataset

import matplotlib.pyplot as plt

import numpy as np



# Corrected validation paths (removed leading slash)

val_images_path = "Brats_2D_Unet_Project/data/2d/val/images"

val_masks_path = "Brats_2D_Unet_Project/data/2d/val/masks"



# Load validation dataset

val_dataset = BrainTumor2DSliceDataset(val_images_path, val_masks_path)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



# Load model with correct input/output channels to match checkpoint

model = UNet(in_channels=3, out_channels=1)

model.load_state_dict(torch.load("unet_model_brats.pth", map_location=torch.device('cpu')))

model.eval()



# Show one prediction

images, masks = next(iter(val_loader))

with torch.no_grad():

    preds = model(images)

    preds = torch.sigmoid(preds)

    preds = (preds > 0.5).float()



# Plot original image, ground truth mask, predicted mask

fig, axs = plt.subplots(1, 3, figsize=(15, 5))



axs[0].imshow(images[0][0].numpy(), cmap='gray')  # First channel for visualization

axs[0].set_title("Image (channel 0)")



axs[1].imshow(masks[0][0].numpy(), cmap='gray')

axs[1].set_title("Ground Truth Mask")



# For predicted mask, show channel 0 (or adjust as needed)

axs[2].imshow(preds[0][0].numpy(), cmap='gray')

axs[2].set_title("Predicted Mask (channel 0)")



for ax in axs:

    ax.axis('off')



plt.tight_layout()

plt.show()
