import os
import numpy as np

def save_2d_slices(src_img_dir, src_mask_dir, dest_img_dir, dest_mask_dir):
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)

    files = sorted(os.listdir(src_img_dir))

    for f in files:
        if not f.endswith('.npy'):
            continue

        image_3d = np.load(os.path.join(src_img_dir, f))   # Shape: (128, 128, 128, 4)
        mask_3d = np.load(os.path.join(src_mask_dir, f))   # Shape: (128, 128, 128)

        for i in range(image_3d.shape[2]):  # slicing along the z-axis (128 slices)
            img_slice = image_3d[:, :, i, :]         # Shape: (128, 128, 4)
            mask_slice = mask_3d[:, :, i]            # Shape: (128, 128)

            slice_name = f.replace('.npy', f'_slice_{i:03d}.npy')
            np.save(os.path.join(dest_img_dir, slice_name), img_slice)
            np.save(os.path.join(dest_mask_dir, slice_name), mask_slice)

    print(f"✅ Finished slicing. Total 2D slices per volume: {image_3d.shape[2]}")

# ✅ Replace this with the actual correct path in your project
save_2d_slices(
    src_img_dir="Brats_2D_Unet_Project/data/train/images",
    src_mask_dir="Brats_2D_Unet_Project/data/train/masks",
    dest_img_dir="Brats_2D_Unet_Project/data/2d/train/images",
    dest_mask_dir="Brats_2D_Unet_Project/data/2d/train/masks"
)

save_2d_slices(
    src_img_dir="Brats_2D_Unet_Project/data/val/images",
    src_mask_dir="Brats_2D_Unet_Project/data/val/masks",
    dest_img_dir="Brats_2D_Unet_Project/data/2d/val/images",
    dest_mask_dir="Brats_2D_Unet_Project/data/2d/val/masks"
)
