# Medical Image Segmentation - Brain Tumor Segmentation Using 2D UNet
Colab Code link- https://drive.google.com/drive/folders/1n4ZgcsD-V6ZIjcdS0WQtXKG2ocmxq7Rt?usp=drive_link
report research paper: [Demo Vid Streamlit Link- https://drive.google.com/file/d/1boSZCLsXuYVZhWxafXd-ouKtVYLfhrmB/view?usp=drive_link](https://drive.google.com/drive/folders/1z9ulmThNSG881swJOSB3-8MjevbM532B?usp=sharing)
Presentation link- https://docs.google.com/presentation/d/1td_aMhVHhoQlM3EeFJ6NasTR2-Y6lbK5/edit?usp=sharing&ouid=112896692540103231027&rtpof=true&sd=true

## Project Description
This project implements a 2D UNet model for brain tumor segmentation on MRI slices. Brain tumor segmentation is a critical task in medical image analysis, aiding diagnosis, treatment planning, and monitoring. Manual segmentation is time-consuming and prone to variability, motivating automated methods. Deep learning, particularly convolutional neural networks (CNNs), has revolutionized medical image segmentation.

The UNet architecture, introduced by Ronneberger et al. (2015), is a widely adopted CNN model for biomedical image segmentation. It features an encoder-decoder structure with skip connections, enabling precise localization and context capture. This project applies a 2D UNet model to segment brain tumors from MRI slices using the BraTS dataset.

## Features
- Data preprocessing of 3D MRI volumes into 2D slices for training and validation.
- Custom PyTorch Dataset class for loading 2D slices and masks.
- Implementation of the UNet architecture with double convolution blocks, downsampling, upsampling, and skip connections.
- Training and validation using cross-entropy loss and Adam optimizer.
- Interactive Flask web application frontend for visualizing input images, ground truth masks, and predicted masks.

## Installation

### Prerequisites
- Python 3.7 or higher
- PyTorch
- NumPy
- tqdm
- Flask
- Matplotlib

### Installing dependencies
You can install the required Python packages using pip:

```bash
pip install torch numpy tqdm flask matplotlib
```

## Dataset
The project uses the BraTS (Brain Tumor Segmentation) dataset, which provides multi-modal MRI scans with expert annotations. The 3D MRI volumes and corresponding tumor masks are preprocessed by slicing along the z-axis into 2D slices (128x128 pixels with 4 channels) and saved as `.npy` files for training and validation.

## Usage

### Data Preprocessing
Run the `preprocess_slices.py` script to convert 3D MRI volumes into 2D slices suitable for training.

### Training
The training process is implemented in the `training.ipynb` notebook. It sets up the dataset, dataloaders, model, loss function, and optimizer, and trains the UNet model for 10 epochs. The trained model weights are saved as `unet_model_brats.pth`.

### Visualization
The Flask web application located in the `webapp/` directory provides an interactive frontend where users can select slice indices to visualize the input image, ground truth mask, and predicted mask. The frontend uses matplotlib to convert tensor images to base64-encoded PNGs for display.

To run the webapp, execute:

```bash
python webapp/app_fixed.py
```

Then open your browser and navigate to `http://localhost:5000`.

## Model Architecture
The UNet model consists of an encoder-decoder structure with skip connections. It uses double convolution blocks, max-pooling for downsampling, and transposed convolutions for upsampling. The model takes 4-channel input images and outputs 4-channel segmentation masks corresponding to different tumor classes.

## Results and Analysis
The project demonstrates the feasibility of 2D UNet for brain tumor segmentation on MRI slices. The model effectively segments tumor regions in many slices, capturing shape and location consistent with ground truth. Some slices may show false positives or missed tumor areas, indicating room for improvement.

The slicing approach simplifies 3D data handling but may lose inter-slice contextual information, suggesting potential benefits from 3D or hybrid models. The interactive frontend aids in understanding model behavior and supports iterative refinement.

## Folder Structure
```
.
├── Brain_Tumor_Segmentation_Report.md
├── dataset_fixed.py
├── preprocess_slices.py
├── training.ipynb
├── unet_model.py
├── unet_model_brats.pth
├── visualization_fixed.py
├── visualization.ipynb
├── webapp/
│   ├── app_fixed.py
│   ├── static/
│   └── templates/
└── Brats_2D_Unet_Project/
    └── data/
```

## References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
- BraTS Dataset: https://www.med.upenn.edu/cbica/brats2020/data.html




