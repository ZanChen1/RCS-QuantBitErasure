This repository contains the official implementation of our paper "Robust Compressive Sensing lmaging forQuantization Bit Erasure".

## Pretrained models

This repository does **not** ship large pretrained weights.  
Please download the following models manually and place them at the
indicated paths under `Trained_Weights/`:

- `Trained_Weights/Restormer/gaussian_gray_denoising_blind.pth`  
  Download the gray Gaussian denoising Restormer model from HuggingFace  
  (file **`gaussian_gray_denoising_blind.pth`** in the `deepinv/Restormer` repo):  
  https://huggingface.co/deepinv/Restormer/tree/main

- `Trained_Weights/DPIR/drunet_gray.pth`  
  Download the gray DRUNet model used in DPIR from HuggingFace  
  (file **`drunet_gray.pth`** in the `deepinv/drunet` repo):  
  https://huggingface.co/deepinv/drunet/tree/main

- `Trained_Weights/MWCNN/MWCNN_150_300.pth`  
- `Trained_Weights/MWCNN/MWCNN_300_500.pth`  
- `Trained_Weights/MWCNN/MWCNN_500_1000.pth`  
  These MWCNN models are trained following the original MWCNN paper and
  are not hosted in this repository due to GitHub’s file size limit.  
  **Please download them from the authors’ shared storage and update the links below:**

  - `MWCNN_150_300.pth`: TODO – your download link here  
  - `MWCNN_300_500.pth`: TODO – your download link here  
  - `MWCNN_500_1000.pth`: TODO – your download link here
