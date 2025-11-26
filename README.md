This repository contains the official implementation of our paper "Robust Compressive Sensing lmaging forQuantization Bit Erasure".

## Main script and how to run

The main entry point for the experiments is:

- `main/main.m`

To run the default experiment:

1. Make sure
   - MATLAB can find this repository (current folder set to the repo root),
   - the Python environment is configured as described above (via `pyenv`),
   - all pretrained weights have been downloaded and placed under `Trained_Weights/` in the correct subfolders.
   
2. In MATLAB, run:
  clear; close all;
  main

## Python environment

The Matlab–Python bridge (Restormer, MWCNN, DPIR, SigCNN) was tested with
the following Python environment:

- Python 3.7.16
- PyTorch 1.12.0+cu116
- torchvision 0.13.0+cu116 (installed together with PyTorch)
- numpy 1.21.6
- opencv-python 4.8.1.78
- einops 0.6.1
- PyYAML 6.0
- basicsr (>= 1.4.2, as used in the original Restormer implementation)

All other Python modules used in the code,
such as `MWCNN_Nested_2group`, `models.network_unet`,
`DPIR_utils`, and `SigCNN_all`,
are part of this repository and do not need to be installed separately.

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
  These MWCNN models are the ones used in our paper

> Zan Chen, Tao Wang, Jun Li, Wenlong Guo, Yuanjing Feng,  
> Xueming Qian, and Xingsong Hou.  
> **Discard Significant Bits of Compressed Sensing: A Robust Image Coding for Resource-Limited Contexts.**  
> ACM Trans. Multimedia Comput. Commun. Appl. 21, 1, Article 31 (January 2025), 25 pages. 
  **Please download them from the authors’ shared storage and update the links below:**
  https://drive.google.com/drive/folders/1T5yvuDCToA_NU11GLnZKoVaXq2at7fSL?usp=drive_link
