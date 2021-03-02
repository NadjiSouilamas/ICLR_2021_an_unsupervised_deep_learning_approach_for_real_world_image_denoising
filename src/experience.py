import torch
from train import *
from matplotlib import image
from matplotlib import pyplot as plt 
import os

IMAGE_PATH = os.path.join(IMAGES_DIR, "d800_iso6400_3_real.png")
REAL_IMAGE_PATH = os.path.join(IMAGES_DIR, "d800_iso6400_3_mean.png")

# load images
y_noisy = image.imread(IMAGE_PATH)
y_real = image.imread(REAL_IMAGE_PATH)

# compute PSNR and SSIM before denoising
ssim_noisy = ssim(y_real, y_noisy, multichannel=True) 
psnr_noisy = psnr(y_real, y_noisy)

print(f"noisy PSNR \t{psnr_noisy}")
print(f"noisy SSIM \t{ssim_noisy}")

# Choose your Gaussian Denoiser mode
gaussian_denoiser = "nlm"

# Denoise
hist_1 = denoise(y_noisy, y_real, gaussian_denoiser=gaussian_denoiser, verbose=True)