import torch
from train import *
from matplotlib import image
from matplotlib import pyplot as plt 

IMAGE_PATH = os.path.join(IMAGES_DIR, "d800_iso6400_3_noisy.png")
REAL_IMAGE_PATH = os.path.join(IMAGES_DIR, "d800_iso6400_3_mean.png")

# load images
y_noisy = image.imread(IMAGE_PATH)
y_real = image.imread(REAL_IMAGE_PATH)

x, _, _, _ = denoise(y_noisy, y_real)
