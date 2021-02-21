from matplotlib import pyplot as plt 
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch
import os
from model import *


WORKING_DIR = "/content/drive/MyDrive/Projet_AMAL/"
IMAGES_DIR = os.path.join(WORKING_DIR, "images")

ENCODER_PATH = os.path.join(WORKING_DIR, "encoder.cpkt")
DECODER_PATH = os.path.join(WORKING_DIR, "decoder.cpkt")

# Hyperparameters according to the paper

NB_EPOCHS = 10
LEARNING_RATE = 0.01
RHO = 1
SIGMA = 5
MU = 0.5
LITTLE_M = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_x(x_hat, y, p, q):

    numerator = x_hat / (SIGMA**2) + RHO * p - q
    denominator = RHO + (1 / SIGMA**2)

    return numerator / denominator

def criterion(x, y, x_hat, y_hat):
    
    return ((y_hat - y)**2).sum() / 2 + ((x_hat - x)**2).sum() / (2 * SIGMA**2)


def denoise(y, y_real, gaussian_denoiser="nlm"):
    """
        y : np array of size (512, 512, 3)
    """
    list_psnr = []
    list_ssim = []
    list_loss = []


    y = torch.Tensor(y).permute(2, 0, 1).unsqueeze(0).to(device)
    
    x = y
    x = x.to(device)
    
    p = y
    p = p.to(device)

    q = torch.zeros_like(y).to(device)

    encoder = U_Net().to(device)
    decoder = U_Net().to(device)

    optimizer = torch.optim.Adam(params=list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    
    for i in range(NB_EPOCHS):
        mean_psnr = 0
        mean_ssim = 0
        mean_loss = 0

        for j in range(LITTLE_M):

            optimizer.zero_grad()
            
            x_hat = encoder(y)
            
            epsilon = torch.randn_like(x_hat).to(device) * 0.03
            y_hat = decoder(x_hat + epsilon)
            
            loss = criterion(x, y, x_hat, y_hat)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                
                y_hat = encoder(y)
                x = update_x(x_hat, y, p, q)

                mean_psnr += psnr(y_real, x.squeeze(0).permute(1, 2, 0).cpu().numpy()) / LITTLE_M
                mean_ssim += ssim(y_real, x.squeeze(0).permute(1, 2, 0).cpu().numpy(), multichannel=True) / LITTLE_M
                mean_loss += loss / LITTLE_M
                
                print(f"EPOCH {j}")
                
                print(f"\tLOSS\t{loss}")
                print(f"\tPSNR\t{psnr(y_real, x.squeeze(0).permute(1, 2, 0).cpu().numpy())}")
                print(f"\tSSIM\t{ssim(y_real, x.squeeze(0).permute(1, 2, 0).cpu().numpy(), multichannel=True)}")


        # outside nested loop
        with torch.no_grad():
          # use white gaussian denoiser to update p
          if gaussian_denoiser == "nlm":
            #sigma_est = np.mean(estimate_sigma(x.view(3, 512, 512).permute(1, 2, 0), multichannel=True))
            p = torch.Tensor(
                    denoise_nl_means(x.view(3, 512, 512).permute(1, 2, 0).cpu().numpy() 
                    + q.view(3, 512, 512).permute(1, 2, 0).cpu().numpy() / RHO,
                    multichannel=True)
                ).to(device).permute(2, 0, 1).unsqueeze(0)

          elif gaussian_denoiser == "bm3d":
            p = torch.Tensor(bm3d.bm3d_rgb(x.view(3, 512, 512).permute(1, 2, 0).cpu().numpy(), sigma_psd= 30/255)).to(device).permute(2, 0, 1).unsqueeze(0)
            
          q = q + MU * RHO * (x - p)

        # saving models weights
        torch.save(encoder.state_dict(), ENCODER_PATH)
        torch.save(decoder.state_dict(), DECODER_PATH)


        # showing metrics
        print(f"EPOCH {i}")
        print(f"\tLOSS\t{mean_loss}")
        print(f"\tPSNR\t{mean_psnr}")
        print(f"\tSSIM\t{mean_ssim}")

        # saving metrics
        list_psnr.append(mean_psnr)
        list_ssim.append(mean_ssim)
        list_loss.append(mean_loss)


        # display image
        with torch.no_grad():
            plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
            plt.show()

        # progress
        if i % 5 == 0:
            print(f"completed {i} epochs")

    return x, list_psnr, list_ssim, list_loss
