from matplotlib import pyplot as plt 
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from model import *

WORKING_DIR = "."
IMAGES_DIR = os.path.join(WORKING_DIR, "images")
GEN_IMAGES_DIR = os.path.join(WORKING_DIR, "generated_images")

ENCODER_PATH = os.path.join(WORKING_DIR, "encoder.cpkt")
DECODER_PATH = os.path.join(WORKING_DIR, "decoder.cpkt")

# Hyperparameters according to the paper
NB_EPOCHS = 500
LEARNING_RATE = 0.01
RHO = 1
SIGMA = 5
MU = 0.5

# you may need to adjust these parameters for some images, especially with BM3D
BIG_M = NB_EPOCHS // 100
LITTLE_M = NB_EPOCHS // BIG_M

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_x(x_hat, y, p, q):
    # update the X with fixed VAE parameters
    numerator = x_hat / (SIGMA**2) + RHO * p - q
    denominator = RHO + (1 / SIGMA**2)

    return numerator / denominator

def criterion(x, y, x_hat, y_hat):
    # only compute f + R since the rest doesn't influence the gradient with respect to VAE parameters
    return ((y_hat - y)**2).sum() / 2 + ((x_hat - x)**2).sum() / (2 * SIGMA**2)


def denoise(y, y_real, gaussian_denoiser="nlm", verbose=False, interval=5):
    """
        y : np array of size (512, 512, 3) representing the noisy image.
        y_real : np array of size (512, 512, 3) representing the denoised image (only used to compute PSNR and SSIM)
        gaussian_denoiser : AWGD used to denoise image in latent space
        verbose : if True, prints result in console, else doesn't
        interval : number of updates between two computes of PSNR and SSIM
    """
    list_psnr = []
    list_ssim = []
    list_loss = []

    best_image_psnr = 0
    best_image_ssim = 0

    y = torch.Tensor(y).permute(2, 0, 1).unsqueeze(0).to(device)
    
    x = y
    x = x.to(device)
    
    p = y
    p = p.to(device)

    q = torch.zeros_like(y).to(device)

    encoder = U_Net().to(device)
    decoder = U_Net().to(device)

    optimizer = torch.optim.Adam(params=list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    
    # for tensorboard
    writer = SummaryWriter()

    for i in range(BIG_M):
        mean_psnr = 0
        mean_ssim = 0
        mean_loss = 0

        for j in range(LITTLE_M):

            optimizer.zero_grad()
            
            x_hat = encoder(y)
            
            epsilon = torch.randn_like(x_hat).to(device) / 255 
            y_hat = decoder(x_hat + epsilon)
            
            loss = criterion(x, y, x_hat, y_hat)
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                
                x_hat = encoder(y)
                x = update_x(x_hat, y, p, q)

                if i * LITTLE_M + j % interval == 0:
                  actual_psnr = psnr(y_real, x.squeeze(0).permute(1, 2, 0).cpu().numpy())
                  actual_ssim = ssim(y_real, x.squeeze(0).permute(1, 2, 0).cpu().numpy(), multichannel=True) 

                  mean_psnr += actual_psnr / LITTLE_M
                  mean_ssim += actual_ssim / LITTLE_M
                  mean_loss += loss / LITTLE_M
                  
                  writer.add_scalar('Loss', loss, i * LITTLE_M + j)
                  writer.add_scalar('PSNR', actual_psnr, i * LITTLE_M + j)
                  writer.add_scalar('SSIM', actual_ssim, i * LITTLE_M + j)

                  list_psnr.append(actual_psnr)
                  list_ssim.append(actual_ssim)
                  list_loss.append(loss)
                  
                  if actual_psnr > best_image_psnr:
                    best_image = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    best_image_psnr = actual_psnr
                    best_image_ssim = actual_ssim

                  if verbose :
                    print(f"EPOCH {i * LITTLE_M + j}")
                  
                    print(f"\tLOSS\t{loss}")
                    print(f"\tPSNR\t{actual_psnr}")
                    print(f"\tSSIM\t{actual_ssim}")


        # outside nested loop
        with torch.no_grad():

          # use white gaussian denoiser to update p
          if gaussian_denoiser == "nlm":
            sigma_est = np.mean(
                estimate_sigma(x.view(3, 512, 512).permute(1, 2, 0).cpu().numpy()
                + q.view(3, 512, 512).permute(1, 2, 0).cpu().numpy() / RHO, 
                multichannel=True))
            p = torch.Tensor(
                    denoise_nl_means(x.view(3, 512, 512).permute(1, 2, 0).cpu().numpy() 
                    + q.view(3, 512, 512).permute(1, 2, 0).cpu().numpy() / RHO,
                    h=0.6 * sigma_est, 
                    sigma=sigma_est,
                    multichannel=True)
                ).to(device).permute(2, 0, 1).unsqueeze(0)

          elif gaussian_denoiser == "bm3d":
            p = torch.Tensor(
                bm3d.bm3d_rgb(x.view(3, 512, 512).permute(1, 2, 0).cpu().numpy() 
                + q.view(3, 512, 512).permute(1, 2, 0).cpu().numpy() / RHO, sigma_psd= 30/255)).to(device).permute(2, 0, 1).unsqueeze(0)
            
          
          q = q + MU * RHO * (x - p) / 255
            

        # saving models weights
        torch.save(encoder.state_dict(), ENCODER_PATH)
        torch.save(decoder.state_dict(), DECODER_PATH)

        # display image
        with torch.no_grad():
            plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().numpy())
            plt.show()

        # progress
        if verbose:
            print(f"completed {i} global passes")

    # saving best image in file
    if gaussian_denoiser == "nlm":
      plt.imsave(os.path.join(GEN_IMAGES_DIR, f"NN+NLM ({best_image_psnr:.2f} - {best_image_ssim:.4f}).png"), np.clip(best_image, 0., 1.))

    elif gaussian_denoiser == "bm3d":
      plt.imsave(os.path.join(GEN_IMAGES_DIR, f"NN+BM3D ({best_image_psnr:.2f} - {best_image_ssim:.4f}).png"), np.clip(best_image, 0., 1.))

    return {"best_image" : best_image, "psnr": best_image_psnr, "ssim": best_image_ssim,"list_psnr": list_psnr, "list_ssim" : list_ssim, "list_loss": list_loss}
