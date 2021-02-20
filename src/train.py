import torch
import model


"""
We will need the following  :
    1. Two Unets for encoding and decoding
    2. A denoiser (We'll begin with one of openCV)
    3. Estimating noise level with the method of the paper mentionned for that
    
"""
# Hyperparameters according to the paper

NB_EPOCHS = 500
RHO = 1
SIGMA = 5
MU = 0.5
LITTLE_M = 10


def update_x(encoder, y, p, q):
    numerator = (1 / SIGMA ** 2) * encoder(y) + RHO * p - q
    denominator = RHO + (1 / SIGMA ** 2)

    return numerator / denominator


def denoising_algo(y, rho, sigma, mu, gaussianDenoiser):
    """
      Still need :
        1. Define Adam with both encoder and decoder hyperparams
        2. Use a gaussian denoiser
        3. Think of how I can train the networks without wasting time
            on stupid things

    """
    x = y
    p = y
    q = torch.zeros_like(y)

    encoder = U_Net()
    decoder = U_Net()

    optim = torch.optim.Adam()
    for i in range(NB_EPOCHS):
        for j in range(LITTLE_M):
            optim.zero_grad()
            epsilon = torch.randn_like(x)
            loss = compute_langragian(x, encoder, decoder, p, q)
            loss.backward()
            optim.step()

            with torch.no_grad():
                x = update_x(encoder, y, p, q)

        # outside nested loop
        p = gaussianDenoiser(x + q / RHO)
        q = q + MU * RHO * (x - p)

    return x
