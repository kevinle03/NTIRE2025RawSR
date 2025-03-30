import rawpy
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
from starter_kit.blur import apply_psf, add_blur
from starter_kit.noise import add_natural_noise, add_gnoise, add_heteroscedastic_gnoise
from starter_kit.imutils import downsample_raw, convert_to_tensor
from starter_kit import utils_blindsr as blindsr
# from starter_kit.motionblur import Kernel



# img_lq, img_hq = blindsr.degradation_bsrgan_plus(img, sf=4, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
#    k = Kernel()

#     k.applyTo(image, keep_image_dim=True).show()
def extract_into_tensor(a, t, x_shape):
    # b, *_ = t.shape
    b = 1
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, alphas_cumprod, noise=None):

    if noise is None:
       noise = torch.randn_like(x_start)
       noise = noise * 0.001
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_alphas_cumprod_t = extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def simple_deg_simulation(img, kernels):
    """
    Pipeline to add synthetic degradations to a (RAW/RGB) image.
    y = down(x * k) + n
    """

    img = convert_to_tensor(img.astype(np.float32))

    # Apply psf blur: x * k
    img = add_blur(img, kernels)

    # Apply downsampling down(x*k)
    img = downsample_raw(img)
#    values = [float(line.strip()) for line in open('./noise.txt')]
    # values = [float(line.strip()) for line in open('./noise.txt') if line.strip()]
    # values.reverse()
    # values = torch.tensor(values)
    # t = torch.randint(1, 100, (1,))

    # img = q_sample(img, torch.tensor(t, dtype=int), values)
    # Add noise down(x*k) + n
    p_noise = np.random.rand()
    if p_noise < 0.5:
        img = add_natural_noise(img)
    else:
        img = add_heteroscedastic_gnoise(img)
    # else:
    #     img = q_sample(img, torch.tensor(t, dtype=int), values)
    img[img > 1] = 1  # Set values greater than 1 to 1
    img[img < 0] = 0

    return img