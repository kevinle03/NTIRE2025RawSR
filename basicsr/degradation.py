import os
import torch
import torch.nn.functional as F
from degra_utils import square_mask, random_mask, paintbrush_mask, gaussian_blur, gaussian_2d_kernel, downsample, upsample, bicubic_filter, create_downsampling_matrix
import numpy as np
import hdf5storage

class Degradation:

    def H(self, x):
        raise NotImplementedError()

    def H_adj(self, x):
        raise NotImplementedError()


class Denoising(Degradation):
    def H(self, x):
        return x

    def H_adj(self, x):
        return x


class BoxInpainting(Degradation):
    def __init__(self, half_size_mask):
        super().__init__()
        self.half_size_mask = half_size_mask

    def H(self, x):
        return square_mask(x, self.half_size_mask)

    def H_adj(self, x):
        return square_mask(x, self.half_size_mask)


class RandomInpainting(Degradation):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def H(self, x):
        return random_mask(x, self.p)

    def H_adj(self, x):
        return random_mask(x, self.p)


class PaintbrushInpainting(Degradation):
    def H(self, x):
        return paintbrush_mask(x)

    def H_adj(self, x):
        return paintbrush_mask(x)


class GaussianDeblurring(Degradation):
    def __init__(self, sigma_blur, kernel_size,  mode="fft", num_channels=3, dim_image=128, device="cuda") -> None:
        super().__init__()
        self.mode = mode
        self.sigma = sigma_blur
        self.kernel_size = kernel_size
        self.kernel = gaussian_2d_kernel(sigma_blur, kernel_size).to(device)
        filter = torch.zeros(
            (1, num_channels) + (dim_image, dim_image), device=device
        )

        filter[..., : kernel_size, : kernel_size] = self.kernel
        self.filter = torch.roll(
            filter, shifts=(-(kernel_size-1)//2, -(kernel_size-1)//2), dims=(2, 3))
        self.device = device

    def H(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(
                1, 1, self.kernel_size,  self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        else:
            return torch.real(torch.fft.ifft2(
                torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.filter)))

    def H_adj(self, x):
        if self.mode != "fft":
            kernel = self.kernel.view(
                1, 1, self.kernel_size,  self.kernel_size)
            kernel = self.kernel.repeat(x.shape[1], 1, 1, 1)
            return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])
        else:
            return torch.real(torch.fft.ifft2(
                torch.fft.fft2(x.to(self.device)) * torch.conj(torch.fft.fft2(self.filter))))


class Superresolution(Degradation):
    def __init__(self, sf, dim_image, mode=None, device="cuda") -> None:
        super().__init__()
        self.sf = sf
        self.mode = mode
        if mode == "bicubic":
            self.filter = torch.nn.Parameter(
                bicubic_filter(sf), requires_grad=False
            ).to(device)
            # Move batch dim of the input into channels

            filter = torch.zeros(
                (1, 3) + (dim_image, dim_image), device=device)

            filter[..., : self.filter.shape[-1],
                   : self.filter.shape[-1]] = self.filter
            self.filter = torch.roll(
                filter, shifts=(-(self.filter.shape[-1]-1)//2, -(self.filter.shape[-1]-1)//2), dims=(2, 3)).to(device)
        self.downsampling_matrix = create_downsampling_matrix(
            dim_image, dim_image, sf, device)

    def H(self, x):

        if self.mode == None:
            return downsample(x, self.sf)
        elif self.mode == "bicubic":
            x_ = torch.real(torch.fft.ifft2(
                torch.fft.fft2(x) * torch.fft.fft2(self.filter)))
            return downsample(x_, self.sf)

    def H_adj(self, x):
        if self.mode == None:
            return upsample(x, self.sf)
        elif self.mode == "bicubic":
            x_ = upsample(x, self.sf)
            return torch.real(torch.fft.ifft2(torch.fft.fft2(x_) * torch.conj(torch.fft.fft2(self.filter))))

# For making training more robust by adding a variety of blur kernels and Gaussian noise        
class Superresolution_blur(Degradation):
    def __init__(self, sf, dim_image, mode=None, device="cuda", blur_mode=0, sigma_blur=1, kernel_size=25, num_channels=3, sigma_noise=2.55, kernel_path=None) -> None:
        super().__init__()
        self.sf = sf
        self.mode = mode
        self.device = device
        self.blur_mode = blur_mode  # 0 for Guassian blur, 1 to 8 for the eight different motion blur kernels from PnP_GS
        
        if kernel_path is None:
            kernel_path = os.path.join(os.path.dirname(__file__), 'kernels', 'kernels_12.mat')

        if mode == "bicubic":
            self.filter = torch.nn.Parameter(
                bicubic_filter(sf), requires_grad=False
            ).to(device)
            # Move batch dim of the input into channels

            filter = torch.zeros(
                (1, 3) + (dim_image, dim_image), device=device)

            filter[..., : self.filter.shape[-1],
                   : self.filter.shape[-1]] = self.filter
            self.filter = torch.roll(
                filter, shifts=(-(self.filter.shape[-1]-1)//2, -(self.filter.shape[-1]-1)//2), dims=(2, 3)).to(device)
        self.downsampling_matrix = create_downsampling_matrix(
            dim_image, dim_image, sf, device)

        # For Gaussian blurring
        if blur_mode == 0:
            self.kernel = gaussian_2d_kernel(sigma_blur, kernel_size).to(device)
            blur_filter = torch.zeros(
                (1, num_channels) + (dim_image, dim_image), device=device
            )
            blur_filter[..., : kernel_size, : kernel_size] = self.kernel
            self.blur_filter = torch.roll(
                blur_filter, shifts=(-(kernel_size-1)//2, -(kernel_size-1)//2), dims=(2, 3))

        # For motion blur kernels
        if blur_mode >= 1 and blur_mode <= 8:
            k_list = []
            kernels = hdf5storage.loadmat(kernel_path)['kernels']
            # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
            for k_index in range(8) :
                k = kernels[0, k_index]
                k_tensor = torch.from_numpy(k).float()
                k_list.append(k_tensor)
        
            kernel_size = 25
            self.kernel = k_list[blur_mode-1].to(device)
            blur_filter = torch.zeros(
                (1, num_channels) + (dim_image, dim_image), device=device
            )
            blur_filter[..., : kernel_size, : kernel_size] = self.kernel
            self.blur_filter = torch.roll(
                blur_filter, shifts=(-(kernel_size-1)//2, -(kernel_size-1)//2), dims=(2, 3))
            self.sigma_noise = sigma_noise

    def H(self, x):
        # Add Gaussian blur/motion blur
        if self.blur_mode >= 0 and self.blur_mode <= 8:
            x = torch.real(torch.fft.ifft2(
                    torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.blur_filter)))

        # Add Gaussian noise (currently only added on top of motion blur kernel)
        if self.blur_mode >= 1 and self.blur_mode <= 8:
            noise = torch.randn_like(x) * (self.sigma_noise / 255.)
            x += noise

        if self.mode == None:
            return downsample(x, self.sf)
        elif self.mode == "bicubic":
            x_ = torch.real(torch.fft.ifft2(
                torch.fft.fft2(x.to(self.device)) * torch.fft.fft2(self.filter)))
            return downsample(x_, self.sf)