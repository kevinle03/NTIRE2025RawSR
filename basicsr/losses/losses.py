import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision
# from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


# @LOSS_REGISTRY.register()
# class PerceptualLoss(nn.Module):
#     """Perceptual loss with commonly used style loss.
#
#     Args:
#         layer_weights (dict): The weight for each layer of vgg feature.
#             Here is an example: {'conv5_4': 1.}, which means the conv5_4
#             feature layer (before relu5_4) will be extracted with weight
#             1.0 in calculating losses.
#         vgg_type (str): The type of vgg network used as feature extractor.
#             Default: 'vgg19'.
#         use_input_norm (bool):  If True, normalize the input image in vgg.
#             Default: True.
#         range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
#             Default: False.
#         perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
#             loss will be calculated and the loss will multiplied by the
#             weight. Default: 1.0.
#         style_weight (float): If `style_weight > 0`, the style loss will be
#             calculated and the loss will multiplied by the weight.
#             Default: 0.
#         criterion (str): Criterion used for perceptual loss. Default: 'l1'.
#     """
#
#     def __init__(self,
#                  layer_weights,
#                  vgg_type='vgg19',
#                  use_input_norm=True,
#                  range_norm=False,
#                  perceptual_weight=1.0,
#                  style_weight=0.,
#                  criterion='l1'):
#         super(PerceptualLoss, self).__init__()
#         self.perceptual_weight = perceptual_weight
#         self.style_weight = style_weight
#         self.layer_weights = layer_weights
#         self.vgg = VGGFeatureExtractor(
#             layer_name_list=list(layer_weights.keys()),
#             vgg_type=vgg_type,
#             use_input_norm=use_input_norm,
#             range_norm=range_norm)
#
#         self.criterion_type = criterion
#         if self.criterion_type == 'l1':
#             self.criterion = torch.nn.L1Loss()
#         elif self.criterion_type == 'l2':
#             self.criterion = torch.nn.L2loss()
#         elif self.criterion_type == 'fro':
#             self.criterion = None
#         else:
#             raise NotImplementedError(f'{criterion} criterion has not been supported.')
#
#     def forward(self, x, gt):
#         """Forward function.
#
#         Args:
#             x (Tensor): Input tensor with shape (n, c, h, w).
#             gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
#
#         Returns:
#             Tensor: Forward results.
#         """
#         # extract vgg features
#         x_features = self.vgg(x)
#         gt_features = self.vgg(gt.detach())
#
#         # calculate perceptual loss
#         if self.perceptual_weight > 0:
#             percep_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
#                 else:
#                     percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
#             percep_loss *= self.perceptual_weight
#         else:
#             percep_loss = None
#
#         # calculate style loss
#         if self.style_weight > 0:
#             style_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     style_loss += torch.norm(
#                         self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
#                 else:
#                     style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
#                         gt_features[k])) * self.layer_weights[k]
#             style_loss *= self.style_weight
#         else:
#             style_loss = None
#
#         return percep_loss, style_loss
#
#     def _gram_mat(self, x):
#         """Calculate Gram matrix.
#
#         Args:
#             x (torch.Tensor): Tensor with shape of (n, c, h, w).
#
#         Returns:
#             torch.Tensor: Gram matrix.
#         """
#         n, c, h, w = x.size()
#         features = x.view(n, c, w * h)
#         features_t = features.transpose(1, 2)
#         gram = features.bmm(features_t) / (c * h * w)
#         return gram

@LOSS_REGISTRY.register()
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[], mask=None, return_feature=False):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                if mask is not None:
                    _,_,H,W = x.shape
                    mask_resized = F.interpolate(mask, size=(H, W), mode='nearest')[:, 0:1, :, :]
                    x = x*mask_resized
                    y = y*mask_resized
                    loss += torch.nn.functional.l1_loss(x, y)
                else:
                    loss += torch.nn.functional.l1_loss(x, y)
                    
                if return_feature:
                    return x, y
                    
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight

import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY
import utils.SWT as SWT
import pywt
import numpy as np

@LOSS_REGISTRY.register()
class SWTLoss(nn.Module):
    def __init__(self, WAVELET, WEIGHT_PEC, WEIGHT_L1, loss_weight_ll=0.05, loss_weight_lh=0.01, loss_weight_hl=0.01, loss_weight_hh=0.00, reduction='mean'):
        super(SWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh
        self.WAVELET = WAVELET
        self.WEIGHT_L1 = WEIGHT_L1
        self.criterion = nn.L1Loss(reduction=reduction)
        # self.multi_vgg_loss = multi_VGGPerceptualLoss(lam_p=WEIGHT_PEC, lam_l=WEIGHT_L1).to("cuda")
        # Initialize the wavelet transform filters
        wavelet = pywt.Wavelet('sym19')
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2 * np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi
        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        self.sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")  # Assume using CUDA

    def forward(self, out, gt, feature_layers=[2]):
        result = {}
        # vgg_perceptual_loss = self.multi_vgg_loss(out, gt, feature_layers=feature_layers)

        LL_1, LH_1, HL_1, HH_1 = self.compute_wavelet_loss(out, gt)
        swt_loss = LL_1 + LH_1 + HL_1 + HH_1
        l1_loss=self.criterion(out,gt)
        # result['total'] = self.WAVELET * (swt_loss) + vgg_perceptual_loss['total']
        # result['VGG'] = vgg_perceptual_loss['total']
        # return result
        
        # return self.WAVELET * (swt_loss) + vgg_perceptual_loss['total']
        return self.WAVELET*swt_loss + self.WEIGHT_L1*l1_loss

    def compute_wavelet_loss(self, pred, target):
        # Convert images to Y channel
        sr_img_y = 16.0 + (pred[:, 0:1, :, :] * 65.481 + pred[:, 1:2, :, :] * 128.553 + pred[:, 2:, :, :] * 24.966)
        hr_img_y = 16.0 + (target[:, 0:1, :, :] * 65.481 + target[:, 1:2, :, :] * 128.553 + target[:, 2:, :, :] * 24.966)
        
        # Compute wavelet transforms
        wavelet_sr = self.sfm(sr_img_y)[0]
        wavelet_hr = self.sfm(hr_img_y)[0]
        
        # Extract subbands
        LL_sr, LH_sr, HL_sr, HH_sr = wavelet_sr[:, 0:1, :, :], wavelet_sr[:, 1:2, :, :], wavelet_sr[:, 2:3, :, :], wavelet_sr[:, 3:, :, :]
        LL_hr, LH_hr, HL_hr, HH_hr = wavelet_hr[:, 0:1, :, :], wavelet_hr[:, 1:2, :, :], wavelet_hr[:, 2:3, :, :], wavelet_hr[:, 3:, :, :]
        
        # Compute losses for each subband
        loss_subband_LL = self.loss_weight_ll * self.criterion(LL_sr, LL_hr)
        loss_subband_LH = self.loss_weight_lh * self.criterion(LH_sr, LH_hr)
        loss_subband_HL = self.loss_weight_hl * self.criterion(HL_sr, HL_hr)
        loss_subband_HH = self.loss_weight_hh * self.criterion(HH_sr, HH_hr)
        
        return loss_subband_LL, loss_subband_LH, loss_subband_HL, loss_subband_HH
     
def color_loss(x1, x2):
    x1_norm = torch.sqrt(torch.sum(torch.pow(x1, 2)) + 1e-8)
    x2_norm = torch.sqrt(torch.sum(torch.pow(x2, 2)) + 1e-8)
    # 内积
    x1_x2 = torch.sum(torch.mul(x1, x2))
    cosin = 1 - x1_x2 / (x1_norm * x2_norm)
    return cosin
        
class ColorLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.s
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred,tuple):
            loss_1 = color_loss(pred[0], target)
            target_2 = F.interpolate(target, scale_factor=0.5, mode='bilinear')
            loss_2 = color_loss(pred[1], target_2)
            target_3 = F.interpolate(target_2, scale_factor=0.5, mode='bilinear')
            loss_3 = color_loss(pred[2], target_3)
            return self.loss_weight * (loss_1+loss_2+loss_3)
        else:
            return self.loss_weight * color_loss(pred, target)
           
class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, p_resize=True, lam_p=1.0, lam_l=0.5, lam_c=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss(p_resize)
        self.lam_p = lam_p
        self.lam_l = lam_l
        self.color_loss = ColorLoss(lam_c)

    def forward(self, out1, gt1, feature_layers=[2]):
        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam_l*F.l1_loss(out1, gt1) + self.color_loss(out1, gt1)
        result = {}
        result['total'] = loss1
        return result