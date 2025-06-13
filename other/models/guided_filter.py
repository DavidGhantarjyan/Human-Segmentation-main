import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from other.models.box_filter import BoxFilter

class FastGuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        """
        Fast Guided Filter for edge-preserving smoothing, operating on low-resolution inputs
        and applying results to high-resolution guidance.

        Computes filter coefficients at low resolution and upsamples them for efficiency.
        """
        super(FastGuidedFilter, self).__init__()
        """
        Args:
            r (int): Radius of the box filter (window size is 2r+1).
            eps (float): Regularization term to avoid division by zero, default 1e-8.
        """
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, lr_x, lr_y, hr_x):
        """
               Apply fast guided filter.

               Args:
                   lr_x (torch.Tensor): Low-resolution guidance image, shape (N, C, H_lr, W_lr).
                   lr_y (torch.Tensor): Low-resolution input image, shape (N, C, H_lr, W_lr).
                   hr_x (torch.Tensor): High-resolution guidance image, shape (N, C, H_hr, W_hr).

               Returns:
                   torch.Tensor: Filtered high-resolution output, shape (N, C, H_hr, W_hr).

               Notes:
                   - Assumes lr_x and lr_y have the same size, and hr_x has matching batch and channels.
                   - lr_x channels must be 1 or match lr_y channels.
                   - Input tensors should have padding of size r to avoid boundary issues.
                   - BoxFilter outputs sums, which are normalized by window area for mean computation.
               """
        # Get input dimensions
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        # Validate input dimensions
        assert n_lrx == n_lry and n_lry == n_hrx , "Batch sizes must match"
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry),  "Invalid channel dimensions"
        assert h_lrx == h_lry and w_lrx == w_lry,  "Low-res dimensions must match"
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1, "Low-res size too small for radius"

        # Compute window area (N = (2r+1)^2)
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # Compute means and covariances using box filter
        mean_x = self.boxfilter(lr_x) / N
        mean_y = self.boxfilter(lr_y) / N
        cov_xy = self.boxfilter(lr_x * lr_y) / N - mean_x * mean_y
        var_x = self.boxfilter(lr_x * lr_x) / N - mean_x * mean_x

        # Compute filter coefficients
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        # Upsample coefficients to high resolution
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        # Apply filter to high-res guidance
        return mean_A*hr_x+mean_b


class GuidedFilter(nn.Module):
    """
    Standard Guided Filter for edge-preserving smoothing on high-resolution inputs.

    Computes and smooths filter coefficients at high resolution for accuracy.
    """
    def __init__(self, r, eps=1e-8):
        """
        Args:
            r (int): Radius of the box filter (window size is 2r+1).
            eps (float): Regularization term to avoid division by zero, default 1e-8.
        """
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        """
              Apply guided filter.

              Args:
                  x (torch.Tensor): High-resolution guidance image, shape (N, C, H, W).
                  y (torch.Tensor): High-resolution input image, shape (N, C, H, W).

              Returns:
                  torch.Tensor: Filtered output, shape (N, C, H, W).

              Notes:
                  - Assumes x and y have the same size.
                  - x channels must be 1 or match y channels.
                  - Input tensors should have padding of size r to avoid boundary issues.
                  - BoxFilter outputs sums, which are normalized by window area for mean computation.
              """
        # Get input dimensions
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        # Validate input dimensions
        assert n_x == n_y, "Batch sizes must match"
        assert c_x == 1 or c_x == c_y, "Invalid channel dimensions"
        assert h_x == h_y and w_x == w_y, "Dimensions must match"
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1, "Image size too small for radius"

        # Compute window area (N = (2r+1)^2)
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # Compute means and covariances using box filter
        mean_x = self.boxfilter(x) / N
        mean_y = self.boxfilter(y) / N
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # Compute filter coefficients
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        # Smooth coefficients with box filter
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        # Apply filter to guidance image
        return mean_A * x + mean_b

class ConvGuidedFilter(nn.Module):
    """
    Convolutional Guided Filter with learnable coefficients for edge-preserving smoothing.

    Uses a ConvNet to compute filter coefficients, applied to low-res inputs and upsampled
    for high-res output.
    """
    def __init__(self, radius=1, norm=nn.BatchNorm2d):
        """
        Args:
            radius (int): Radius for the box filter convolution (kernel size is 2r+1), default 1.
            norm (nn.Module): Normalization layer, default nn.BatchNorm2d.
        """
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3)
        self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 3, kernel_size=1, bias=False))
        # Initialize box filter weights to 1.0 for mean computation
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        """
        Apply convolutional guided filter.

        Args:
            x_lr (torch.Tensor): Low-resolution guidance image, shape (N, 3, H_lr, W_lr).
            y_lr (torch.Tensor): Low-resolution input image, shape (N, 3, H_lr, W_lr).
            x_hr (torch.Tensor): High-resolution guidance image, shape (N, 3, H_hr, W_hr).

        Returns:
            torch.Tensor: Filtered high-resolution output, shape (N, 3, H_hr, W_hr).

        Notes:
            - Assumes 3-channel inputs (C=3).
            - Input tensors should have padding to match convolution size.
            - Box filter convolution outputs sums, normalized by window area.
        """
        # Get input dimensions
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        # Compute window area (N = kernel_size^2)
        N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))

        # Compute means and covariances using convolutional box filter
        mean_x = self.box_filter(x_lr)/N
        mean_y = self.box_filter(y_lr)/N
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        # Compute filter coefficients using ConvNet
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - A * mean_x

        # Upsample coefficients to high resolution
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        # Apply filter to high-res guidance
        return mean_A * x_hr + mean_b