import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import scipy.ndimage
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
import cupy as cp
import cupyx.scipy.ndimage
from other.parsing.train_args_parser import *
import random


class ImageProcessor:
    @staticmethod
    def _validate_tensor(tensor: torch.Tensor):
        """
        Validate that the input is a 4D torch.Tensor (B, C, H, W).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if tensor.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")

    @staticmethod
    def find_boundary_pixels(batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Find the boundary pixels in a batch of binary images.

        :param batch_tensor: 4D tensor of shape (B, C, H, W)
        :return: A binary tensor of the same shape with boundaries marked as 1.
        """
        ImageProcessor._validate_tensor(batch_tensor)
        device = batch_tensor.device
        batch_size, channel_size, height, width = batch_tensor.shape
        # Define a 3x3 kernel for boundary detection (shape: (b, c, h, w))
        kernel = torch.tensor([[[[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]]]], dtype=torch.float32, device=device)
        # Pad the tensor to preserve dimensions during convolution
        padded = F.pad(batch_tensor.to(dtype=torch.float32), (1, 1, 1, 1), mode='replicate')
        # Eroded image: pixels that are completely surrounded by 1's
        # RuntimeError: Input type (torch.cuda.IntTensor) and weight type (torch.cuda.ByteTensor) should be the same
        eroded = F.conv2d(padded, kernel, stride=1, groups=channel_size) == kernel.sum()
        # Closed image: pixels where the kernel overlaps at least one non-zero pixel
        closed = F.conv2d(padded, kernel, stride=1, groups=channel_size) > 0
        # Boundary pixels are those that are in the closed image but not in the eroded one
        boundaries = (closed & ~eroded).to(torch.uint8)
        return boundaries

    @staticmethod
    def find_boundary_pixels_using_blur(batch_tensor: torch.Tensor, alpha_blur_type, alpha_blur_kernel_range):
        ImageProcessor._validate_tensor(batch_tensor)
        if alpha_blur_type == 'GaussianBlur':
            k_min, k_max = alpha_blur_kernel_range
            ksize = random.randrange(k_min, k_max + 1, 2)
            blured_alpha = TF.gaussian_blur(batch_tensor.float(), kernel_size=[ksize,ksize]) * batch_tensor.float()

        else:
            raise ValueError(f"Unsupported blur type: {alpha_blur_type}")

        boundaries = ((blured_alpha > 0.1) & (blured_alpha < 0.9)).to(torch.uint8)
        return boundaries

    @staticmethod
    def binarize_array(tensor: torch.Tensor):
        ImageProcessor._validate_tensor(tensor)
        binary = (tensor > 0).to(torch.uint8)
        return binary

    # @staticmethod
    # def binarize_array(tensor: torch.Tensor):
    #     """
    #     Binarizes an input tensor using Otsu's method.
    #
    #     :param tensor: 4D tensor (B, C, H, W)
    #     :return: A tuple (binary tensor, thresholds) where binary is the binarized tensor and thresholds is the computed threshold.
    #     """
    #     ImageProcessor._validate_tensor(tensor)
    #     batch_size, channel_size, height, width = tensor.shape
    #     device = tensor.device
    #
    #     # Scale tensor values to [0, 255] and round to nearest integer
    #     # torch.Size([1, 1, 640, 360])
    #     scaled = (tensor * 255).round().to(torch.float32)
    #     # Compute histogram with 256 bins for each image in the batch
    #     # torch.Size([1, 1, 256]) for one batch hist works ???? for many?
    #     # hist = torch.histc(scaled, bins=256, min=0, max=255, out=None).reshape(batch_size, channel_size, 256)
    #     histograms = []
    #     for b in range(batch_size):
    #         hist = torch.histc(scaled[b, 0].flatten(), bins=256, min=0, max=255).to(device)
    #         histograms.append(hist.unsqueeze(0))
    #
    #     hist = torch.stack(histograms)
    #
    #     bins = torch.arange(256, device=device, dtype=torch.float32).view(1, 1, -1)
    #     wB = torch.cumsum(hist, dim=-1)  # Cumulative sum of background weights
    #     wF = wB[:, :, -1:] - wB  # Cumulative sum of foreground weights
    #     valid = (wB > 0) & (wF > 0)
    #     mB = torch.cumsum(hist * bins, dim=-1)  # Cumulative background means
    #     mF = (mB[:, :, -1:] - mB) / wF.clamp(min=1e-6)  # Cumulative foreground means
    #     mB = mB / wB.clamp(min=1e-6)
    #     # Compute between-class variance
    #     between = valid * wB * wF * (mB - mF) ** 2
    #     _, optimal_index = torch.max(between, dim=-1)
    #     thresholds = optimal_index.float() / 255.0
    #     thresholds = thresholds.view(batch_size, channel_size, 1, 1)
    #     binary = (tensor > thresholds).to(torch.uint8)
    #     return binary, thresholds


class DistanceCalculator:
    """
    Calculates distance maps from boundary images.

    The output distance map shape: (B, C, H, W)
    Target images shape: (B, H, W)
    """

    def __init__(self, target_array: torch.Tensor, binarizer=ImageProcessor):
        if not isinstance(target_array, torch.Tensor):
            raise TypeError("target_array must be a torch.Tensor")
        if target_array.dim() != 3:
            raise ValueError("target_array must have 3 dimensions (B, H, W)")
        # target_array -> torch.Size([B, H, W])
        self.target_array = target_array
        # Binarize target_array by adding a channel dimension; shape -> (B, C=1, H, W)
        self.target_array_binary = binarizer.binarize_array(self.target_array.unsqueeze(1).detach())
        # Find boundary pixels from the binarized array; shape -> (B, C, H, W)
        self.target_binary_boundary_array = binarizer.find_boundary_pixels(self.target_array_binary)

        self.device = None
        if self.target_array.is_cuda:
            self.device = self.target_binary_boundary_array.device

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor):
        """
        Validate that the input is a 4D tensor (B, C, H, W).
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if tensor.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")

    def compute_distance_matrix_on_gpu(self):
        """
        Compute the distance map on the GPU.

        The function converts the target binary boundary array to a cupy array,
        inverts it, computes the distance transform for each image in the batch,
        normalizes the result to the range [0, 1], and returns a torch.Tensor on GPU.
        """
        if self.device:
            device = self.device
        else:
            raise ValueError("array must be on GPU")
        DistanceCalculator._validate_tensor(self.target_binary_boundary_array)
        batch_size, channel_size, _, _ = self.target_binary_boundary_array.shape

        # Convert the tensor to a cupy array (from torch.Tensor)
        tensor_cp = cp.asarray(self.target_binary_boundary_array)
        # Invert the image: same as CPU version (inverted_image = 1 - target)
        inverted_image = 1 - tensor_cp
        # Initialize distance_maps with zeros; shape: (b, c, h, w)
        distance_maps = cp.zeros_like(inverted_image, dtype=cp.float32)
        for b in range(batch_size):
            for c in range(channel_size):
                # Check if there are any boundary pixels in the current image
                if cp.any(tensor_cp[b, c]):
                    # Compute the Euclidean distance transform for each channel
                    distance_map = cupyx.scipy.ndimage.distance_transform_edt(inverted_image[b, c])
                    # Normalize the distance map by dividing by its maximum value (with protection against division by zero)
                    max_val = cp.amax(distance_map, keepdims=True)  # (b, c, h, w)
                    distance_maps[b, c] = distance_map / cp.maximum(max_val, 1e-6)
                else:
                    distance_maps[b, c] = cp.zeros_like(tensor_cp[b, c])
        # torch.from_dlpack(distance_maps.toDlpack()).to(device)
        return torch.as_tensor(distance_maps).to(device)

    def compute_distance_matrix_on_cpu(self):
        """
        Compute the distance map on the CPU.

        The function converts the target binary boundary array to a numpy array,
        inverts it, computes the distance transform for each image in the batch,
        normalizes the result to the range [0, 1], and returns a torch.Tensor.
        """
        if self.device is not None:
            raise ValueError("array must be on CPU")
        DistanceCalculator._validate_tensor(self.target_binary_boundary_array)

        batch_size, channel_size, height, width = self.target_binary_boundary_array.shape  # (b, c, h, w)
        # Convert the tensor to a numpy array of type uint8
        target_binary_boundary_array = self.target_binary_boundary_array.numpy().astype(np.uint8)
        # Invert the image: same as GPU version (inverted_image = 1 - target)
        inverted_image = 1 - target_binary_boundary_array

        # Initialize distance_maps with zeros; shape: (b, c, h, w)
        distance_maps = np.zeros((batch_size, channel_size, height, width), dtype=np.float32)
        for b in range(batch_size):
            for c in range(channel_size):
                if np.any(inverted_image[b, c]):
                    # Compute the Euclidean distance transform for the current image
                    distance_map = distance_transform_edt(inverted_image[b, c])
                    # Normalize the distance map by dividing by its maximum value (with protection against division by zero)
                    max_val = np.amax(distance_map, keepdims=True)  # (b, c, h, w)
                    distance_map /= np.maximum(max_val, 1e-6)
                    distance_maps[b, c] = distance_map
                else:
                    # If there are no boundary pixels, set the distance map to zeros
                    distance_maps[b, c] = np.zeros_like(distance_maps[b, c])
        # Return the computed distance maps as a torch.Tensor (float32)
        return torch.tensor(distance_maps, dtype=torch.float32)
