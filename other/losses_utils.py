import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt
import cupy as cp
import cupyx.scipy.ndimage
from other.parsing.train_args_parser import *
import random
import kornia.contrib as K

###############################################################################
# ImageProcessor Class
###############################################################################
class ImageProcessor:
    @staticmethod
    def _validate_tensor(tensor: torch.Tensor):
        """
        Validate that the input is a 4D torch.Tensor with dimensions (B, C, H, W).

        :param tensor: A torch.Tensor to validate.
        :raises TypeError: If the input is not a torch.Tensor.
        :raises ValueError: If the tensor does not have 4 dimensions.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if tensor.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")

    @staticmethod
    def find_boundary_pixels(batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Find boundary pixels in a batch of binary images using a simple morphological approach.

        This function performs a convolution with a 3x3 kernel to determine the boundary.
        It compares the result of a convolution to a fully 'on' kernel to detect eroded areas.
        Boundary pixels are those that are present in the closed image but not in the eroded image.

        :param batch_tensor: 4D tensor of shape (B, C, H, W) containing binary images.
        :return: A binary tensor of the same shape with boundary pixels marked as 1.
        """
        # Validate the tensor dimensions
        ImageProcessor._validate_tensor(batch_tensor)
        device = batch_tensor.device
        batch_size, channel_size, height, width = batch_tensor.shape
        # Define a 3x3 kernel that emphasizes the center and its immediate neighbors.
        kernel = torch.tensor([[[[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]]]], dtype=torch.float32, device=device)
        # Pad the input to preserve the spatial dimensions after convolution.
        padded = F.pad(batch_tensor.to(dtype=torch.float32), (1, 1, 1, 1), mode='replicate')
        # Eroded image: locations where the kernel fully overlaps with ones.
        eroded = F.conv2d(padded, kernel, stride=1, groups=channel_size) == kernel.sum()
        # Closed image: locations where at least one pixel is non-zero.
        closed = F.conv2d(padded, kernel, stride=1, groups=channel_size) > 0
        # Boundary pixels: pixels in the closed image that are not part of the eroded image.
        boundaries = (closed & ~eroded).to(torch.uint8)
        return boundaries

    @staticmethod
    def find_boundary_pixels_using_blur(batch_tensor: torch.Tensor, alpha_blur_type, alpha_blur_kernel_range):
        """
        Find boundary pixels by applying a blur to the input batch and then thresholding.

        For example, using a Gaussian blur, the function multiplies the blurred tensor with the original,
        and then identifies boundaries by checking where the values lie between 0.1 and 0.9.

        :param batch_tensor: 4D tensor of shape (B, C, H, W) representing the input images.
        :param alpha_blur_type: The type of blur to use (e.g., 'GaussianBlur').
        :param alpha_blur_kernel_range: Tuple or list defining the range of kernel sizes.
        :return: A binary tensor with boundaries marked as 1.
        :raises ValueError: If the provided blur type is unsupported.
        """
        ImageProcessor._validate_tensor(batch_tensor)
        if alpha_blur_type == 'GaussianBlur':
            # Choose a random odd kernel size between k_min and k_max.
            k_min, k_max = alpha_blur_kernel_range
            ksize = random.randrange(k_min, k_max + 1, 2)
            float_batch_tensor = batch_tensor.float()
            # Apply Gaussian blur using torchvision's functional API.
            blured_alpha = TF.gaussian_blur(float_batch_tensor, kernel_size=[ksize,ksize]) * float_batch_tensor
        else:
            raise ValueError(f"Unsupported blur type: {alpha_blur_type}")
        # Identify boundary regions by thresholding the blurred output.
        boundaries = ((blured_alpha > 0.1) & (blured_alpha < 0.9)).to(torch.uint8)
        return boundaries

    @staticmethod
    def binarize_array(tensor: torch.Tensor, threshold=0):
        """
        Convert the tensor to a binary format based on a threshold.

        :param tensor: 4D tensor of shape (B, C, H, W).
        :param threshold: Threshold value for binarization.
        :return: A binary tensor of the same shape.
        """
        ImageProcessor._validate_tensor(tensor)
        binary = (tensor > threshold).to(torch.uint8)
        return binary

###############################################################################
# DistanceCalculator Class
###############################################################################
class DistanceCalculator:
    """
    Calculates distance maps from boundary images.

    The computed distance map has the same shape as the input (B, C, H, W).
    Target images are expected to be provided as (B, H, W) tensors.
    """
    def __init__(self, target_array: torch.Tensor, binarizer=ImageProcessor):
        """
        Initialize the DistanceCalculator.

        :param target_array: A 3D torch.Tensor of shape (B, H, W) representing the target masks.
        :param binarizer: A module (or class) that provides a binarize_array function.
        :raises TypeError: If target_array is not a torch.Tensor.
        :raises ValueError: If target_array does not have 3 dimensions.
        """
        if not isinstance(target_array, torch.Tensor):
            raise TypeError("target_array must be a torch.Tensor")
        if target_array.dim() != 3:
            raise ValueError("target_array must have 3 dimensions (B, H, W)")
        # Binarize the target array by adding a channel dimension, resulting in shape (B, 1, H, W)
        self.target_array = target_array
        # Binarize target_array by adding a channel dimension; shape -> (B, C=1, H, W)
        self.target_array_binary = binarizer.binarize_array(self.target_array.unsqueeze(1))
        # Find boundary pixels from the binarized array, shape remains (B, 1, H, W)
        self.target_binary_boundary_array = binarizer.find_boundary_pixels(self.target_array_binary)
        self.device = None
        if self.target_array.is_cuda:
            self.device = self.target_binary_boundary_array.device

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor):
        """
        Validate that the input is a 4D tensor (B, C, H, W).

        :param tensor: A torch.Tensor to validate.
        :raises TypeError: If the input is not a torch.Tensor.
        :raises ValueError: If the tensor does not have 4 dimensions.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if tensor.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")

    def compute_distance_matrix_on_gpu(self,use_edt):
        """
        Compute the distance map on the GPU.

        This method converts the target binary boundary array to a CuPy array,
        inverts it, computes the distance transform for each image in the batch,
        normalizes the result to the range [0, 1], and returns the result as a torch.Tensor on GPU.

        :param use_edt: If True, use the Euclidean distance transform; if False, use a default transform.
        :return: A normalized distance map tensor on the GPU.
        :raises ValueError: If the target array is not on the GPU.
        """
        if self.device:
            device = self.device
        else:
            raise ValueError("array must be on GPU")
        DistanceCalculator._validate_tensor(self.target_binary_boundary_array)
        batch_size, channel_size, _, _ = self.target_binary_boundary_array.shape
        if not use_edt:
            # Use kornia's distance_transform function (GPU implementation)
            distance_maps = K.distance_transform(self.target_binary_boundary_array.float())
            max_vals = torch.amax(distance_maps, dim=(2, 3), keepdim=True)
            return distance_maps  / torch.clamp(max_vals, min=1e-6)
        else:
            # Convert the target tensor to a CuPy array
            tensor_cp = cp.asarray(self.target_binary_boundary_array.float())
            # Invert the image (1 - target)
            inverted_image = 1 - tensor_cp
            # Initialize a distance map array with the same shape
            distance_maps = cp.zeros_like(inverted_image)
            for b in range(batch_size):
                    for c in range(channel_size):
                        # Compute the Euclidean distance transform using Cupyx
                        distance_map = cupyx.scipy.ndimage.distance_transform_edt(inverted_image[b, c])
                        max_val = cp.amax(distance_map, keepdims=True)
                        # Normalize the distance map to [0, 1]
                        distance_maps[b, c] = distance_map / cp.maximum(max_val, 1e-6)
            return torch.as_tensor(distance_maps)



    def compute_distance_matrix_on_cpu(self):
        """
        Compute the distance map on the CPU.

        Converts the target binary boundary array to a NumPy array,
        inverts it, computes the Euclidean distance transform for each image,
        normalizes each distance map to [0, 1], and returns the result as a torch.Tensor.

        :return: A torch.Tensor containing the normalized distance maps (dtype: float32).
        :raises ValueError: If the target array is on GPU.
        """
        if self.device is not None:
            raise ValueError("array must be on CPU")
        DistanceCalculator._validate_tensor(self.target_binary_boundary_array)

        batch_size, channel_size, height, width = self.target_binary_boundary_array.shape  # (b, c, h, w)
        # Convert the tensor to a NumPy array of type uint8 for processing.
        target_binary_boundary_array = self.target_binary_boundary_array.numpy().astype(np.uint8)
        # Invert the binary array.
        inverted_image = 1 - target_binary_boundary_array

        # Initialize an array to store the distance maps.
        distance_maps = np.zeros((batch_size, channel_size, height, width), dtype=np.float32)
        for b in range(batch_size):
            # Create a mask to determine which channels require distance calculation.
            mask = np.any(inverted_image[b], axis=(1, 2))
            for c in np.where(mask)[0]:
                if np.any(inverted_image[b, c]):
                    # Compute the Euclidean distance transform.
                    distance_map = distance_transform_edt(inverted_image[b, c])
                    max_val = np.max(distance_map,keepdims=True)
                    # Normalize the distance map; prevent division by zero.
                    distance_maps[b, c] = distance_map / max(max_val, 1e-6)
        # Return the computed distance maps as a torch.Tensor (float32)
        return torch.tensor(distance_maps, dtype=torch.float32)
