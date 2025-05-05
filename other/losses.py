import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from other.parsing.train_args_parser import *
from other.losses_utils import *


###############################################################################
# BlurBoundaryLoss Class
###############################################################################
class BlurBoundaryLoss(nn.Module):
    """
    Computes a loss that focuses on boundary regions using blurred masks.

    This loss modifies the standard binary cross entropy by weighting the loss
    at the boundary pixels more heavily. The boundary pixels are detected by
    blurring the target mask and then extracting the boundary regions.
    """
    def __init__(self):
        # Initialize the parent class (nn.Module)
        super(BlurBoundaryLoss, self).__init__()

    def _compute_boundary_loss_weight_map(self,targets):
        """
               Computes a weight map for the boundaries in the target masks.

               This method uses a helper from ImageProcessor to find boundary pixels
               by applying a blur. The returned boundaries (weight map) will be used to
               weight the loss such that errors on boundary pixels are penalized more.

               :param targets: Ground truth segmentation masks (tensor of shape [B, H, W]).
               :return: A tensor representing boundary weights.
               """
        # The function below finds boundaries after applying a blur to the targets.
        # Note: targets are unsqueezed to add a channel dimension (i.e. [B, 1, H, W])
        boundaries = ImageProcessor.find_boundary_pixels_using_blur(
            targets.unsqueeze(1), alpha_blur_type, synthetic_alpha_blur_kernel_range
        )
        return boundaries

    def forward(self,outputs,targets):
        """
        Computes the boundary-weighted binary cross-entropy loss.

        :param outputs: Raw model outputs (logits) with shape [B, C, H, W].
        :param targets: Ground truth segmentation masks with shape [B, H, W].
        :return: Scalar loss value averaged over all pixels.
        """
        # Compute a boundary weight map based on the target mask.
        masks = self._compute_boundary_loss_weight_map(targets)
        # Calculate binary cross entropy loss for each pixel without reduction.
        bce_loss = F.binary_cross_entropy_with_logits(
            outputs, targets.unsqueeze(1), reduction='none'
        )
        # Weight the BCE loss by the boundary mask.
        loss = bce_loss * masks
        # Return the mean loss value over the batch.
        return loss.mean()

###############################################################################
# BoundaryLossCalculator Class
###############################################################################
class BoundaryLossCalculator(nn.Module):
    """
    Calculates a boundary loss by comparing model predictions with the target boundaries.

    This class computes a weight map based on distance matrices computed from the target
    masks. It then uses these weight maps to emphasize errors along object boundaries.

    Parameters:
        device: The device (e.g., CUDA) on which tensors are processed.
        natural_data_mask_pre_calculation_mode: Mode to pre-calculate masks for natural data.
        synthetic_mask_pre_calculation_mode: Mode to pre-calculate masks for synthetic data.
    """
    def __init__(self, device, natural_data_mask_pre_calculation_mode=natural_data_mask_pre_calculation_mode,
                 synthetic_mask_pre_calculation_mode=synthetic_mask_pre_calculation_mode):
        super(BoundaryLossCalculator, self).__init__()
        self.device = device
        self.natural_data_mask_pre_calculation_mode = natural_data_mask_pre_calculation_mode
        self.synthetic_mask_pre_calculation_mode = synthetic_mask_pre_calculation_mode

    def _compute_boundary_loss_weight_map(self, outputs, targets, masks):
        """
        Computes the final weight map to be used for boundary loss.

        The weight map is derived from distance matrices calculated from the target masks.
        It differentiates between 'loss' modes for natural and synthetic data. After computing
        the distance-based masks, it calculates errors by comparing binarized outputs and targets.

        :param outputs: Model logits of shape [B, 1, H, W].
        :param targets: Ground truth masks of shape [B, H, W].
        :param masks: Precomputed masks of shape [B, H, W] (if available).
        :return: A weighted mask (tensor of shape [B, H, W]) that emphasizes boundary errors.
        """
        # If both natural and synthetic modes are 'loss', compute distance maps using GPU.
        if self.natural_data_mask_pre_calculation_mode == 'loss' and self.synthetic_mask_pre_calculation_mode == 'loss':
            distance_maps = DistanceCalculator(targets).compute_distance_matrix_on_gpu(use_edt=False)
            masks = distance_maps.squeeze(1)
        # If one of the modes is 'loss', compute distance maps only for samples with an empty mask.
        elif self.natural_data_mask_pre_calculation_mode == 'loss' or self.synthetic_mask_pre_calculation_mode == 'loss':
            mask_any = torch.any(masks, dim=(1, 2))
            indices = torch.nonzero(~mask_any, as_tuple=True)[0]
            targets_to_compute = targets[indices]
            distance_maps = (DistanceCalculator(targets_to_compute).compute_distance_matrix_on_gpu(use_edt=True))
            masks[indices] = distance_maps.squeeze(1)

        # Binarize the outputs using a given threshold (global parameter 'threshold').
        # print(torch.min(torch.sigmoid(outputs)),torch.max(torch.sigmoid(outputs)))
        output_array_binary = ImageProcessor.binarize_array(torch.sigmoid(outputs).detach(),threshold=threshold)
        # Initialize an empty tensor to store error signals.
        res = torch.zeros_like(output_array_binary, dtype=torch.int, device=self.device)
        # Set error signal to 1 where prediction is positive but target is non-positive.
        res = torch.where((output_array_binary > 0) & (targets <= 0), 1, res)
        # Set error signal to -1 where target is positive but prediction is non-positive.
        res = torch.where((targets > 0) & (output_array_binary <= 0), -1, res)

        # Multiply the computed error signal with the masks to weight the boundary regions.
        return res.squeeze(1) * masks

    def forward(self, outputs, targets, masks):
        """
        Computes the boundary loss.

        The loss is calculated by multiplying the sigmoid activation of the outputs with the
        computed boundary weight map, summing over spatial dimensions, and returning a mean loss.

        :param outputs: Model outputs (logits) of shape [B, 1, H, W].
        :param targets: Ground truth masks of shape [B, H, W].
        :param masks: Precomputed masks (or weight maps) of shape [B, H, W].
        :return: The computed boundary loss (scalar tensor).
        """
        # Compute the boundary loss weight map.
        masks = self._compute_boundary_loss_weight_map(outputs, targets, masks)
        # Apply sigmoid activation to the model outputs.
        s_theta = torch.sigmoid(outputs)
        # Multiply the activated outputs with the boundary weight map, then sum over all pixels.
        loss = (masks.unsqueeze(1) * s_theta).sum(dim=(1,2, 3))
        return loss
