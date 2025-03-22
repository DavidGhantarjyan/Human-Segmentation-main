import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from other.parsing.train_args_parser import *
from other.losses_utils import *
import cv2
import matplotlib.pyplot as plt


class BlurBoundaryLoss(nn.Module):
    def __init__(self):
        super(BlurBoundaryLoss, self).__init__()

    def _compute_boundary_loss_weight_map(self,targets):

        targets_array_binary = ImageProcessor.binarize_array(targets.unsqueeze(1))

        boundaries = ImageProcessor.find_boundary_pixels_using_blur(
            targets_array_binary, alpha_blur_type, alpha_blur_kernel_range
        )

        return boundaries

    def forward(self,outputs,targets):
        masks = self._compute_boundary_loss_weight_map(targets)

        bce_loss = F.binary_cross_entropy_with_logits(
            outputs, targets.unsqueeze(1), reduction='none'
        )
        loss = bce_loss * masks
        # loss -> torch.Size([b, 1, h, w])
        return loss.mean()


class BoundaryLossCalculator(nn.Module):
    # targets: torch.Size([b,h,w]) masks: torch.Size([b,h,w]) outputs: (b,c,h,w)
    def __init__(self, device, natural_data_mask_pre_calculation_mode=natural_data_mask_pre_calculation_mode,
                 synthetic_mask_pre_calculation_mode=synthetic_mask_pre_calculation_mode):
        super(BoundaryLossCalculator, self).__init__()
        self.device = device
        self.natural_data_mask_pre_calculation_mode = natural_data_mask_pre_calculation_mode
        self.synthetic_mask_pre_calculation_mode = synthetic_mask_pre_calculation_mode

    def _compute_boundary_loss_weight_map(self, outputs, targets, masks):
        # outputs, targets, masks = outputs.to(self.device), targets.to(self.device), masks.to(self.device)
        if self.natural_data_mask_pre_calculation_mode == 'loss' or self.synthetic_mask_pre_calculation_mode == 'loss':
            new_masks = torch.zeros_like(masks, dtype=torch.float32, device=self.device)
            for b in range(masks.shape[0]):
                if not torch.any(masks[b]).item():
                    new_masks[b] = DistanceCalculator(targets[b].unsqueeze(0)).compute_distance_matrix_on_gpu()
                else:
                    new_masks[b] = masks[b]

            masks = new_masks
        targets_array_binary = ImageProcessor.binarize_array(targets.unsqueeze(1).detach())
        output_array_binary = ImageProcessor.binarize_array(outputs.detach())

        difference_coords_s_beside_g = (output_array_binary == 1) & (targets_array_binary == 0)
        difference_coords_g_beside_s = (targets_array_binary == 1) & (output_array_binary == 0)
        # dist_matrix / dist_max -> normalization to (0,1) для истинности расстояний
        res = torch.zeros_like(output_array_binary, dtype=torch.int, device=self.device)
        res[difference_coords_s_beside_g] = -1
        res[difference_coords_g_beside_s] = +1

        return res.squeeze(1) * masks

    def forward(self, outputs, targets, masks):
        masks = self._compute_boundary_loss_weight_map(outputs, targets, masks)
        # masks: torch.Size([b, h, w])
        # outputs:  torch.Size([b,1,h,w])
        s_theta = torch.sigmoid(outputs)
        loss = (masks.unsqueeze(1) * s_theta).sum(dim=(1,2, 3))
        return loss
