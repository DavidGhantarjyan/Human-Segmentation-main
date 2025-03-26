import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from other.parsing.train_args_parser import *
from other.losses_utils import *


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
        if self.natural_data_mask_pre_calculation_mode == 'loss' and self.synthetic_mask_pre_calculation_mode == 'loss':
            distance_maps = DistanceCalculator(targets).compute_distance_matrix_on_gpu(use_edt=False)
            masks = distance_maps.squeeze(1)
        elif self.natural_data_mask_pre_calculation_mode == 'loss' or self.synthetic_mask_pre_calculation_mode == 'loss':
            mask_any = torch.any(masks, dim=(1, 2))
            indices = torch.nonzero(~mask_any, as_tuple=True)[0]
            targets_to_compute = targets[indices]
            distance_maps = (DistanceCalculator(targets_to_compute).compute_distance_matrix_on_gpu(use_edt=True))
            masks[indices] = distance_maps.squeeze(1)


        targets_array_binary = ImageProcessor.binarize_array(targets.unsqueeze(1))
        output_array_binary = ImageProcessor.binarize_array(outputs.detach())
        res = torch.zeros_like(output_array_binary, dtype=torch.int, device=self.device)
        res = torch.where((output_array_binary > 0) & (targets_array_binary <= 0), -1, res)
        res = torch.where((targets_array_binary > 0) & (output_array_binary <= 0), 1, res)

        return res.squeeze(1) * masks

    def forward(self, outputs, targets, masks):
        masks = self._compute_boundary_loss_weight_map(outputs, targets, masks)
        # masks: torch.Size([b, h, w])
        # outputs:  torch.Size([b,1,h,w])
        s_theta = torch.sigmoid(outputs)
        loss = (masks.unsqueeze(1) * s_theta).sum(dim=(1,2, 3))
        return loss
