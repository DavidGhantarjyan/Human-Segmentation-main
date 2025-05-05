import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from other.data.processing import get_train_val_dataloaders
from other.data.datasets import (CocoDataset, SyntheticDataset, MixedDataset, TripletTransform,
                                 base_transform, apply_custom_transform, input_to_tensor_transform,
                                 val_base_transform, val_input_to_tensor_transform)


if __name__ == '__main__':
    # Load training arguments from parser. This includes all hyperparameters and directory paths.
    from other.parsing.train_args_parser import *

    # Print a separator to indicate the start of a training run in the console log.
    print(f"\n{'=' * 100}\n")
    # -------------------------------
    # Data Preparation
    # -------------------------------
    # Create a COCO-format dataset for training with a specified set of transformations.
    train_coco_dataset = CocoDataset(
        cocodataset_path=train_coco_data_dir,
        transform=TripletTransform(
            transform=apply_custom_transform,  # Custom augmentation function for foreground objects.
            base_transform=base_transform,  # Base transformation applied to the input images.
            input_to_tensor_transform=input_to_tensor_transform  # Converts inputs to PyTorch tensors.
        )
    )
    train_mapillary_dataset = CocoDataset(
        cocodataset_path=train_mapillary_data_dir,
        transform=TripletTransform(
            transform=apply_custom_transform,
            base_transform=base_transform,
            input_to_tensor_transform=input_to_tensor_transform
        )
    )
    # Create a synthetic dataset (for example, generated images or data augmentation).
    synthetic_dataset = SyntheticDataset(
        length=None,  # If None, use the default or full dataset length.
        test_safe_to_desk=False,  # Whether to persist generated data to disk.
    )

    # Create a validation COCO dataset with validation-specific transformations.
    val_coco_dataset = CocoDataset(
        cocodataset_path=val_coco_data_dir,
        transform=TripletTransform(
            transform=None,  # No extra augmentation on validation images.
            base_transform=val_base_transform,  # Base validation transformation.
            input_to_tensor_transform=val_input_to_tensor_transform  # Convert validation inputs to tensors.
        )
    )
    combined_dataset = ConcatDataset([train_coco_dataset, train_mapillary_dataset])

    # Combine training datasets (real + synthetic)
    mixed_dataset = MixedDataset(combined_dataset, synthetic_dataset, scale_factor=scale_factor)
    print(f"Total samples in mixed dataset: {len(mixed_dataset)}")

    # Create dataloaders
    train_dataloader, val_dataloader, _, _ = get_train_val_dataloaders(
            train_dataset=mixed_dataset,
            val_dataset=val_coco_dataset,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            val_num_workers=val_num_workers,
        )


    # -------------------------------
    # Compute Class Imbalance for Focal Loss
    # -------------------------------
    total_pixels = 0
    positive_pixels = 0
    for _, targets, _ in tqdm(train_dataloader, desc="Computing positive pixel ratio"):
        # Assuming binary masks where positive class is labeled as 1
        targets = targets.to("cpu")
        positive_pixels += (targets == 1).sum().item()
        total_pixels += targets.numel()

    positive_ratio = positive_pixels / total_pixels if total_pixels > 0 else 0
    print(f"Positive pixel ratio (class == 1): {positive_ratio:.4f}")