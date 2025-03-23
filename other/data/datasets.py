import os
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from other.losses_utils import DistanceCalculator

# Import parameters and necessary classes/functions from other modules
from other.parsing.train_args_parser import *
from SyntheticData_main.run import (
    BackgroundTransformer,
    DataGenerator,
    FileManager,
    background_dir,
    images_per_combination,
)


###############################################################################
# Custom Background Transform
###############################################################################
class CustomBackgroundTransform:
    """
    Applies affine transformations to an image using the BackgroundTransformer.
    """

    def __init__(self,resize_width,resize_height, rotation_enabled,
                 noise_level_range, flip_probability, jitter_probability, brightness_range,
                 contrast_range, saturation_range, hue_range):
        self.bg_transformer = BackgroundTransformer(
            resize_width,
            resize_height,
            rotation_enabled,
            noise_level_range,
            flip_probability,
            jitter_probability,
            brightness_range,
            contrast_range,
            saturation_range,
            hue_range,
        )

    def __call__(self, img, label, mask):
        # Convert image to BGR for OpenCV processing
        img = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGB2BGR)
        label = cv2.cvtColor(np.array(label).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(np.array(mask).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img, label, mask = self.bg_transformer.apply_transformations(img, label, mask)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return img, label, mask


# Initialize custom transform with provided parameters
custom_transform = CustomBackgroundTransform(
    resize_width=resize_width,
    resize_height = resize_height,
    rotation_enabled=background_rotation,
    noise_level_range=noise_level_range,
    flip_probability=flip_probability,
    jitter_probability=jitter_probability,
    brightness_range=brightness_range,
    contrast_range=contrast_range,
    saturation_range=saturation_range,
    hue_range=hue_range,
)

###############################################################################
# Define Base and Input/Target Transforms
###############################################################################
base_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# For synthetic data, apply normalization after base transform
input_to_tensor_transform = transforms.Compose([
    base_transform,
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
def apply_custom_transform(img, label, mask):
    return custom_transform(img, label, mask)

###############################################################################
# Triplet Transform for Dataset
###############################################################################
class TripletTransform:
    """
    Applies sequential transformations to the input, target, and mask images.
    """
    def __init__(self,transform):
        self.transform = transform
        self.base_transform = base_transform
        self.input_to_tensor_transform = input_to_tensor_transform
        # 'computing' mode is determined by the global natural_data_mask_pre_calculation_mode
        self.computing = natural_data_mask_pre_calculation_mode

    def __call__(self, input_image, target_image, mask):
        if isinstance(self.transform, transforms.Compose):
            input_image,target_image, mask = [self.transform(item) for item in [input_image, target_image, mask]]
        else:
            input_image,target_image,mask_image = self.transform(input_image, target_image, mask)
        input_image = self.input_to_tensor_transform(input_image)
        target_image = self.base_transform(target_image)

        if self.computing == 'dataloader':
            dist_calcul = DistanceCalculator(target_image[0].unsqueeze(0) / 255).compute_distance_matrix_on_cpu()
            mask = dist_calcul.squeeze() * 255.0

            mask = self.base_transform(mask.to(dtype=torch.float32))[0]
        elif self.computing == 'loss':
            mask = torch.zeros_like(target_image[0])
        else:
            mask = mask_image
        return input_image, target_image, mask



###############################################################################
# COCO Dataset
###############################################################################
class CocoDataset(Dataset):
    """
    COCO Dataset with precomputed masks.
    """
    def __init__(self, cocodataset_path: str, transform=None):
        # pre_computing depends on the global natural_data_mask_saving flag
        self.pre_computing = natural_data_mask_saving is True
        self.computing = natural_data_mask_pre_calculation_mode
        self.input_path = os.path.join(cocodataset_path, "input")
        self.target_path = os.path.join(cocodataset_path, "target")
        self.mask_path = os.path.join(cocodataset_path, "masks")
        self.input_images = glob(os.path.join(self.input_path, "*.jpg"))
        self.target_images = glob(os.path.join(self.target_path, "*.png"))
        self.mask_images = glob(os.path.join(self.mask_path, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx: int):
        input_img = cv2.imread(self.input_images[idx])
        target_img = cv2.imread(self.target_images[idx])
        # Initialize mask with zeros if not precomputed
        dist_mask = np.zeros_like(target_img)
        if self.pre_computing:
            dist_mask = cv2.imread(self.mask_images[idx])
        if self.transform:
            input_img, target_img, dist_mask = self.transform(input_img, target_img, dist_mask)

            if self.pre_computing:
                # Assume mask has shape [1, H, W]
                dist_mask = dist_mask[0, :, :]
        # Convert input image from BGR to RGB by reordering channels
        input_img = input_img[[2, 1, 0], :, :]
        return input_img, target_img[0, :, :], dist_mask


###############################################################################
# Synthetic Dataset
###############################################################################

class SyntheticDataset(Dataset):
    """
    Synthetic Dataset that generates images on the fly.
    """
    def __init__(self, length = None, test_safe_to_desk: bool = False):
        self.test_safe_to_desk = test_safe_to_desk
        self.generator = iter(DataGenerator(test_safe_to_desk=self.test_safe_to_desk))
        self.input_to_tensor_transform = input_to_tensor_transform
        self.base_transform = base_transform
        self.len = length

    def __len__(self):
        if self.len is None:
            raise Exception("Length of synthetic dataset is None")
        return self.len

    def __getstate__(self):
        state = self.__dict__.copy()
        if "generator" in state:
            del state["generator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.generator = iter(DataGenerator(test_safe_to_desk=self.test_safe_to_desk))
    def __getitem__(self, idx: int):
        input_img, target_img, dist_mask = next(self.generator)
        # Convert image from BGR to RGB
        input_img = cv2.cvtColor(input_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        input_img = self.input_to_tensor_transform(input_img)
        target_img = self.base_transform(target_img)
        dist_mask = self.base_transform(dist_mask)
        # Assume target_img and dist_mask are tensors with shape [C, H, W] and mask is in the first channel
        return input_img, target_img[0, :, :], dist_mask[0, :, :]

###############################################################################
# Mixed Dataset
###############################################################################
class MixedDataset(Dataset):
    """
    Mixed dataset combining COCO and Synthetic datasets.
    """
    def __init__(self, coco_dataset: Dataset, synthetic_dataset: Dataset, scale_factor: float = 1.5):
        self.coco_dataset = coco_dataset
        self.synthetic_dataset = synthetic_dataset
        self.scale_factor = scale_factor
        self.coco_length = len(self.coco_dataset)
        self.synthetic_length = int(self.coco_length * (self.scale_factor-1))
        self.synthetic_dataset.len = self.synthetic_length
        self.total_length = self.coco_length + self.synthetic_length


    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < self.coco_length:
            return self.coco_dataset[idx]
        else:
            synthetic_idx = idx - self.coco_length
            if synthetic_idx < self.synthetic_length:
                return self.synthetic_dataset[synthetic_idx]
            else:
                raise IndexError("Index out of range for MixedDataset.")


train_coco_dataset = CocoDataset(
        cocodataset_path=train_coco_data_dir,
        transform=TripletTransform(
            transform = apply_custom_transform
        ))

synthetic_dataset = SyntheticDataset(
        length=None,
        test_safe_to_desk=False
    )

val_coco_dataset = CocoDataset(
        cocodataset_path=val_coco_data_dir,
        transform=TripletTransform(
            transform = base_transform
        ))

dataset = MixedDataset(train_coco_dataset, synthetic_dataset, scale_factor=1.5)
