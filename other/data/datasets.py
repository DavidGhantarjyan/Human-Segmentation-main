from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from other.losses_utils import DistanceCalculator

# Import global parameters and helper classes/functions from external modules.
from other.parsing.train_args_parser import *
from SyntheticData_main.run import (
    BackgroundTransformer,
    DataGenerator,
)
from torchvision.transforms import InterpolationMode


###############################################################################
# Custom Background Transform
###############################################################################
class CustomBackgroundTransform:
    """
    Applies affine and photometric transformations to a background image by leveraging
    the BackgroundTransformer. This custom transform is designed to work on triplets of
    images (input, label, mask) by converting them to the required format, applying OpenCV
    transformations, and then converting them back.
    """
    def __init__(self,resize_width,resize_height, rotation_enabled,
                 noise_level_range, flip_probability, jitter_probability, brightness_range,
                 contrast_range, saturation_range, hue_range):
        # Instantiate the BackgroundTransformer with provided configuration parameters.
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
        """
        Converts input images from PIL (or array) format to OpenCV BGR format, applies the
        background transformations, then converts the transformed images back to RGB.

        :param img: Input background image.
        :param label: Corresponding label image.
        :param mask: Associated mask image.
        :return: Tuple (img, label, mask) after transformation.
        """
        # Convert the images from RGB to BGR for OpenCV processing.
        img = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGB2BGR)
        label = cv2.cvtColor(np.array(label).astype(np.uint8), cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(np.array(mask).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Apply transformations using the BackgroundTransformer.
        img, label, mask = self.bg_transformer.apply_transformations(img, label, mask)
        # Convert the images back to RGB for further processing in PyTorch.
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return img, label, mask

###############################################################################
# Define Base and Input/Target Transforms
###############################################################################

# Initialize our custom transform using global parameters.
custom_transform = CustomBackgroundTransform(
    resize_width=resize_width,
    resize_height=resize_height,
    rotation_enabled=background_rotation,
    noise_level_range=noise_level_range,
    flip_probability=flip_probability,
    jitter_probability=jitter_probability,
    brightness_range=brightness_range,
    contrast_range=contrast_range,
    saturation_range=saturation_range,
    hue_range=hue_range,
)

# The base transform converts a PIL image to a tensor.
base_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# For synthetic data, normalization is applied after converting to tensor.
input_to_tensor_transform = transforms.Compose([
    base_transform,
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Validation base transform includes resizing using nearest interpolation.
val_base_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize_height, resize_width), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Validation input transform applies resizing with bilinear interpolation and normalization.
val_input_to_tensor_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize_height, resize_width), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function that applies the custom background transform.
def apply_custom_transform(img, label, mask):
    return custom_transform(img, label, mask)

###############################################################################
# Triplet Transform for Dataset
###############################################################################
class TripletTransform:
    """
    Applies a sequence of transformations to a triplet of images: the input, the target, and the mask.

    The transform can optionally apply a custom transformation (e.g., data augmentation) before
    converting images into tensors. The mask is processed differently based on a global computing mode.
    """
    def __init__(self, base_transform,input_to_tensor_transform,transform=None):
        # Optional custom transformation function (e.g., custom augmentation).
        self.transform = transform
        # Base transformation (e.g., converting to tensor).
        self.base_transform = base_transform
        # Additional input normalization and tensor conversion.
        self.input_to_tensor_transform = input_to_tensor_transform
        # Mode of computing masks, defined by global natural_data_mask_pre_calculation_mode.
        self.computing = natural_data_mask_pre_calculation_mode

    def __call__(self, input_image, target_image, mask):
        # If a custom transform is provided, apply it to all three images.
        if self.transform:
            input_image,target_image,mask_image = self.transform(input_image, target_image, mask)
        # Convert the input image using the normalization transform.
        input_image = self.input_to_tensor_transform(input_image)
        # Convert the target image using the base transform.
        target_image = self.base_transform(target_image)
        # Process the mask based on the mode:
        # - 'dataloader': Precompute a distance matrix for the mask.
        # - 'loss': Use a zero mask.
        # - Otherwise, use the base transform on the mask image.
        if self.computing == 'dataloader':
            dist_calcul = DistanceCalculator(target_image[0].unsqueeze(0) / 255).compute_distance_matrix_on_cpu()
            mask = dist_calcul.squeeze() * 255.0

            mask = self.base_transform(mask.to(dtype=torch.float32))[0]
        elif self.computing == 'loss':
            mask = torch.zeros_like(target_image[0])
        else:
            mask = base_transform(mask_image)[0].squeeze()
        # Return the transformed triplet.
        return input_image, target_image, mask



###############################################################################
# COCO Dataset
###############################################################################
class CocoDataset(Dataset):
    """
    COCO Dataset that loads precomputed masks if available.

    The dataset expects images to be stored in separate folders for inputs, targets, and masks.
    """
    def __init__(self, cocodataset_path: str, transform=None):
        # Determine whether to use precomputed masks based on global configuration.
        self.pre_computing = natural_data_mask_saving is True
        # Set the paths for input images, target images, and masks.
        self.computing = natural_data_mask_pre_calculation_mode
        self.input_path = os.path.join(cocodataset_path, "input")
        self.target_path = os.path.join(cocodataset_path, "target")
        self.mask_path = os.path.join(cocodataset_path, "masks")
        # Retrieve all image file paths using glob.
        self.input_images = glob(os.path.join(self.input_path, "*.jpg"))
        self.target_images = glob(os.path.join(self.target_path, "*.png"))
        self.mask_images = glob(os.path.join(self.mask_path, "*.png"))
        # Store the transformation function.
        self.transform = transform

    def __len__(self):
        # The length of the dataset is determined by the number of input images.
        return len(self.input_images)

    def __getitem__(self, idx: int):
        # Load images from disk using OpenCV.
        input_img = cv2.imread(self.input_images[idx])
        target_img = cv2.imread(self.target_images[idx])
        # Use precomputed mask if available; otherwise, create a zero mask.
        dist_mask = cv2.imread(self.mask_images[idx]) if self.pre_computing else np.zeros_like(target_img)
        # Apply the provided transformations.
        if self.transform:
            input_img, target_img, dist_mask = self.transform(input_img, target_img, dist_mask)
            if self.pre_computing:
                # Assume the mask is a tensor with shape [1, H, W] after transformation.
                dist_mask = dist_mask[0, :, :]
        # Convert the input image from BGR (OpenCV default) to RGB.
        input_img = input_img[[2, 1, 0], :, :]
        # Return the input tensor, target tensor (first channel), and mask.
        return input_img, target_img[0, :, :], dist_mask

###############################################################################
# Synthetic Dataset
###############################################################################

class SyntheticDataset(Dataset):
    """
    Synthetic Dataset that generates images on the fly using a data generator.

    The dataset leverages a DataGenerator instance to produce synthetic samples.
    """
    def __init__(self, length = None, test_safe_to_desk: bool = False):
        # Flag indicating whether to save generated samples to disk.
        self.test_safe_to_desk = test_safe_to_desk
        # Initialize the generator as an iterator over the DataGenerator.
        self.generator = iter(DataGenerator(test_safe_to_desk=self.test_safe_to_desk))
        self.input_to_tensor_transform = input_to_tensor_transform
        self.base_transform = base_transform
        self.len = length

    def __len__(self):
        # Raise an exception if length is not set.
        if self.len is None:
            raise Exception("Length of synthetic dataset is None")
        return self.len

    def __getstate__(self):
        # Ensure the generator is not pickled.
        state = self.__dict__.copy()
        if "generator" in state:
            del state["generator"]
        return state

    def __setstate__(self, state):
        # Reinitialize the generator upon unpickling.
        self.__dict__.update(state)
        self.generator = iter(DataGenerator(test_safe_to_desk=self.test_safe_to_desk))

    def __getitem__(self, idx: int):
        # Generate the next synthetic sample.
        input_img, target_img, dist_mask = next(self.generator)
        # Convert the generated image from BGR to RGB.
        input_img = cv2.cvtColor(input_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        input_img = self.input_to_tensor_transform(input_img)
        target_img = self.base_transform(target_img)
        dist_mask = self.base_transform(dist_mask)
        # Assume target_img and dist_mask are tensors with shape [C, H, W];
        # return the first channel as the target mask.
        return input_img, target_img[0, :, :], dist_mask[0, :, :]

###############################################################################
# Mixed Dataset
###############################################################################
class MixedDataset(Dataset):
    """
    Combines COCO and Synthetic datasets into a single dataset.

    The COCO dataset forms the primary dataset and the synthetic dataset is added to
    augment the total number of samples. The scale_factor determines the proportion of synthetic data.
    """
    def __init__(self, coco_dataset: Dataset, synthetic_dataset: Dataset, scale_factor: float = 1.5):
        self.coco_dataset = coco_dataset
        self.synthetic_dataset = synthetic_dataset
        self.scale_factor = scale_factor
        # Calculate lengths for COCO and synthetic parts.
        self.coco_length = len(self.coco_dataset)
        self.synthetic_length = int(self.coco_length * (self.scale_factor-1))
        # Set the length of the synthetic dataset.
        self.synthetic_dataset.len = self.synthetic_length
        # Total dataset length is the sum of COCO and synthetic samples.
        self.total_length = self.coco_length + self.synthetic_length


    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # If the index is within the COCO dataset, retrieve from it.
        if idx < self.coco_length:
            with torch.profiler.record_function("## Get from CocoDataset ## "):
                return self.coco_dataset[idx]
        else:
            # Otherwise, retrieve from the synthetic dataset.
            synthetic_idx = idx - self.coco_length
            if synthetic_idx < self.synthetic_length:
                with torch.profiler.record_function("##  Get from SyntheticDataset ## "):
                    return self.synthetic_dataset[synthetic_idx]
            else:
                raise IndexError("Index out of range for MixedDataset.")

###############################################################################
# Dataset Initialization Examples
###############################################################################
# Initialize the validation COCO dataset with validation-specific transforms.
val_coco_dataset = CocoDataset(
        cocodataset_path =val_coco_data_dir,
        transform=TripletTransform(
            transform=None,base_transform=val_base_transform,input_to_tensor_transform=val_input_to_tensor_transform
        ))
# Initialize the synthetic dataset without saving generated samples to disk.
synthetic_dataset = SyntheticDataset(
        length=None,
        test_safe_to_desk=False,

    )
# Initialize the training COCO dataset with a custom augmentation transform.
train_coco_dataset = CocoDataset(
        cocodataset_path=train_coco_data_dir,
        transform=TripletTransform(
            transform=apply_custom_transform,base_transform=base_transform,input_to_tensor_transform=input_to_tensor_transform
        ))
# Create the mixed dataset combining COCO and synthetic samples using the provided scale factor.
mixed_dataset = MixedDataset(train_coco_dataset, synthetic_dataset, scale_factor=1.5)
