from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from other.losses_utils import DistanceCalculator
from torch.utils.data import ConcatDataset

# Import global parameters and helper classes/functions from external modules.
from other.parsing.train_args_parser import *
from SyntheticData_main.run import (
    BackgroundTransformer,
    DataGenerator,
)
from torchvision.transforms import InterpolationMode

class AdaptivePad:
    """
        Pads an image to a target size while preserving aspect ratio, used for validation preprocessing.
    """
    def __init__(self, target_size, fill=0):
        """
        Args:
            target_size (tuple): (width, height) of the target size.
            fill (int or tuple): Padding fill value (default 0 for black).
        """
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        """
        Pads the image to the target size.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Padded image.

        Raises:
            ValueError: If img is not a PIL Image.
        """

        img = img.convert("RGB") if img.mode != "RGB" else img
        w, h = img.size
        target_w, target_h = self.target_size
        target_aspect = target_w / target_h
        aspect_ratio = w / h

        if aspect_ratio > target_aspect:
            # new_h = int(w / target_aspect)
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            padding = (0, pad_top, 0, pad_bottom)
        elif aspect_ratio < target_aspect:
            # new_w = int(h * target_aspect)
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            padding = (pad_left, 0, pad_right, 0)
        else:
            return img


        return transforms.functional.pad(img, padding, fill=self.fill)

###############################################################################
# Custom Background Transform
###############################################################################
class CustomBackgroundTransform:
    """
        Applies affine and photometric transformations to image-label pairs using BackgroundTransformer.
        Designed for data augmentation in training datasets.
    """
    def __init__(self,resize_width, resize_height, rotation_enabled,
                 noise_level_range, noise_type, flip_probability, jitter_probability
                 , saturation_range, hue_range, gamma_probability, gamma_limit, sigmoid_probability, sigmoid_limit,
                 contrast_brightness_probability, brightness_range, contrast_range,
                 clahe_probability, tile_grid_size, clip_limit, motion_blur_probability, motion_blur_limit
                 , full_img_shadow_roi, full_img_num_shadows_limit, full_img_shadow_dimension,
                 full_img_shadow_intensity_range, full_img_shadow_transform_probability,
                 full_img_dropout_probability, full_img_dropout_holes_range, full_img_dropout_height_range,
                 full_img_dropout_width_range, full_img_dropout_mask_fill_value, full_img_dropout_roi):
        """
                Args:
                    resize_width (int): Target width for resizing.
                    resize_height (int): Target height for resizing.
                    rotation_enabled (bool): Enable random rotations.
                    noise_level_range (tuple): Range for noise intensity.
                    noise_type (str): Type of noise ('uniform' or 'gaussian').
                    flip_probability (float): Probability of horizontal flip.
                    jitter_probability (float): Probability of color jitter.
                    saturation_range (tuple): Range for saturation adjustment.
                    hue_range (tuple): Range for hue adjustment.
                    gamma_probability (float): Probability of gamma correction.
                    gamma_limit (tuple): Range for gamma values.
                    sigmoid_probability (float): Probability of sigmoid contrast.
                    sigmoid_limit (tuple): Range for sigmoid steepness.
                    contrast_brightness_probability (float): Probability of contrast/brightness adjustment.
                    brightness_range (tuple): Range for brightness adjustment.
                    contrast_range (tuple): Range for contrast adjustment.
                    clahe_probability (float): Probability of CLAHE enhancement.
                    tile_grid_size (tuple): Tile size for CLAHE.
                    clip_limit (float): Clip limit for CLAHE.
                    motion_blur_probability (float): Probability of motion blur.
                    motion_blur_limit (tuple): Range for motion blur kernel size.
                    full_img_shadow_roi (tuple): ROI for shadow application.
                    full_img_num_shadows_limit (tuple): Range for number of shadows.
                    full_img_shadow_dimension (tuple): Range for shadow dimensions.
                    full_img_shadow_intensity_range (tuple): Range for shadow intensity.
                    full_img_shadow_transform_probability (float): Probability of shadow application.
                    full_img_dropout_probability (float): Probability of dropout.
                    full_img_dropout_holes_range (tuple): Range for number of dropout holes.
                    full_img_dropout_height_range (tuple): Range for dropout hole height.
                    full_img_dropout_width_range (tuple): Range for dropout hole width.
                    full_img_dropout_mask_fill_value (int): Fill value for dropout.
                    full_img_dropout_roi (tuple): ROI for dropout.
                """

        # Instantiate the BackgroundTransformer with provided configuration parameters.
        self.bg_transformer = BackgroundTransformer(
            resize_width, resize_height, rotation_enabled,
            noise_level_range, noise_type, flip_probability, jitter_probability
            , saturation_range, hue_range, gamma_probability, gamma_limit, sigmoid_probability, sigmoid_limit,
            contrast_brightness_probability, brightness_range, contrast_range,
            clahe_probability, tile_grid_size, clip_limit, motion_blur_probability, motion_blur_limit
            , full_img_shadow_roi, full_img_num_shadows_limit, full_img_shadow_dimension,
            full_img_shadow_intensity_range, full_img_shadow_transform_probability,
            full_img_dropout_probability, full_img_dropout_holes_range, full_img_dropout_height_range,
            full_img_dropout_width_range, full_img_dropout_mask_fill_value, full_img_dropout_roi,
        )

    def __call__(self, img, label):
        # def __call__(self, img, label, mask):

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
        # mask = cv2.cvtColor(np.array(mask).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Apply transformations using the BackgroundTransformer.
        # img, label, mask = self.bg_transformer.apply_transformations(img, label, mask)
        img, label = self.bg_transformer.apply_transformations(img, label)

        # Convert the images back to RGB for further processing in PyTorch.
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return img, label


###############################################################################
# Define Base and Input/Target Transforms
###############################################################################

# Initialize our custom transform using global parameters.
custom_transform = CustomBackgroundTransform(
    resize_width=real_resize_width,
    resize_height=real_resize_height,
    rotation_enabled=real_background_rotation,  # Boolean flag for rotation
    noise_level_range=real_noise_level_range,
    noise_type=real_noise_type,
    flip_probability=real_flip_probability,

    jitter_probability=real_jitter_probability,

    saturation_range=real_saturation_range,
    hue_range=real_hue_range,
    gamma_probability=real_gamma_probability,
    gamma_limit=real_gamma_limit,

    sigmoid_probability=real_sigmoid_probability,
    sigmoid_limit=real_sigmoid_limit,

    contrast_brightness_probability=real_contrast_brightness_probability,
    brightness_range=real_brightness_range,
    contrast_range=real_contrast_range,

    clahe_probability=real_clahe_probability,
    tile_grid_size=real_tile_grid_size,
    clip_limit=real_clip_limit,
    motion_blur_probability=real_motion_blur_probability,
    motion_blur_limit=real_motion_blur_limit,

    full_img_shadow_roi=real_shadow_roi,
    full_img_num_shadows_limit=real_num_shadows_limit,
    full_img_shadow_dimension=real_shadow_dimension,
    full_img_shadow_intensity_range=real_shadow_intensity_range,
    full_img_shadow_transform_probability=real_shadow_transform_probability,

    full_img_dropout_probability=real_dropout_probability,
    full_img_dropout_holes_range=real_dropout_holes_range,
    full_img_dropout_height_range=real_dropout_height_range,
    full_img_dropout_width_range=real_dropout_width_range,
    full_img_dropout_mask_fill_value=real_dropout_mask_fill_value,
    full_img_dropout_roi=real_dropout_roi,
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
    AdaptivePad(target_size=(real_resize_width*2, real_resize_height*2), fill=0),
    transforms.Resize((real_resize_height, real_resize_width), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Validation input transform applies resizing with bilinear interpolation and normalization.
val_input_to_tensor_transform = transforms.Compose([
    transforms.ToPILImage(),
    AdaptivePad(target_size=(real_resize_width*2, real_resize_height*2), fill=(0, 0, 0)),
    transforms.Resize((real_resize_height, real_resize_width), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])



# Function that applies the custom background transform.
def apply_custom_transform(img, label):
    return custom_transform(img, label)


###############################################################################
# Triplet Transform for Dataset
###############################################################################
class TripletTransform:
    """
    Applies a sequence of transformations to a triplet of images: the input, the target, and the mask.

    The transform can optionally apply a custom transformation (e.g., data augmentation) before
    converting images into tensors. The mask is processed differently based on a global computing mode.
    """

    def __init__(self, base_transform, input_to_tensor_transform, transform=None):
        # Optional custom transformation function (e.g., custom augmentation).
        self.transform = transform
        # Base transformation (e.g., converting to tensor).
        self.base_transform = base_transform
        # Additional input normalization and tensor conversion.
        self.input_to_tensor_transform = input_to_tensor_transform
        # Mode of computing masks, defined by global natural_data_mask_pre_calculation_mode.
        self.computing = natural_data_mask_pre_calculation_mode


    def __call__(self, input_image, target_image):
        # If a custom transform is provided, apply it to all three images.
        if self.transform:
            input_image, target_image = self.transform(input_image, target_image)
            # input_image, target_image, mask_image = self.transform(input_image, target_image, mask)
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
        # Apply the provided transformations.
        if self.transform:
            input_img, target_img, dist_mask = self.transform(input_img, target_img)
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

    def __init__(self, length=None, test_safe_to_desk: bool = False):
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
        self.synthetic_length = int(self.coco_length * (self.scale_factor - 1))
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


class EvalDataset(Dataset):
    def __init__(self, eval_dir, transform=val_input_to_tensor_transform):
        self.input_path = os.path.join(eval_dir, "input")
        self.files = [f for f in os.listdir(self.input_path) if os.path.isfile(os.path.join(self.input_path, f))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.input_path, self.files[idx])
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        original_h, original_w = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            input_img = self.transform(input_img)
        return input_img, self.files[idx], (original_h, original_w)
