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

    def __init__(self, rotation_enabled, blur_kernel_range, blur_intensity_range, blur_probability,
                 noise_level_range, flip_probability, jitter_probability, brightness_range,
                 contrast_range, saturation_range, hue_range):
        self.bg_transformer = BackgroundTransformer(
            rotation_enabled,
            blur_kernel_range,
            blur_intensity_range,
            blur_probability,
            noise_level_range,
            flip_probability,
            jitter_probability,
            brightness_range,
            contrast_range,
            saturation_range,
            hue_range,
        )

    def __call__(self, img, label_apply: bool) -> np.ndarray:
        """
        Applies transformations to the input image.

        :param img: Input image as a PIL Image or NumPy array.
        :param label_apply: Flag indicating if label-dependent transformations should be applied.
        :return: Transformed image as a NumPy array in RGB format.
        """
        # Convert image to BGR for OpenCV processing
        img = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGB2BGR)
        transformed_cv_img = self.bg_transformer.apply_transformations(img, label_apply=label_apply)
        # Convert back to RGB
        transformed_cv_img = cv2.cvtColor(transformed_cv_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return transformed_cv_img


# Initialize custom transform with provided parameters
custom_transform = CustomBackgroundTransform(
    rotation_enabled=background_rotation,
    blur_kernel_range=blur_kernel_range,
    blur_intensity_range=blur_intensity_range,
    blur_probability=blur_probability,
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
    transforms.Resize((160, 90)),
    transforms.ToTensor(),
])

# For synthetic data, apply normalization after base transform
input_transform_synthetic = transforms.Compose([
    base_transform,
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define module-level functions instead of lambda functions for picklability
def apply_custom_transform_input(img, label_apply=False):
    return custom_transform(img, label_apply=label_apply)

def apply_custom_transform_target(img, label_apply=True):
    return custom_transform(img, label_apply=label_apply)

# Compose input transform using custom transformation followed by synthetic input transform
input_transform = transforms.Compose([
    apply_custom_transform_input,
    input_transform_synthetic,
])

# Compose target transform using custom transformation followed by base transform
target_transform = transforms.Compose([
    apply_custom_transform_target,
    base_transform,
])
# Use the same transformation for mask as for the target
mask_transform = target_transform

###############################################################################
# Triplet Transform for Dataset
###############################################################################
class TripletTransform:
    """
    Applies sequential transformations to the input, target, and mask images.
    """
    def __init__(self, input_transform=None, target_transform=None, mask_transform=None):
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.base_transform = base_transform
        # 'computing' mode is determined by the global natural_data_mask_pre_calculation_mode
        self.computing = natural_data_mask_pre_calculation_mode

    def __call__(self, input_image, target_image, mask):
        # Save numpy random state to synchronize random transformations
        np_random_state = np.random.get_state()

        if self.input_transform:
            input_image = self.input_transform(input_image)

        # Reset random state to ensure same transformation for target
        np.random.set_state(np_random_state)
        if self.target_transform:
            target_image = self.target_transform(target_image)

        # Reset random state for mask transformation or distance calculation
        np.random.set_state(np_random_state)
        #ПРИОРИТЕТ ИМЕЕТ ИМЕННО
        if self.computing == 'dataloader':

            # Compute distance mask on the fly using DistanceCalculator
            dist_calcul = DistanceCalculator(target_image[0].unsqueeze(0) / 255).compute_distance_matrix_on_cpu()
            mask = dist_calcul.squeeze() * 255.0
            # *********
            mask = self.base_transform(mask.to(dtype=torch.uint8))[0]
        elif self.computing == 'loss':
            # In 'loss' mode, use a zero mask
            mask = torch.zeros_like(target_image[0])
        # elif self.mask_transform:
        #     mask = self.mask_transform(mask)

        return input_image, target_image, mask



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

    def __init__(self, rotation_enabled, blur_kernel_range, blur_intensity_range, blur_probability,
                 noise_level_range, flip_probability, jitter_probability, brightness_range,
                 contrast_range, saturation_range, hue_range):
        self.bg_transformer = BackgroundTransformer(
            rotation_enabled,
            blur_kernel_range,
            blur_intensity_range,
            blur_probability,
            noise_level_range,
            flip_probability,
            jitter_probability,
            brightness_range,
            contrast_range,
            saturation_range,
            hue_range,
        )

    def __call__(self, img, label_apply: bool) -> np.ndarray:
        """
        Applies transformations to the input image.

        :param img: Input image as a PIL Image or NumPy array.
        :param label_apply: Flag indicating if label-dependent transformations should be applied.
        :return: Transformed image as a NumPy array in RGB format.
        """
        # Convert image to BGR for OpenCV processing
        img = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_RGB2BGR)
        transformed_cv_img = self.bg_transformer.apply_transformations(img, label_apply=label_apply)
        # Convert back to RGB
        transformed_cv_img = cv2.cvtColor(transformed_cv_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return transformed_cv_img


# Initialize custom transform with provided parameters
custom_transform = CustomBackgroundTransform(
    rotation_enabled=background_rotation,
    blur_kernel_range=blur_kernel_range,
    blur_intensity_range=blur_intensity_range,
    blur_probability=blur_probability,
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
    transforms.Resize((160, 90)),
    transforms.ToTensor(),
])

# For synthetic data, apply normalization after base transform
input_transform_synthetic = transforms.Compose([
    base_transform,
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define module-level functions instead of lambda functions for picklability
def apply_custom_transform_input(img, label_apply=False):
    return custom_transform(img, label_apply=label_apply)

def apply_custom_transform_target(img, label_apply=True):
    return custom_transform(img, label_apply=label_apply)

# Compose input transform using custom transformation followed by synthetic input transform
input_transform = transforms.Compose([
    apply_custom_transform_input,
    input_transform_synthetic,
])

# Compose target transform using custom transformation followed by base transform
target_transform = transforms.Compose([
    apply_custom_transform_target,
    base_transform,
])
# Use the same transformation for mask as for the target
mask_transform = target_transform

###############################################################################
# Triplet Transform for Dataset
###############################################################################
class TripletTransform:
    """
    Applies sequential transformations to the input, target, and mask images.
    """
    def __init__(self, input_transform=None, target_transform=None, mask_transform=None):
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.base_transform = base_transform
        # 'computing' mode is determined by the global natural_data_mask_pre_calculation_mode
        self.computing = natural_data_mask_pre_calculation_mode


    def __call__(self, input_image, target_image, mask):
        # Save numpy random state to synchronize random transformations
        np_random_state = np.random.get_state()

        if self.input_transform:
            input_image = self.input_transform(input_image)

        # Reset random state to ensure same transformation for target
        np.random.set_state(np_random_state)
        if self.target_transform:
            target_image = self.target_transform(target_image)

        # Reset random state for mask transformation or distance calculation
        np.random.set_state(np_random_state)

        # mask@ (500, 381, 3) kam zero kam uint8
        # irakaknum qani vor kam self.computing = natural_data_mask_pre_calculation_mode kam che
        if self.computing == 'dataloader':
            # Compute distance mask on the fly using DistanceCalculator
            dist_calcul = DistanceCalculator(target_image[0].unsqueeze(0) / 255).compute_distance_matrix_on_cpu()
            mask = dist_calcul.squeeze() * 255.0
            # *********
            mask = self.base_transform(mask.to(dtype=torch.float32))[0]
            print(mask.shape)
        elif self.computing == 'loss':
            # In 'loss' mode, use a zero mask
            mask = torch.zeros_like(target_image[0])
        # elif self.mask_transform:
        #     mask = self.mask_transform(mask)

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
    # #     train_coco_dataset = CocoDataset(
    # #         cocodataset_path=train_coco_data_dir,
    # #         transform=TripletTransform(
    # #             input_transform=input_transform,
    # #             target_transform=target_transform,
    # #             mask_transform=mask_transform
    # #         ))
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx: int):
        input_img = cv2.imread(self.input_images[idx])
        target_img = cv2.imread(self.target_images[idx])
        # Initialize mask with zeros if not precomputed
        dist_mask = np.zeros_like(target_img)
        if self.pre_computing:
            # Аrden uint8 formati-ya, ete goyutyun uni
            # (500, 381, 3)
            dist_mask = cv2.imread(self.mask_images[idx])
        if self.transform:
            input_img, target_img, dist_mask = self.transform(input_img, target_img, dist_mask)
            if self.pre_computing:
                # Assume mask has shape [1, H, W]
                dist_mask = dist_mask[0, :, :]
        # Convert input image from BGR to RGB by reordering channels
        input_img = input_img[[2, 1, 0], :, :]
        return input_img, target_img[0, :, :], dist_mask
  # torch.Size([3, 640, 360]) torch.Size([640, 360]) torch.Size([640, 360])


###############################################################################
# Synthetic Dataset
###############################################################################
class SyntheticDataset(Dataset):
    """
    Synthetic Dataset that generates images on the fly.
    """
    def __init__(self, test_safe_to_desk: bool = False, input_transform=None, target_transform=None):
        self.test_safe_to_desk = test_safe_to_desk
        self.generator = iter(DataGenerator(test_safe_to_desk=self.test_safe_to_desk))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        num_objects = len(FileManager.get_objects())
        num_backgrounds = len(os.listdir(background_dir))
        return num_objects * num_backgrounds * images_per_combination

    def __getstate__(self):
        state = self.__dict__.copy()
        if "generator" in state:
            del state["generator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.generator = iter(DataGenerator(test_safe_to_desk=self.test_safe_to_desk))
    # #     synthetic_dataset = SyntheticDataset(
    # #         test_safe_to_desk=False,
    # #         target_transform=base_transform,
    # #         input_transform=input_transform_synthetic
    # #     )
    def __getitem__(self, idx: int):
        input_img, target_img, dist_mask = next(self.generator)
        # Convert image from BGR to RGB
        input_img = cv2.cvtColor(input_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if self.input_transform:
            input_img = self.input_transform(input_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)
            dist_mask = self.target_transform(dist_mask)
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
        self.synthetic_len = len(self.synthetic_dataset)
        self.coco_len = len(self.coco_dataset)
        self.new_size = int(self.coco_len * scale_factor) if \
            int(self.coco_len * scale_factor - self.coco_len) <= self.synthetic_len else self.synthetic_len + self.coco_len
        # print("Synthetic dataset length:", self.synthetic_len)
        # print("COCO dataset length:", self.coco_len)
        # print("Mixed dataset length:", self.new_size)

    def __len__(self):
        return self.new_size

    def __getitem__(self, idx):
        if idx < self.coco_len:
            return self.coco_dataset[idx]
        else:
            return self.synthetic_dataset[idx]

#     train_coco_dataset = CocoDataset(
#         cocodataset_path=train_coco_data_dir,
#         transform=TripletTransform(
#             input_transform=input_transform,
#             target_transform=target_transform,
#             mask_transform=mask_transform
#         ))
#
#     synthetic_dataset = SyntheticDataset(
#         test_safe_to_desk=False,
#         target_transform=base_transform,
#         input_transform=input_transform_synthetic
#     )
#
#     val_coco_dataset = CocoDataset(
#         cocodataset_path=val_coco_data_dir,
#         transform=TripletTransform(
#             input_transform=base_transform,
#             target_transform=base_transform,
#             mask_transform=base_transform
#         ))


###############################################################################
# Dataset Initialization and Visualization
###############################################################################
synthetic_dataset = SyntheticDataset(
    test_safe_to_desk=False,
    target_transform=base_transform,
    input_transform=input_transform_synthetic
)

coco_dataset = CocoDataset(
    cocodataset_path=train_coco_data_dir,
    transform=TripletTransform(
        input_transform=input_transform,
        target_transform=target_transform,
        mask_transform=mask_transform
    ),
    # pre_computing=natural_data_mask_saving
)

val_dataset = CocoDataset(
    cocodataset_path=val_coco_data_dir,
    transform=TripletTransform(
        input_transform=base_transform,
        target_transform=base_transform,
        mask_transform=base_transform
    ),
    # pre_computing=natural_data_mask_saving
)

dataset = MixedDataset(coco_dataset, synthetic_dataset, scale_factor=1.5)

# Visualize examples from the synthetic dataset
for i in range(5):
    input_img, target_img, dist_img = coco_dataset[i]
    # Print min and max values for debugging purposes
    # print("Input:", input_img.min(), input_img.max())
    # print("Target:", target_img.min(), target_img.max())
    # print("Mask:", dist_img.min(), dist_img.max())

    # If input_img is a tensor, unnormalize it and convert to a NumPy array for display
    if isinstance(input_img, torch.Tensor):
        unnormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        input_img = unnormalize(input_img)
        input_np = input_img.permute(1, 2, 0).numpy()
        input_np = (input_np * 255).astype(np.uint8)
        input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
    else:
        input_bgr = input_img

    if isinstance(target_img, torch.Tensor):
        target_np = target_img.numpy()
        target_np = (target_np * 255).astype(np.uint8)
        target_bgr = cv2.cvtColor(target_np, cv2.COLOR_GRAY2BGR)
    else:
        target_bgr = target_img

    if isinstance(dist_img, torch.Tensor):
        dist_np = dist_img.numpy()
        dist_np = (dist_np * 255).astype(np.uint8)
        dist_bgr = cv2.cvtColor(dist_np, cv2.COLOR_GRAY2BGR)
    else:
        dist_bgr = dist_img

    combined = np.hstack((input_bgr, target_bgr, dist_bgr))
    cv2.imshow("Input (left), Target (middle), Mask (right)", combined)
    key = cv2.waitKey(0)
    if key == 27:  # Exit if ESC is pressed
        break

cv2.destroyAllWindows()
