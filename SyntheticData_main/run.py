import os
import cv2
import random
import numpy as np
from progressbar import ProgressBar
from other.parsing.train_args_parser import *
from other.losses_utils import ImageProcessor, DistanceCalculator
import torch
from SyntheticData_main.utils import FileManager, ImageUtils, ImageGeneratorUtils


###############################################################################
# BackgroundTransformer
###############################################################################
class BackgroundTransformer:
    """
    Applies a series of affine and photometric transformations to a background image.
    Transformations include horizontal flipping, rotation, color jitter (brightness, contrast,
    saturation, and hue adjustments), noise addition, optional background blur, and resizing.
    These augmentations are applied to both the background image and its associated label/mask.
    """

    def __init__(self, resize_width, resize_height, rotation_enabled,
                 noise_level_range, flip_probability, jitter_probability, brightness_range,
                 contrast_range, saturation_range, hue_range, bg_blur_chance=None, bg_blur_type=None,
                 bg_blur_kernel_range=None):
        # Target dimensions for resizing the image.
        self.resize_width = resize_width
        self.resize_height = resize_height
        # Boolean flag indicating whether rotation is enabled.
        self.rotation_enabled = rotation_enabled
        # Noise level range to apply random noise.
        self.noise_level_range = noise_level_range
        # Probability to perform horizontal flip.
        self.flip_probability = flip_probability
        # Probability to apply color jitter adjustments.
        self.jitter_probability = jitter_probability
        # Ranges for brightness, contrast, saturation, and hue adjustments.
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        # Optional background blur parameters.
        self.bg_blur_chance = bg_blur_chance
        self.bg_blur_type = bg_blur_type
        self.bg_blur_kernel_range = bg_blur_kernel_range
    def apply_transformations(self, img, label, mask=None):
        """
        Applies the same set of transformations to the input image and label (and optionally mask).

        :param img: Input background image as a NumPy array.
        :param label: Label or segmentation map corresponding to the image.
        :param mask: Optional additional mask.
        :return: Tuple (img, label, mask) after applying all transformations.
        """
        # Random horizontal flip with given probability.
        if np.random.rand() < self.flip_probability:
            img = np.fliplr(img)
            label = np.fliplr(label)
            if mask is not None:
                mask = np.fliplr(mask)

        # Determine rotation angle using a utility function (from global config: background_rotation_angle).
        angle = int(ImageUtils.get_rate(background_rotation_angle))
        if self.rotation_enabled:
            # Rotate image with linear interpolation.
            _, img = ImageUtils.image_rotation(img, angle, interpolation_type=cv2.INTER_LINEAR)
            # Rotate label using nearest-neighbor to preserve discrete values.
            _, label = ImageUtils.image_rotation(label, angle, interpolation_type=cv2.INTER_NEAREST)
            if mask is not None:
                _, mask = ImageUtils.image_rotation(mask, angle, interpolation_type=cv2.INTER_NEAREST)

        # Apply color jitter (brightness, contrast, saturation, hue adjustments).
        img = ImageUtils.apply_color_jitter(
            img,
            brightness_range=self.brightness_range,
            contrast_range=self.contrast_range,
            saturation_range=self.saturation_range,
            hue_range=self.hue_range,
            jitter_probability=self.jitter_probability
        )
        # Add random noise to the image.
        img = ImageUtils.add_noise(img, self.noise_level_range)

        # Optionally apply background blur if all blur parameters are set.
        if (self.bg_blur_chance is not None and
                self.bg_blur_type is not None and
                self.bg_blur_kernel_range is not None):
            img = ImageUtils.add_background_blur(img, label, self.bg_blur_chance, self.bg_blur_type,
                                                 self.bg_blur_kernel_range)

        # Resize image, label, and mask to target dimensions.
        img = cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.resize_width, self.resize_height), interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask = cv2.resize(mask, (self.resize_width, self.resize_height), interpolation=cv2.INTER_NEAREST)

        return img, label, mask

###############################################################################
# ObjectAdder
###############################################################################
class ObjectAdder:
    """
    Overlays object images onto a background image.

    Randomly selects a subset of object images and determines their placement on the
    background. Depending on the output format (classification or segmentation), it either
    produces a combined label or a segmentation mask.
    """

    def __init__(self, obj_images, objects_per_image_range, foreground_rotation_angle, output_format):
        # List of available object images.
        self.obj_images = obj_images
        # Tuple defining the minimum and maximum number of objects to place per image.
        self.objects_per_image_range = objects_per_image_range
        # Range for random rotation to be applied to foreground objects.
        self.foreground_rotation_angle = foreground_rotation_angle
        # Output type: either "classification" or "segmentation".
        self.output_format = output_format

    def overlay_objects(self, background_img):
        """
        Overlays randomly selected objects onto the background image.

        :param background_img: Background image as a NumPy array.
        :return: Tuple (edited_img, final_mask) where:
                 - edited_img: Background image with objects overlaid.
                 - final_mask: A segmentation mask or a classification label depending on configuration.
        """
        # Copy the background image to avoid modifying the original.
        edited_img = background_img.copy()
        final_out = []  # To collect segmentation masks for each object.
        placement_params = []  # Parameters (e.g., position, scale) for object placement.

        # Determine random number of objects to overlay.
        num_objects = random.randint(*self.objects_per_image_range)
        # Randomly sample objects from the available list.
        sample_objects = random.sample(self.obj_images, num_objects)
        # Generate placement parameters for each selected object.
        for obj in sample_objects:
            placement_params.append(ImageGeneratorUtils.generate_point(edited_img, obj))

        # Sort placements based on y-coordinate to manage occlusion (objects lower in the image may cover those above).
        sorted_params = sorted(placement_params, key=lambda item: item[1] + item[2].shape[0])
        label = None

        # Overlay each object onto the background.
        for params in sorted_params:
            edited_img, out = ImageGeneratorUtils.generate_img(edited_img, params, self.foreground_rotation_angle)
            if self.output_format == "classification":
                label = out  # For classification, a single label may suffice.
            elif self.output_format == "segmentation":
                final_out.append(out)  # For segmentation, combine all masks.

        # If segmentation output is expected, combine individual object masks.
        if self.output_format == "segmentation":
            final_mask = np.maximum.reduce(final_out)
        else:
            final_mask = label

        return edited_img, final_mask

###############################################################################
# DataGenerator
###############################################################################
class DataGenerator:
    """
    Generates synthetic data by overlaying objects onto background images and applying transformations.

    It can optionally save the generated images, labels, and masks to disk.
    """

    def __init__(self, test_safe_to_desk=False):
        # Initialize the background transformer with parameters from the configuration.
        self.bg_transformer = BackgroundTransformer(
            resize_width,
            resize_height,
            background_rotation,         # Boolean flag for rotation
            noise_level_range,
            flip_probability,
            jitter_probability,
            brightness_range,
            contrast_range,
            saturation_range,
            hue_range,
            bg_blur_probability,
            bg_blur_type,
            bg_blur_kernel_range
        )

        # Flag indicating whether to save generated data to disk.
        self.save_to_disk = test_safe_to_desk
        self.name_amount = 0  # Counter for naming generated files.

        # If saving to disk, validate and create necessary directories.
        if self.save_to_disk:
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)
            if not os.path.isdir(background_dir):
                raise Exception("Wrong BACKGROUND_DIR")
            if not os.path.isdir(objects_dir):
                raise Exception("Wrong OBJECTS_DIR")
            # Create a new epoch directory for this run.
            self.ep_dir = FileManager.get_new_epoch_path()
            os.makedirs(self.ep_dir, exist_ok=True)
            if merge_outputs:
                self.data_path = self.ep_dir
                self.out_path = self.ep_dir
                self.mask_path = self.ep_dir
            else:
                self.data_path = os.path.join(self.ep_dir, "data") + os.sep
                self.out_path = os.path.join(self.ep_dir, "label") + os.sep
                self.mask_path = os.path.join(self.ep_dir, "masks") + os.sep
                os.makedirs(self.data_path, exist_ok=True)
                os.makedirs(self.out_path, exist_ok=True)
                os.makedirs(self.mask_path, exist_ok=True)

        # Load background image names from the designated directory.
        self.bg_names = os.listdir(background_dir)
        # Load object images using a file manager utility.
        self.obj_images = FileManager.get_objects()

        # Initialize the object adder with object images and transformation parameters.
        self.object_adder = ObjectAdder(
            self.obj_images,
            objects_per_image_range,
            foreground_rotation_angle,
            output_format
        )

    def __iter__(self):
        """
        Infinite generator that yields synthetic data samples.

        Each sample is produced by:
          1. Randomly selecting a background image.
          2. Optionally creating subdirectories based on background name if packaging is enabled.
          3. Overlaying objects onto the background.
          4. Applying photometric and affine transformations to the composite image.
          5. Optionally pre-calculating a distance matrix for segmentation masks.
          6. Saving the generated images, labels, and masks to disk if enabled.

        :yield: A tuple (transformed_img, transformed_mask, distance_matrix) for segmentation output,
                or (transformed_img, label) for classification output.
        """
        while True:
            # Randomly select a background image.
            bg_name = random.choice(self.bg_names)
            bg_path = os.path.join(background_dir, bg_name)
            bg = cv2.imread(str(bg_path))

            # Set current output paths based on whether saving to disk is enabled.
            curr_data_path = self.data_path if self.save_to_disk else None
            curr_out_path = self.out_path if self.save_to_disk else None
            if package_by_background and self.save_to_disk:
                # Create a subfolder for outputs based on the background image name.
                subfolder = bg_name.replace(".", "__")
                curr_data_path = os.path.join(self.data_path, subfolder) + os.sep
                curr_out_path = os.path.join(self.out_path, subfolder) + os.sep
                os.makedirs(curr_data_path, exist_ok=True)
                if not merge_outputs:
                    os.makedirs(curr_out_path, exist_ok=True)

            # Overlay objects onto the background image.
            composed_img, mask_or_label = self.object_adder.overlay_objects(bg)

            # Apply background transformations (affine and photometric).
            transformed_img, transformed_mask, _ = self.bg_transformer.apply_transformations(composed_img,
                                                                                             mask_or_label)

            if output_format == "segmentation":
                # Convert mask to binary format.
                transformed_mask = (transformed_mask > 0).astype(np.uint8) * 255
                # Convert mask to a tensor for distance calculation.
                grayscale_tensor = torch.from_numpy(transformed_mask[:, :, 0]).unsqueeze(0)
                dist_calcul = np.zeros(shape=grayscale_tensor.squeeze().shape)
                if synthetic_mask_pre_calculation_mode == 'dataloader':
                    # Compute a distance matrix based on the mask.
                    dist_calcul = DistanceCalculator(grayscale_tensor / 255).compute_distance_matrix_on_cpu()
                    dist_calcul = dist_calcul.squeeze() * 255.0
                    dist_calcul = dist_calcul.numpy().astype(np.float32)

                # If saving to disk, save the transformed images, labels, and masks.
                if self.save_to_disk:
                    FileManager.save_as_grayscale_img(
                        transformed_mask,
                        os.path.join(curr_out_path, f"{self.name_amount}.png")
                    )
                    cv2.imwrite(os.path.join(curr_data_path, f"{self.name_amount}.jpg"), transformed_img)

                    if synthetic_mask_pre_calculation_mode == 'dataloader':
                            FileManager.save_as_grayscale_img(
                                dist_calcul.astype(np.uint8),
                                os.path.join(self.mask_path, f"{self.name_amount}.png")
                            )
                # Yield the synthetic sample.
                yield transformed_img, transformed_mask, dist_calcul

            else:
                yield transformed_img, mask_or_label

            self.name_amount += 1


###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    # Total number of synthetic samples to generate.
    total_iterations = 10000

    # Create an instance of the data generator with the flag to save data to disk if specified.
    data_generator = DataGenerator(test_safe_to_desk=test_safe_to_desk)

    # Initialize a progress bar for monitoring generation progress.
    progress_bar = ProgressBar(max_value=total_iterations)

    # Iterate over generated data samples.
    for iteration, (input_img, target, dist_mask) in enumerate(data_generator):
        progress_bar.update(iteration + 1)

        if iteration + 1 >= total_iterations:
            break


