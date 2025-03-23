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

    The transformations include flipping, rotation, color jitter, blur, and noise.
    """

    def __init__(self, rotation_enabled, blur_kernel_range, blur_intensity_range, blur_probability,
                 noise_level_range, flip_probability, jitter_probability, brightness_range,
                 contrast_range, saturation_range, hue_range):
        self.rotation_enabled = rotation_enabled
        self.blur_kernel_range = blur_kernel_range
        self.blur_intensity_range = blur_intensity_range
        self.blur_probability = blur_probability
        self.noise_level_range = noise_level_range
        self.flip_probability = flip_probability
        self.jitter_probability = jitter_probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def apply_transformations(self, img, label_apply=False):
        """
        Applies the configured transformations to the input image.

        :param img: Input image (as a NumPy array).
        :param label_apply: Boolean flag. If True, label-dependent transformations are applied.
        :return: Transformed image.
        """
        # Flip the image based on the configured flip probability.
        img = ImageUtils.flip_img(img, self.flip_probability)
        # Get a rotation angle based on the background_rotation_angle global variable.
        angle = int(ImageUtils.get_rate(background_rotation_angle))
        if self.rotation_enabled:
            # Rotate the image. The function returns a tuple; we use the rotated image.
            _, img = ImageUtils.image_rotation(img, angle)
        if not label_apply:
            # Apply color jitter if not processing label/mask.
            img = ImageUtils.apply_color_jitter(
                img,
                brightness_range=self.brightness_range,
                contrast_range=self.contrast_range,
                saturation_range=self.saturation_range,
                hue_range=self.hue_range,
                jitter_probability=self.jitter_probability
            )
            # Add blur to the image.
            img = ImageUtils.add_blur(
                img.astype(np.float32),
                self.blur_kernel_range,
                self.blur_intensity_range,
                self.blur_probability
            ).astype(np.uint8)
            # Add noise to the image.
            img = ImageUtils.add_noise(img, self.noise_level_range)
        return img


###############################################################################
# ObjectAdder
###############################################################################
class ObjectAdder:
    """
    Overlays object images onto a background image.

    It randomly selects a number of objects, determines their placement,
    and then overlays them onto the background image. The output can be used
    for classification or segmentation.
    """

    def __init__(self, obj_images, objects_per_image_range, foreground_rotation_angle, output_format):
        self.obj_images = obj_images
        self.objects_per_image_range = objects_per_image_range
        self.foreground_rotation_angle = foreground_rotation_angle
        self.output_format = output_format

    def overlay_objects(self, background_img):
        """
        Overlays randomly selected objects onto the background image.

        :param background_img: Background image as a NumPy array.
        :return: A tuple (edited_img, final_mask) where:
                 - edited_img is the background with objects overlaid.
                 - final_mask is either a segmentation mask or a classification label.
                 Shape comments:
                     - background_img: (360, 640, 3)
        """
        edited_img = background_img.copy()
        final_out = []
        placement_params = []
        num_objects = random.randint(*self.objects_per_image_range)
        sample_objects = random.sample(self.obj_images, num_objects)
        # Generate placement parameters for each selected object.
        for obj in sample_objects:
            placement_params.append(ImageGeneratorUtils.generate_point(edited_img, obj))

        # Sort placement parameters by the y-coordinate.
        sorted_params = sorted(placement_params, key=lambda item: item[1])
        label = None

        # Overlay objects on the background image.
        for params in sorted_params:
            edited_img, out = ImageGeneratorUtils.generate_img(edited_img, params, self.foreground_rotation_angle)
            if self.output_format == "classification":
                label = out
            elif self.output_format == "segmentation":
                final_out.append(out)

        if self.output_format == "segmentation":
            # Combine individual masks by taking the pixel-wise maximum.
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
        self.bg_transformer = BackgroundTransformer(
            background_rotation,
            # blur_kernel_range,
            # blur_intensity_range,
            # blur_probability,
            noise_level_range,
            flip_probability,
            jitter_probability,
            brightness_range,
            contrast_range,
            saturation_range,
            hue_range
        )
        self.save_to_disk = test_safe_to_desk
        self.name_amount = 0

        # Setup directories if saving to disk.
        if self.save_to_disk:
            if not os.path.isdir(result_dir):
                os.makedirs(result_dir)
            if not os.path.isdir(background_dir):
                raise Exception("Wrong BACKGROUND_DIR")
            if not os.path.isdir(objects_dir):
                raise Exception("Wrong OBJECTS_DIR")
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

        # List of background image filenames.
        self.bg_names = os.listdir(background_dir)
        # List of object images.
        self.obj_images = FileManager.get_objects()
        # Total number of combinations.
        self.total_images = FileManager.get_amount(self.obj_images, self.bg_names)

        self.object_adder = ObjectAdder(
            self.obj_images,
            objects_per_image_range,
            foreground_rotation_angle,
            output_format
        )

    def __iter__(self):
        """
        Iterates over all background images and generates synthetic data.
        For each background, iterates over a set number of object placements.
        """
        for bg_name in self.bg_names:
            bg_path = os.path.join(background_dir, bg_name)
            bg = cv2.imread(bg_path)

            curr_data_path = self.data_path if self.save_to_disk else None
            curr_out_path = self.out_path if self.save_to_disk else None
            if package_by_background and self.save_to_disk:
                subfolder = bg_name.replace(".", "__")
                curr_data_path = os.path.join(self.data_path, subfolder) + os.sep
                curr_out_path = os.path.join(self.out_path, subfolder) + os.sep
                os.makedirs(curr_data_path, exist_ok=True)
                if not merge_outputs:
                    os.makedirs(curr_out_path, exist_ok=True)

            # Iterate over object images and number of combinations.
            for _ in range(len(self.obj_images)):
                for _ in range(images_per_combination):
                    composed_img, mask_or_label = self.object_adder.overlay_objects(bg)

                    # Save the numpy random state to synchronize transformations.
                    np_random_state = np.random.get_state()
                    transformed_img = self.bg_transformer.apply_transformations(composed_img)
                    np.random.set_state(np_random_state)
                    # transformed_mask: min=0, max=255; shape comment: (360, 640, 3)
                    transformed_mask = self.bg_transformer.apply_transformations(mask_or_label, label_apply=True)

                    # Convert the first channel of the mask to a torch tensor.
                    # grayscale_tensor: min=0, max=255; shape: (1, H, W) where H=360, W=640
                    grayscale_tensor = torch.from_numpy(transformed_mask[:, :, 0]).unsqueeze(0)

                    # Initialize distance calculation array.
                    dist_calcul = np.zeros(shape=grayscale_tensor.squeeze().shape)
                    if synthetic_mask_pre_calculation_mode == 'dataloader':
                        # For dataloader mode, compute distance map on CPU.
                        # grayscale_tensor values (0,255) are scaled to (0,1) for binarization.
                        dist_calcul = DistanceCalculator(grayscale_tensor / 255).compute_distance_matrix_on_cpu()
                        # Scale the normalized distance map to 0-255.
                        dist_calcul = dist_calcul.squeeze() * 255.0
                        # Convert to unit8 matrix.
                        dist_calcul = dist_calcul.numpy().astype(np.uint8)

                    # If saving to disk and in segmentation mode, save images and masks.
                    if self.save_to_disk and output_format == "segmentation":
                        FileManager.save_as_grayscale_img(
                            transformed_mask,
                            os.path.join(curr_out_path, f"{self.name_amount}.png")
                        )
                        if synthetic_mask_pre_calculation_mode == 'dataloader':
                            FileManager.save_as_grayscale_img(
                                dist_calcul,
                                os.path.join(self.mask_path, f"{self.name_amount}.png")
                            )

                    if self.save_to_disk:
                        # Save the transformed image (in BGR format) to disk.
                        cv2.imwrite(os.path.join(curr_data_path, f"{self.name_amount}.jpg"), transformed_img)
                    if output_format == "segmentation":
                        # Yield tuple: transformed_img (BGR), transformed_mask (0-255), dist_calcul (0-255)
                        yield transformed_img, transformed_mask, dist_calcul
                    else:
                        yield transformed_img, mask_or_label

                    self.name_amount += 1


###############################################################################
# Data Generator with Progress Bar
###############################################################################
def generate_data_with_progress(generator):
    """
    Wraps a data generator with a progress bar.

    :param generator: A data generator instance.
    :yield: Data items from the generator.
    """
    progress_bar = ProgressBar(max_value=generator.total_images)
    for idx, data in enumerate(generator):
        progress_bar.update(idx)
        yield data


###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    data_generator = DataGenerator(test_safe_to_desk=test_safe_to_desk)
    # Expected shapes:
    # target -> (360, 640, 3)
    # dist_calcul -> (360, 640)
    for input_img, target, dist_mask in generate_data_with_progress(data_generator):
        print("Input shape:", input_img.shape, "Target shape:", target.shape, "Distance mask:", dist_mask.shape)