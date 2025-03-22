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

    def __init__(self,resize_width,resize_height, rotation_enabled,
                 noise_level_range, flip_probability, jitter_probability, brightness_range,
                 contrast_range, saturation_range, hue_range, bg_blur_chance=None,bg_blur_type=None,bg_blur_kernel_range=None):
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.rotation_enabled = rotation_enabled
        self.noise_level_range = noise_level_range
        self.flip_probability = flip_probability
        self.jitter_probability = jitter_probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.bg_blur_chance = bg_blur_chance
        self.bg_blur_type = bg_blur_type
        self.bg_blur_kernel_range = bg_blur_kernel_range

    def apply_transformations(self, img, label, mask = None):
        """
        Applies the same transformations to both the image and the mask.

        :param img: Input image (as a NumPy array).
        :param mask: Corresponding mask (as a NumPy array).
        :param label_apply: Boolean flag. If True, label-dependent transformations are applied.
        :return: Transformed image and mask.
        """
        if np.random.rand() < self.flip_probability:
            img = np.fliplr(img)
            label = np.fliplr(label)
            if mask is not None:
                mask = np.fliplr(mask)

        angle = int(ImageUtils.get_rate(background_rotation_angle))
        if self.rotation_enabled:
            _, img = ImageUtils.image_rotation(img, angle)
            _, label = ImageUtils.image_rotation(label, angle)
            if mask is not None:
                _, mask = ImageUtils.image_rotation(mask, angle)

        img = ImageUtils.apply_color_jitter(
                img,
                brightness_range=self.brightness_range,
                contrast_range=self.contrast_range,
                saturation_range=self.saturation_range,
                hue_range=self.hue_range,
                jitter_probability=self.jitter_probability
            )
        img = ImageUtils.add_noise(img, self.noise_level_range)

        if (self.bg_blur_chance is not None and
            self.bg_blur_type is not None and
            self.bg_blur_kernel_range is not None):
            img = ImageUtils.add_background_blur(img, label, self.bg_blur_chance, self.bg_blur_type,self.bg_blur_kernel_range)

        img = cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
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
        sorted_params = sorted(placement_params, key=lambda item: item[1] + item[2].shape[0])
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
            resize_width,
            resize_height,
            background_rotation,
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

        self.bg_names = os.listdir(background_dir)
        self.obj_images = FileManager.get_objects()

        self.object_adder = ObjectAdder(
            self.obj_images,
            objects_per_image_range,
            foreground_rotation_angle,
            output_format
        )

    def __iter__(self):
        while True:
            bg_name = random.choice(self.bg_names)
            bg_path = os.path.join(background_dir, bg_name)
            bg = cv2.imread(str(bg_path))

            curr_data_path = self.data_path if self.save_to_disk else None
            curr_out_path = self.out_path if self.save_to_disk else None
            if package_by_background and self.save_to_disk:
                subfolder = bg_name.replace(".", "__")
                curr_data_path = os.path.join(self.data_path, subfolder) + os.sep
                curr_out_path = os.path.join(self.out_path, subfolder) + os.sep
                os.makedirs(curr_data_path, exist_ok=True)
                if not merge_outputs:
                    os.makedirs(curr_out_path, exist_ok=True)


            composed_img, mask_or_label = self.object_adder.overlay_objects(bg)

            transformed_img, transformed_mask, _ = self.bg_transformer.apply_transformations(composed_img,
                                                                                             mask_or_label)

            if output_format == "segmentation":
                transformed_mask = (transformed_mask > 0).astype(np.uint8) * 255
                grayscale_tensor = torch.from_numpy(transformed_mask[:, :, 0]).unsqueeze(0)
                dist_calcul = np.zeros(shape=grayscale_tensor.squeeze().shape)
                if synthetic_mask_pre_calculation_mode == 'dataloader':
                    dist_calcul = DistanceCalculator(grayscale_tensor / 255).compute_distance_matrix_on_cpu()
                    dist_calcul = dist_calcul.squeeze() * 255.0
                    dist_calcul = dist_calcul.numpy().astype(np.float32)

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

                yield transformed_img, transformed_mask, dist_calcul

            else:
                yield transformed_img, mask_or_label

            self.name_amount += 1


###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    total_iterations = 10000

    data_generator = DataGenerator(test_safe_to_desk=test_safe_to_desk)

    progress_bar = ProgressBar(max_value=total_iterations)

    for iteration, (input_img, target, dist_mask) in enumerate(data_generator):

        progress_bar.update(iteration + 1)

        if iteration + 1 >= total_iterations:
            break

