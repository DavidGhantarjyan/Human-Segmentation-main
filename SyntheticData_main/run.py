import cv2
import random
from progressbar import ProgressBar
from other.parsing.train_args_parser import *
from other.losses_utils import DistanceCalculator
import torch
from SyntheticData_main.utils import FileManager, ImageUtils, ImageGeneratorUtils
import albumentations as A


class CoarseDropoutWithROI(A.CoarseDropout):
    def __init__(self, dropout_roi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_roi = dropout_roi
        if self.dropout_roi is not None:
            self.x_min = dropout_roi[0]
            self.y_min = dropout_roi[1]
            self.x_max = dropout_roi[2]
            self.y_max = dropout_roi[3]

    def apply(self, img, holes=(), **params):
        height, width = img.shape[:2]
        if self.dropout_roi is not None:
            valid_holes = []
            for x1, y1, x2, y2 in holes:
                if int(self.x_min * width) <= x1 and x2 <= int(self.x_max * width) and int(
                        self.y_min * height) <= y1 and y2 <= int(self.y_max * height):
                    valid_holes.append((x1, y1, x2, y2))
            holes = np.array(valid_holes)
        return super().apply(img, holes=holes, **params)


class RandomCropNearRandomObjBBox(A.RandomCropNearBBox):
    def __init__(self, crop_probability, max_part_shift, *args, **kwargs):
        super().__init__(max_part_shift=max_part_shift, p=crop_probability, *args, **kwargs)

    def __call__(self, **kwargs):
        mask = kwargs.get("mask")[:, :, 0]
        mask_height, mask_width = mask.shape[:2]
        mask_ratio = mask_width / mask_height
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            selected_contour = random.choice(contours)
            x, y, w, h = cv2.boundingRect(selected_contour)

            center_x = x + w / 2.0
            center_y = y + h / 2.0

            current_ratio = w / h
            if current_ratio < mask_ratio:
                new_w = h * mask_ratio
                new_h = h
            else:
                new_w = w
                new_h = w / mask_ratio

            new_w = int(round(new_w))
            new_h = int(round(new_h))

            new_x = int(round(center_x - new_w / 2))
            new_y = int(round(center_y - new_h / 2))

            new_x = max(0, min(new_x, mask_width - new_w))
            new_y = max(0, min(new_y, mask_height - new_h))
            new_w = min(new_w, mask_width - new_x)
            new_h = min(new_h, mask_height - new_y)

            kwargs["cropping_bbox"] = [new_x, new_y, new_x + new_w, new_y + new_h]
        else:
            return kwargs

        return super().__call__(**kwargs)

    def apply(self, image, cropping_bbox=None, **params):
        # cropping_bbox -> max_part_shift ->  crop_coords
        cropping_bbox = list(params["crop_coords"])
        height, width = image.shape[:2]
        cropped_image = super().apply(img=image, cropping_bbox=cropping_bbox, **params)
        cropped_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)
        return cropped_image

    def apply_to_mask(self, mask, cropping_bbox=None, **params):
        cropping_bbox = list(params["crop_coords"])
        height, width = mask.shape[:2]
        cropped_mask = super().apply_to_mask(mask=mask, cropping_bbox=cropping_bbox, **params)
        cropped_mask = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return cropped_mask


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

    # contrast_brightness_probability,brightness_limit,contrast_limit
    def __init__(self, resize_width, resize_height, rotation_enabled,
                 noise_level_range, noise_type, flip_probability, jitter_probability
                 , saturation_range, hue_range, gamma_probability, gamma_limit, sigmoid_probability, sigmoid_limit,
                 contrast_brightness_probability, brightness_range, contrast_range,
                 clahe_probability, tile_grid_size, clip_limit, motion_blur_probability, motion_blur_limit
                 , full_img_shadow_roi, full_img_num_shadows_limit, full_img_shadow_dimension,
                 full_img_shadow_intensity_range, full_img_shadow_transform_probability,
                 full_img_dropout_probability, full_img_dropout_holes_range, full_img_dropout_height_range,
                 full_img_dropout_width_range,
                 full_img_dropout_mask_fill_value, full_img_dropout_roi,
                 crop_img_dropout_probability=None, crop_img_dropout_holes_range=None,
                 crop_img_dropout_height_range=None,
                 crop_img_dropout_width_range=None,
                 crop_img_dropout_mask_fill_value=None, crop_img_dropout_roi=None,
                 cropped_img_shadow_roi=None, cropped_img_num_shadows_limit=None, cropped_img_shadow_dimension=None,
                 cropped_img_shadow_intensity_range=None, cropped_img_shadow_transform_probability=0,
                 crop_probability=None, max_part_shift=None
                 , bg_blur_chance=None, bg_blur_type=None, bg_blur_kernel_range=None):

        # Target dimensions for resizing the image.
        self.resize_width = resize_width
        self.resize_height = resize_height
        # Boolean flag indicating whether rotation is enabled.
        self.rotation_enabled = rotation_enabled
        # Noise level range to apply random noise.
        self.noise_level_range = noise_level_range
        self.noise_type = noise_type
        # Probability to perform horizontal flip.
        self.flip_probability = flip_probability
        # Probability to apply color jitter adjustments.
        self.jitter_probability = jitter_probability

        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gamma_probability = gamma_probability
        self.gamma_limit = gamma_limit

        self.sigmoid_probability = sigmoid_probability
        self.sigmoid_limit = sigmoid_limit

        self.contrast_brightness_probability = contrast_brightness_probability
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        self.clahe_probability = clahe_probability
        self.tile_grid_size = tile_grid_size
        self.clip_limit = clip_limit

        self.motion_blur_probability = motion_blur_probability
        self.motion_blur_limit = motion_blur_limit

        self.full_img_dropout_probability = full_img_dropout_probability
        self.full_img_dropout_holes_range = full_img_dropout_holes_range
        self.full_img_dropout_height_range = full_img_dropout_height_range
        self.full_img_dropout_width_range = full_img_dropout_width_range
        self.full_img_dropout_mask_fill_value = full_img_dropout_mask_fill_value
        self.full_img_dropout_roi = full_img_dropout_roi

        self.crop_img_dropout_probability = crop_img_dropout_probability
        self.crop_img_dropout_holes_range = crop_img_dropout_holes_range
        self.crop_img_dropout_height_range = crop_img_dropout_height_range
        self.crop_img_dropout_width_range = crop_img_dropout_width_range
        self.crop_img_dropout_mask_fill_value = crop_img_dropout_mask_fill_value
        self.crop_img_dropout_roi = crop_img_dropout_roi

        self.full_img_shadow_roi = full_img_shadow_roi
        self.full_img_num_shadows_limit = full_img_num_shadows_limit
        self.full_img_shadow_dimension = full_img_shadow_dimension
        self.full_img_shadow_intensity_range = full_img_shadow_intensity_range
        self.full_img_shadow_transform_probability = full_img_shadow_transform_probability

        self.cropped_img_shadow_roi = cropped_img_shadow_roi
        self.cropped_img_num_shadows_limit = cropped_img_num_shadows_limit
        self.cropped_img_shadow_dimension = cropped_img_shadow_dimension
        self.cropped_img_shadow_intensity_range = cropped_img_shadow_intensity_range
        self.cropped_img_shadow_transform_probability = cropped_img_shadow_transform_probability

        self.crop_probability = crop_probability
        self.max_part_shift = max_part_shift

        # Optional background blur parameters.
        self.bg_blur_chance = bg_blur_chance
        self.bg_blur_type = bg_blur_type
        self.bg_blur_kernel_range = bg_blur_kernel_range

        if (crop_probability is not None and max_part_shift is not None):
            self.crop_transform = A.Compose([
                RandomCropNearRandomObjBBox(crop_probability=self.crop_probability, max_part_shift=self.max_part_shift)
            ], additional_targets={'mask': 'mask'})

        self.shadow_transform_full = A.Compose([
            A.RandomShadow(
                shadow_roi=self.full_img_shadow_roi,
                num_shadows_limit=self.full_img_num_shadows_limit,
                shadow_dimension=self.full_img_shadow_dimension,
                shadow_intensity_range=self.full_img_shadow_intensity_range,
                p=self.full_img_shadow_transform_probability,
            )
        ])
        if (cropped_img_shadow_roi is not None and
                cropped_img_num_shadows_limit is not None and
                cropped_img_shadow_dimension is not None and
                cropped_img_shadow_intensity_range is not None and
                cropped_img_shadow_transform_probability is not None):
            self.shadow_transform_crop = A.Compose([
                A.RandomShadow(
                    shadow_roi=self.cropped_img_shadow_roi,
                    num_shadows_limit=self.cropped_img_num_shadows_limit,
                    shadow_dimension=self.cropped_img_shadow_dimension,
                    shadow_intensity_range=self.cropped_img_shadow_intensity_range,
                    p=self.cropped_img_shadow_transform_probability,
                )
            ])

    def apply_sigmoid_contrast(self, image, **kwargs):
        k = ImageUtils.get_rate(self.sigmoid_limit)
        return ImageUtils.sigmoid_contrast(image, k=k)

    def apply_color_jitter_wrapper(self, image, **kwargs):
        return ImageUtils.apply_color_jitter(
            image,
            contrast_brightness_probability=self.contrast_brightness_probability,
            brightness_range=self.brightness_range,
            contrast_range=self.contrast_range,
            saturation_range=self.saturation_range,
            hue_range=self.hue_range,
            jitter_probability=self.jitter_probability,
        )

    def build_transform(self, was_cropped: bool, was_green_screen: bool = False):
        if was_cropped or was_green_screen:
            dropout_transform = CoarseDropoutWithROI(
                dropout_roi=self.crop_img_dropout_roi,
                num_holes_range=self.crop_img_dropout_holes_range,
                hole_height_range=self.crop_img_dropout_height_range,
                hole_width_range=self.crop_img_dropout_width_range,
                fill='random_uniform',
                fill_mask=self.crop_img_dropout_mask_fill_value,
                p=self.crop_img_dropout_probability
            )
        else:
            dropout_transform = CoarseDropoutWithROI(
                dropout_roi=self.full_img_dropout_roi,
                num_holes_range=self.full_img_dropout_holes_range,
                hole_height_range=self.full_img_dropout_height_range,
                hole_width_range=self.full_img_dropout_width_range,
                fill='random_uniform',
                fill_mask=self.full_img_dropout_mask_fill_value,
                p=self.full_img_dropout_probability
            )

        return A.Compose([
            A.MotionBlur(blur_limit=self.motion_blur_limit, p=self.motion_blur_probability),
            dropout_transform,
            A.RandomGamma(gamma_limit=self.gamma_limit, p=self.gamma_probability),
            A.CLAHE(clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size, p=self.clahe_probability),
            A.Lambda(image=self.apply_sigmoid_contrast, p=self.sigmoid_probability),
            A.Lambda(image=self.apply_color_jitter_wrapper, p=self.jitter_probability),
        ], additional_targets={'mask': 'mask'})

    #  was_tiktok=False
    def apply_transformations(self, img, label, was_green_screen=False):
        """
        Applies the same set of transformations to the input image and label (and optionally mask).

        :param img: Input background image as a NumPy array.
        :param label: Label or segmentation map corresponding to the image.
        :return: Tuple (img, label) after applying all transformations.
        """
        # Random horizontal flip with given probability.
        if np.random.rand() < self.flip_probability:
            img = np.fliplr(img)
            label = np.fliplr(label)

        # >>>>
        was_cropped = False
        # if not was_green_screen:
        if self.crop_probability is not None and self.max_part_shift is not None:
            img_before_crop = img.copy()
            augmented_crop = self.crop_transform(image=img, mask=label)
            img = augmented_crop['image']
            label = augmented_crop['mask']
            was_cropped = not np.array_equal(img_before_crop, img)

        if was_cropped or was_green_screen:
            augmented_shadow = self.shadow_transform_crop(image=img)
        else:
            augmented_shadow = self.shadow_transform_full(image=img)
        img = augmented_shadow['image']

        angle = int(ImageUtils.get_rate(synthetic_background_rotation_angle))
        if self.rotation_enabled:
            # Rotate image with linear interpolation.
            _, img = ImageUtils.image_rotation(img, angle, borderMode=cv2.BORDER_REFLECT,
                                               interpolation_type=cv2.INTER_LINEAR)
            # Rotate label using nearest-neighbor to preserve discrete values.
            _, label = ImageUtils.image_rotation(label, angle, borderMode=cv2.BORDER_REFLECT,
                                                 interpolation_type=cv2.INTER_NEAREST)

        transform = self.build_transform(was_cropped=was_cropped, was_green_screen=was_green_screen)
        augmented = transform(image=img, mask=label)
        img = augmented['image']
        label = augmented['mask']

        # Add random noise to the image.
        img = ImageUtils.add_noise(img, self.noise_level_range, self.noise_type)

        # Optionally apply background blur if all blur parameters are set.
        if (self.bg_blur_chance is not None and
                self.bg_blur_type is not None and
                self.bg_blur_kernel_range is not None):
            blur_chance = self.bg_blur_chance

            if np.random.rand() < blur_chance:
                img = ImageUtils.add_background_blur(img, label, self.bg_blur_chance, self.bg_blur_type,
                                                     self.bg_blur_kernel_range)


        target_width = 2 * self.resize_width
        target_height = 2 * self.resize_height
        # current@ stanum enq (640,360) kam (640,h), (w,360)
        current_height, current_width = img.shape[:2]
        current_ratio = current_height / current_width
        target_ratio = target_height / target_width

        #после преобразования == может быть только при 640*360
        if current_ratio != target_ratio:
            pad_left = (target_width - current_width) // 2
            pad_right = target_width - current_width - pad_left

            pad_top = (target_height - current_height) // 2
            pad_bottom = target_height - current_height - pad_top

            img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])

            label = cv2.copyMakeBorder(label, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize image, label, and mask to target dimensions.
        img = cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.resize_width, self.resize_height), interpolation=cv2.INTER_NEAREST)

        return img, label


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

    def __init__(self, obj_paths, objects_per_image_range, foreground_rotation_angle, output_format):
        # List of available object images.
        self.obj_paths = obj_paths
        # Tuple defining the minimum and maximum number of objects to place per image.
        self.objects_per_image_range = objects_per_image_range
        # Range for random rotation to be applied to foreground objects.
        self.foreground_rotation_angle = foreground_rotation_angle
        # Output type: either "classification" or "segmentation".
        self.output_format = output_format

    # def overlay_green_screen_object(self, background_img, green_screen_img):

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

        # # Determine random number of objects to overlay.
        # num_objects = random.randint(*self.objects_per_image_range)
        # # Randomly sample objects from the available list.
        # sample_objects = random.sample(self.obj_paths, num_objects)
        # # Generate placement parameters for each selected object.
        # for obj in sample_objects:
        #     placement_params.append(ImageGeneratorUtils.generate_point(edited_img, obj))

        num_objects = random.randint(*self.objects_per_image_range)
        selected_paths = random.sample(self.obj_paths, num_objects)
        for path in selected_paths:
            obj = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if obj is None:
                continue  # Skip if the image couldn't be read
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
            resize_width=synthetic_resize_width,
            resize_height=synthetic_resize_height,
            rotation_enabled=synthetic_background_rotation,  # Boolean flag for rotation
            noise_level_range=synthetic_noise_level_range,
            noise_type=synthetic_noise_type,
            flip_probability=synthetic_flip_probability,

            jitter_probability=synthetic_jitter_probability,

            saturation_range=synthetic_saturation_range,
            hue_range=synthetic_hue_range,
            gamma_probability=synthetic_gamma_probability,
            gamma_limit=synthetic_gamma_limit,

            sigmoid_probability=synthetic_sigmoid_probability,
            sigmoid_limit=synthetic_sigmoid_limit,

            contrast_brightness_probability=synthetic_contrast_brightness_probability,
            brightness_range=synthetic_brightness_range,
            contrast_range=synthetic_contrast_range,

            clahe_probability=synthetic_clahe_probability,
            tile_grid_size=synthetic_tile_grid_size,
            clip_limit=synthetic_clip_limit,
            motion_blur_probability=synthetic_motion_blur_probability,
            motion_blur_limit=synthetic_motion_blur_limit,

            full_img_dropout_probability=synthetic_full_img_dropout_probability,
            full_img_dropout_holes_range=synthetic_full_img_dropout_holes_range,
            full_img_dropout_height_range=synthetic_full_img_dropout_height_range,
            full_img_dropout_width_range=synthetic_full_img_dropout_width_range,
            full_img_dropout_mask_fill_value=synthetic_full_img_dropout_mask_fill_value,
            full_img_dropout_roi=synthetic_full_img_dropout_roi,

            crop_img_dropout_probability=synthetic_crop_img_dropout_probability,
            crop_img_dropout_holes_range=synthetic_crop_img_dropout_holes_range,
            crop_img_dropout_height_range=synthetic_crop_img_dropout_height_range,
            crop_img_dropout_width_range=synthetic_crop_img_dropout_width_range,
            crop_img_dropout_mask_fill_value=synthetic_crop_img_dropout_mask_fill_value,
            crop_img_dropout_roi=synthetic_crop_img_dropout_roi,

            full_img_shadow_roi=synthetic_full_img_shadow_roi,
            full_img_num_shadows_limit=synthetic_full_img_num_shadows_limit,
            full_img_shadow_dimension=synthetic_full_shadow_dimension,
            full_img_shadow_intensity_range=synthetic_full_shadow_intensity_range,
            full_img_shadow_transform_probability=synthetic_full_shadow_transform_probability,

            cropped_img_shadow_roi=synthetic_cropped_img_shadow_roi,
            cropped_img_num_shadows_limit=synthetic_cropped_img_num_shadows_limit,
            cropped_img_shadow_dimension=synthetic_cropped_img_shadow_dimension,
            cropped_img_shadow_intensity_range=synthetic_cropped_img_shadow_intensity_range,
            cropped_img_shadow_transform_probability=synthetic_cropped_img_shadow_transform_probability,

            crop_probability=synthetic_crop_probability,
            max_part_shift=synthetic_max_part_shift,

            bg_blur_chance=synthetic_bg_blur_probability,
            bg_blur_type=synthetic_bg_blur_type,
            bg_blur_kernel_range=synthetic_bg_blur_kernel_range,
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

        self.tiktok_bg_dir = tiktok_background_dir
        self.tiktok_obj_dir = tiktok_obj_dir

        # Load background image names from the designated directory.
        self.bg_names = os.listdir(background_dir)
        # Load object images using a file manager utility.
        self.obj_paths = FileManager.get_object_paths(objects_dir)

        self.greenscreen_obj_dir = greenscreen_obj_dir

        # Initialize the object adder with object images and transformation parameters.
        self.object_adder = ObjectAdder(
            self.obj_paths,
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
            # Set current output paths based on whether saving to disk is enabled.
            curr_data_path = self.data_path if self.save_to_disk else None
            curr_out_path = self.out_path if self.save_to_disk else None

            # 70% probability
            if tiktok_application_chance < np.random.random():
                bg_name = random.choice(self.bg_names)
                bg_path = os.path.join(background_dir, bg_name)
                bg = cv2.imread(str(bg_path))

                # 40% —» 28%
                if greenscreen_application_chance > np.random.random():
                    composed_img_path, mask_or_label_path = FileManager.get_random_fgr_and_mask(
                        self.greenscreen_obj_dir)
                    composed_img = cv2.imread(str(composed_img_path))
                    mask_or_label = cv2.imread(str(mask_or_label_path))

                    denoised_mask = cv2.bilateralFilter(mask_or_label, d=9, sigmaColor=75, sigmaSpace=75)

                    # denoised_mask = cv2.fastNlMeansDenoising(mask_or_label, h=10, templateWindowSize=7, searchWindowSize=21)
                    resized_green = cv2.resize(composed_img, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)
                    resized_alpha = cv2.resize(denoised_mask, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_AREA)

                    alpha_normalized = resized_alpha.astype(float) / 255.0
                    blured_alpha_normalized = (ImageUtils.add_alpha_blur(alpha_normalized, alpha_blur_type,
                                                                         video_matte_alpha_blur_kernel_range)) * alpha_normalized
                    resized_green = resized_green.astype(float)
                    composite = resized_green * blured_alpha_normalized + bg.astype(float) * (
                                1 - blured_alpha_normalized)
                    composed_img = np.clip(composite, 0, 255).astype(np.uint8)
                    mask_or_label = cv2.resize((denoised_mask > 80).astype(np.uint8), (bg.shape[1], bg.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                    transformed_img, transformed_mask = self.bg_transformer.apply_transformations(composed_img,
                                                                                                  mask_or_label,
                                                                                                  was_green_screen=True)
                # 60% -» 42%
                else:
                    composed_img, mask_or_label = self.object_adder.overlay_objects(bg)
                    mask_or_label = (mask_or_label > 0).astype(np.uint8)
                    transformed_img, transformed_mask = self.bg_transformer.apply_transformations(composed_img,
                                                                                                  mask_or_label)
            # 30%
            else:
                bg_name, bg_path = FileManager.get_random_background(self.tiktok_bg_dir)
                background = cv2.imread(str(bg_path))
                bg_height, bg_width = background.shape[:2]
                target_w, target_h = synthetic_resize_width * 2, synthetic_resize_height * 2
                aspect_ratio = bg_width / bg_height
                target_aspect = target_w / target_h

                if aspect_ratio > target_aspect:
                    scale = target_w / bg_width
                    new_bg_width, new_bg_height = target_w, int(bg_height * scale)
                elif aspect_ratio < target_aspect:
                    scale = target_h / bg_height
                    new_bg_height, new_bg_width = target_h, int(bg_width * scale)
                else:
                    new_bg_height, new_bg_width = target_h, target_w

                background = cv2.resize(background, (new_bg_width, new_bg_height), interpolation=cv2.INTER_AREA)

                composed_img_path, mask_or_label_path = FileManager.get_random_tiktok_image_and_mask(
                    self.tiktok_obj_dir)
                composed_img = cv2.imread(str(composed_img_path))
                mask_or_label = cv2.imread(str(mask_or_label_path))
                obj_height, obj_width = composed_img.shape[:2]
                target_w, target_h = synthetic_resize_width * 2, synthetic_resize_height * 2
                aspect_ratio = obj_width / obj_height
                target_aspect = target_w / target_h

                if aspect_ratio > target_aspect:
                    scale = target_w / obj_width
                    new_obj_width, new_obj_height = target_w, int(obj_height * scale)
                elif aspect_ratio < target_aspect:
                    scale = target_h / obj_height
                    new_obj_height, new_obj_width = target_h, int(obj_width * scale)
                else:
                    new_obj_height, new_obj_width = target_h, target_w

                resized_green = cv2.resize(composed_img, (new_obj_width, new_obj_height), interpolation=cv2.INTER_AREA)
                resized_alpha = cv2.resize(mask_or_label, (new_obj_width, new_obj_height),
                                           interpolation=cv2.INTER_NEAREST)
                if new_bg_width > new_obj_width:
                    max_start_x = new_bg_width - new_obj_width
                    start_x = random.randint(0, max_start_x)
                    background = background[:, start_x:start_x + new_obj_width]
                else:
                    max_start_x = new_obj_width - new_bg_width
                    start_x = random.randint(0, max_start_x)
                    resized_green = resized_green[:, start_x:start_x + new_bg_width]
                    resized_alpha = resized_alpha[:, start_x:start_x + new_bg_width]

                resized_normlized_alpha = resized_alpha.astype(float) / 255.0
                blured_obj_normalized = (ImageUtils.add_alpha_blur(resized_normlized_alpha, alpha_blur_type,
                                                                   tiktok_alpha_blur_kernel_range)) * resized_normlized_alpha
                resized_green = resized_green.astype(float)
                composite = resized_green * blured_obj_normalized + background.astype(float) * (
                            1 - blured_obj_normalized)
                composed_img = np.clip(composite, 0, 255).astype(np.uint8)

                transformed_img, transformed_mask = self.bg_transformer.apply_transformations(composed_img,
                                                                                              resized_normlized_alpha.astype(
                                                                                                  np.uint8),
                                                                                              was_green_screen=True)
            if package_by_background and self.save_to_disk:
                # Create a subfolder for outputs based on the background image name.
                subfolder = bg_name.replace(".", "__")
                curr_data_path = os.path.join(self.data_path, subfolder) + os.sep
                curr_out_path = os.path.join(self.out_path, subfolder) + os.sep
                os.makedirs(curr_data_path, exist_ok=True)
                if not merge_outputs:
                    os.makedirs(curr_out_path, exist_ok=True)

            if output_format == "segmentation":
                # transformed_mask = (transformed_mask > 0).astype(np.uint8) * 255
                transformed_mask *= 255
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
