import cv2
from other.parsing.train_args_parser import *
import random


# Import configuration parameters from train_args_parser, e.g. images_per_combination, result_dir, etc.
def list_subfolders(path):
    """
        Lists subdirectories in the given path.

        Args:
            path (str): Directory path.

        Returns:
            list: List of subdirectory names.

        Raises:
            ValueError: If no subdirectories are found.
        """
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if not folders:
        raise ValueError(f"No subfolders in {path}")
    return folders

def list_files(path, exts):
    """
        Lists files with specified extensions in the given path.

        Args:
            path (str): Directory path.
            exts (tuple): File extensions to filter (e.g., ('.png', '.jpg')).

        Returns:
            list: List of filenames.

        Raises:
            ValueError: If no files with specified extensions are found.
        """
    files = [f for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f))
             and f.lower().endswith(exts)]
    if not files:
        raise ValueError(f"No files with extensions {exts} in {path}")
    return files

def pick_random_path(path, exts):
    """
        Selects a random file with specified extensions from the given path.

        Args:
            path (str): Directory path.
            exts (tuple): File extensions to filter.

        Returns:
            str: Absolute path to a random file.
        """
    files = list_files(path, exts)
    return os.path.join(path, random.choice(files))

###############################################################################
# FileManager Class
###############################################################################
class FileManager:
    """
        Manages file operations for synthetic data generation, including directory creation,
        file listing, and image/text saving.
        """
    @staticmethod
    def get_amount(obj, bg):
        """
        Calculates the total number of images generated based on object and background counts.

        Args:
            obj (list): List of object image paths.
            bg (list): List of background image paths.

        Returns:
            int: Total number of generated images (images_per_combination * len(obj) * len(bg)).
        """
        # Multiply images_per_combination (a global parameter) by the number of objects and backgrounds.

        return images_per_combination * len(obj) * len(bg)

    @staticmethod
    def get_new_epoch_path():
        """
        Creates a unique epoch directory for saving generated results.

        Returns:
            str: Path to a new epoch directory (e.g., 'result_dir/epoch0/').
        """
        m = 0
        p = os.listdir(result_dir)
        # Loop until an unused epoch folder name is found.

        while True:
            ep_dir = f"epoch{m}"
            if ep_dir in p:
                m += 1
            else:
                break
        # Construct full path and ensure a trailing separator.

        ep_dir = os.path.join(result_dir, ep_dir) + os.sep
        return ep_dir

    @staticmethod
    def get_object_paths(objects_dir):
        """
        Retrieves paths to all object images in the specified directory.

        Args:
            objects_dir (str): Directory containing object images.

        Returns:
            list: List of absolute paths to object images.
        """
        return [os.path.join(objects_dir, name) for name in os.listdir(objects_dir)]

    @staticmethod
    def save_as_txt(text, path):
        """
        Appends text to a file at the specified path.

        Args:
            text (str): Text to save.
            path (str): File path (without .txt extension).
        """
        with open(path + ".txt", 'a') as f:
            f.write(text + '\n')
    @staticmethod
    def get_random_background(base_dir):
        """
        Selects a random background image from a subdirectory.

        Args:
            base_dir (str): Base directory with background subfolders.

        Returns:
            tuple: (filename, full_path) of the selected background image.
        """
        folder = random.choice(list_subfolders(base_dir))
        folder_path = os.path.join(base_dir, folder)

        filename = random.choice(list_files(folder_path, ('.png', '.jpg', '.jpeg')))
        full_path = os.path.join(folder_path, filename)

        return filename, full_path

    @staticmethod
    def get_random_tiktok_image_and_mask(base_dir):
        """
        Selects a random TikTok-style image and its corresponding mask.

        Args:
            base_dir (str): Base directory with subfolders containing 'images' and 'masks'.

        Returns:
            tuple: (image_path, mask_path) of the selected image and mask.

        Raises:
            ValueError: If 'images' or 'masks' subdirectories are missing.
            FileNotFoundError: If no mask is found for the selected image.
        """
        folder = random.choice(list_subfolders(base_dir))
        folder_path = os.path.join(base_dir, folder)

        images_path = os.path.join(folder_path, "images")
        masks_path  = os.path.join(folder_path, "masks")

        if not os.path.exists(images_path) or not os.path.exists(masks_path):
            raise ValueError(f"Folder '{folder}' must contain 'images' and 'masks' subdirectories")

        image_file = random.choice(list_files(images_path, ('.png', '.jpg', '.jpeg')))
        image_path = os.path.join(images_path, image_file)

        mask_name = os.path.splitext(image_file)[0] + ".png"
        mask_path = os.path.join(masks_path, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"No mask '{mask_name}' found for image '{image_file}'")

        return image_path, mask_path

    @staticmethod
    def get_random_fgr_and_mask(base_dir):
        """
        Selects a random foreground image and its alpha mask from subdirectories.

        Args:
            base_dir (str): Base directory with 'fgr' and 'pha' subdirectories.

        Returns:
            tuple: (fgr_path, pha_path) of the selected foreground and alpha mask.

        Raises:
            FileNotFoundError: If no alpha mask is found for the foreground.
        """
        fgr_dir = os.path.join(base_dir, "fgr")
        pha_dir = os.path.join(base_dir, "pha")

        subdirs = sorted(list_subfolders(fgr_dir))
        chosen_subdir = random.choice(subdirs)

        fgr_subdir_path = os.path.join(fgr_dir, chosen_subdir)
        pha_subdir_path = os.path.join(pha_dir, chosen_subdir)

        filename = random.choice(list_files(fgr_subdir_path, ('.jpg',)))
        fgr_path = os.path.join(fgr_subdir_path, filename)
        pha_path = os.path.join(pha_subdir_path, os.path.splitext(filename)[0] + ".jpg")

        if not os.path.exists(pha_path):
            raise FileNotFoundError(f"No alpha mask '{pha_path}' found for foreground '{fgr_path}'")

        return fgr_path, pha_path

    @staticmethod
    def save_as_grayscale_img(img, path):
        """
        Saves a grayscale image to disk, ensuring uint8 format.

        Args:
            img (np.ndarray): Image array (single-channel or 3-channel).
            path (str): Destination file path.

        Raises:
            IOError: If the image fails to save.
            ValueError: If the image has invalid dimensions.
        """
        # If image has 3 channels, select the first channel to convert it to grayscale
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, 0]
        # Ensure image is uint8.
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        success = cv2.imwrite(path, img)
        if not success:
            raise IOError(f"Failed to save image to {path}")


###############################################################################
# ImageUtils Class
###############################################################################
class ImageUtils:
    """
    Provides utility functions for image processing, including scaling, flipping,
    blurring, noise addition, rotation, and color adjustments.
    """
    @staticmethod
    def get_rate(rate):
        """
        Generates a random value based on the input rate.

        Args:
            rate (tuple, list, or float): If tuple, random float in range; if list, random choice; else, returns rate.

        Returns:
            float: Random or fixed rate value.
        """
        if isinstance(rate, tuple):
            return np.random.rand() * (rate[1] - rate[0]) + rate[0]
        elif isinstance(rate, list):
            return np.random.choice(rate, 1)
        else:
            return rate

    @staticmethod
    def flip_img(img, flip_chance):
        """
        Flips the image horizontally with a given probability.

        Args:
            img (np.ndarray): Input image.
            flip_chance (float): Probability of flipping (0 to 1).

        Returns:
            np.ndarray: Flipped or original image.
        """
        if np.random.rand() < flip_chance:
            return img
        return cv2.flip(img, 1)

    @staticmethod
    def add_alpha_blur(alpha, alpha_blur_type, alpha_blur_kernel_range):
        """
        Applies blur to the alpha channel.

        Args:
            alpha (np.ndarray): Alpha channel (single-channel, float32 [0, 1]).
            alpha_blur_type (str): Blur type (e.g., 'GaussianBlur').
            alpha_blur_kernel_range (tuple): Min and max kernel sizes.

        Returns:
            np.ndarray: Blurred alpha channel.
        """
        if alpha_blur_type == 'GaussianBlur':
            k_min, k_max = alpha_blur_kernel_range
            # Choose a random odd kernel size in the given range.
            ksize = random.randrange(k_min, k_max + 1, 2)
            blured_alpha = cv2.GaussianBlur(alpha, ksize=(ksize, ksize), sigmaX=0)
            return blured_alpha
        return alpha

    @staticmethod
    def add_background_blur(img, label, bg_blur_chance, bg_blur_type, bg_blur_kernel_range):
        """
        Applies blur to the background where label is not 1.

        Args:
            img (np.ndarray): Background image.
            label (np.ndarray): Segmentation mask (1 for foreground, 0 for background).
            bg_blur_chance (float): Probability of applying blur.
            bg_blur_type (str): Blur type (e.g., 'GaussianBlur').
            bg_blur_kernel_range (tuple): Min and max kernel sizes.

        Returns:
            np.ndarray: Image with blurred background.
        """
        blured_img = img
        if bg_blur_chance:
            if np.random.rand() < (bg_blur_chance):
                return blured_img
        if bg_blur_type == 'GaussianBlur':
            k_min, k_max = bg_blur_kernel_range
            ksize = random.randrange(k_min, k_max + 1, 2)
            bg_blured = cv2.GaussianBlur(blured_img, (ksize, ksize), sigmaX=0)
            # Restore original image pixels where label equals 1.
            bg_blured = np.where(label == 1, blured_img, bg_blured)
            return bg_blured


    @staticmethod
    def add_noise(img, noise_rate, noise_type):
        """
        Adds uniform or Gaussian noise to the image.

        Args:
            img (np.ndarray): Input image.
            noise_rate (float or tuple): Noise intensity or range.
            noise_type (str): Noise type ('uniform' or 'gaussian').

        Returns:
            np.ndarray: Noisy image (uint8).
        """
        noise_rate = ImageUtils.get_rate(noise_rate)
        arg = int(noise_rate * 127)
        if arg == 0:
            return img

        if noise_type == 'uniform':
            noise = np.random.randint(-arg, arg, img.shape)
        elif noise_type == 'gaussian':
            noise = np.random.randn(*img.shape) * arg
            noise = np.clip(noise, -arg, arg)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
        # return np.clip(img, arg, 255 - arg) + noise

    @staticmethod
    def scale_img(img, scale_rate):
        """
        Scales the image by a given factor, preserving alpha channel if present.

        Args:
            img (np.ndarray): Input image (3 or 4 channels).
            scale_rate (float or tuple): Scaling factor or range.

        Returns:
            np.ndarray: Scaled image.

        Raises:
            ValueError: If scale_rate is zero or image has invalid channels.
        """
        scale_rate = ImageUtils.get_rate(scale_rate)
        if scale_rate == 1:
            return img
        elif scale_rate == 0:
            raise ValueError("scale_rate cannot be zero")
        h, w, channels = img.shape
        if channels == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            new_size = (int(w * scale_rate), int(h * scale_rate))
            bgr_resized = cv2.resize(bgr, new_size, interpolation=cv2.INTER_LINEAR)
            alpha_resized = cv2.resize(alpha, new_size, interpolation=cv2.INTER_LINEAR)
            scaled_img = np.dstack((bgr_resized, alpha_resized))
        elif channels == 3:
            scaled_img = cv2.resize(img, (int(w * scale_rate), int(h * scale_rate)), interpolation=cv2.INTER_LINEAR)
        return scaled_img

    @staticmethod
    def calculate_out_of_bounds_percentage(x, y, bw, bh, ow, oh):
        """
        Calculates the percentage of the object outside the background bounds.

        Args:
            x (int): X-coordinate of object placement.
            y (int): Y-coordinate of object placement.
            bw (int): Background width.
            bh (int): Background height.
            ow (int): Object width.
            oh (int): Object height.

        Returns:
            tuple: (percent_x, percent_y) of out-of-bounds percentages.
        """
        if x < 0:
            percent_x = abs(x) / ow
        elif x + ow > bw:
            percent_x = (x + ow - bw) / ow
        else:
            percent_x = 0

        if y < 0:
            percent_y = abs(y) / oh
        elif y + oh > bh:
            percent_y = (y + oh - bh) / oh
        else:
            percent_y = 0
        return percent_x, percent_y

    @staticmethod
    def get_coord_info(x, y, h, w, bh, bw):
        """
        Generates normalized coordinates and dimensions for classification output.

        Args:
            x (int): X-coordinate of object placement.
            y (int): Y-coordinate of object placement.
            h (int): Object height.
            w (int): Object width.
            bh (int): Background height.
            bw (int): Background width.

        Returns:
            str: 'cx cy ax ay' with normalized center coordinates and sizes.
        """
        cx = str((x + w / 2) / bw)
        cy = str((y + h / 2) / bh)
        ax = str(w / bw)
        ay = str(h / bh)
        return " ".join([cx, cy, ax, ay])

    @staticmethod
    def image_rotation(img, angle_rate,borderMode = None, interpolation_type=cv2.INTER_LINEAR, save_original_shape=True):
        """
        Rotates the image by a given or random angle.

        Args:
            img (np.ndarray): Input image.
            angle_rate (float or tuple): Rotation angle or range (degrees).
            borderMode (int, optional): Border mode for cv2.warpAffine.
            interpolation_type (int): Interpolation flag (default cv2.INTER_LINEAR).
            save_original_shape (bool): Keep original dimensions if True.

        Returns:
            tuple: (angle, rotated_img).
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        if isinstance(angle_rate, (list, tuple)):
            angle = int(ImageUtils.get_rate(angle_rate))
        else:
            angle = angle_rate
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        if save_original_shape:
            output_size = (w, h)
        else:
            corners = np.array([
                [w / 2, -h / 2],
                [w / 2, h / 2],
                [-w / 2, -h / 2],
                [-w / 2, h / 2]
            ])
            cos_theta = matrix[0, 0]
            sin_theta = matrix[0, 1]
            rotation_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])
            rotated_corners = np.dot(corners, rotation_matrix.T)

            new_w = int(2 * np.max(np.abs(rotated_corners[:, 0])))
            new_h = int(2 * np.max(np.abs(rotated_corners[:, 1])))
            output_size = (new_w, new_h)

            matrix[0, 2] += (new_w - w) / 2
            matrix[1, 2] += (new_h - h) / 2

        if borderMode:
            rotated_img = cv2.warpAffine(img, matrix, output_size, borderMode=cv2.BORDER_REFLECT, flags=interpolation_type)
        else:
            rotated_img = cv2.warpAffine(img, matrix, output_size, flags=interpolation_type)
        return angle, rotated_img

    @staticmethod
    def apply_color_jitter(image, contrast_brightness_probability, brightness_range, contrast_range,
                           saturation_range, hue_range, jitter_probability):
    # def apply_color_jitter(image, contrast_brightness_probability= 0.4, brightness_range = (0.7,1.3), contrast_range = (0.7,1.3),
    #                        saturation_range=(0.7, 1.5), hue_range=(-0.1, 0.1), jitter_probability=0.8):
        """
        Applies color jitter to adjust brightness, contrast, saturation, and hue.

        Args:
            image (np.ndarray): Input image (BGR).
            contrast_brightness_probability (float): Probability of brightness/contrast adjustment.
            brightness_range (tuple): Range for brightness factor.
            contrast_range (tuple): Range for contrast factor.
            saturation_range (tuple): Range for saturation factor.
            hue_range (tuple): Range for hue shift.
            jitter_probability (float): Probability of saturation/hue adjustment.

        Returns:
            np.ndarray: Jittered image (BGR).

        Raises:
            ValueError: If image is not 3-channel BGR.
        """
        if np.random.rand() >= max(jitter_probability, contrast_brightness_probability):
            return image
        image_float = image.astype(np.float32)

        if np.random.rand() < contrast_brightness_probability:
            contrast_factor = ImageUtils.get_rate(contrast_range)
            mean_per_channel = np.mean(image_float, axis=(0, 1))
            image_float = mean_per_channel + contrast_factor * (image_float - mean_per_channel)

        hsv = cv2.cvtColor(image_float.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

        if np.random.rand() < contrast_brightness_probability:
            brightness_factor = ImageUtils.get_rate(brightness_range)
            hsv[:, :, 2] *= brightness_factor
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        if np.random.rand() < jitter_probability:
            # Saturation
            saturation_factor = ImageUtils.get_rate(saturation_range)
            hsv[:, :, 1] *= saturation_factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

            # Hue
            hue_shift = ImageUtils.get_rate(hue_range) * 180  # ±18°
            hsv[:, :, 0] += hue_shift
            hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)

        hsv_uint8 = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)

    @staticmethod
    def sigmoid_contrast(image, k):
        """
        Applies sigmoid contrast adjustment to the image.

        Args:
            image (np.ndarray): Input image.
            k_range (tuple): Range for sigmoid steepness parameter k.

        Returns:
            np.ndarray: Adjusted image (uint8).
        """
        myu = 0.5
        image = image / 255.0
        image = 1 / (1 + np.exp(-k * (image - myu)))
        return (image * 255).astype(np.uint8)


###############################################################################
# ImageGeneratorUtils Class
###############################################################################
class ImageGeneratorUtils:
    """
        Generates synthetic images by overlaying objects onto backgrounds with transformations.
    """
    @staticmethod
    def generate_point(bg_img, obj_img):
        """
        Determines placement coordinates and scales the object for overlaying.

        Args:
            bg_img (np.ndarray): Background image.
            obj_img (np.ndarray): Object image (3 or 4 channels).

        Returns:
            tuple: (x, y, scaled_obj) with placement coordinates and scaled object.

        Raises:
            ValueError: If placement_distribution is invalid or image channels are incorrect.
        """
        oh, ow = obj_img.shape[:2]  # Object original dimensions.
        bh, bw = bg_img.shape[:2]  # Background dimensions.

        # Calculate allowed out-of-boundary offsets if enabled.
        if allow_out_of_bounds:
            out_of_bounds_rate_w = ImageUtils.get_rate(out_of_bounds_range)
            out_of_bounds_rate_h = ImageUtils.get_rate(out_of_bounds_range)
            fx = -int(out_of_bounds_rate_w * ow)
            fy = 0  # Could be adjusted based on requirements.
            # fy = -int(out_of_bounds_rate_h * oh)
            tx = bw - int((1 - out_of_bounds_rate_w) * ow)
            ty = bh - int((1 - out_of_bounds_rate_h) * oh)
        else:
            fx, fy = 0, 0
            tx = max(1, bw - ow)
            ty = max(1, bh - oh)
        # Depth-based placement adjustments.
        if depth:
            bm = bh
            bt = 0
            if allow_out_of_bounds:
                bm = bh - int(oh * (1 - out_of_bounds_range[1]))
                bt = 0
        else:
            bm, bt = bh, 0
        # Scale the object image using a global scaling range.

        scaled_obj = ImageUtils.scale_img(obj_img, object_scale_range)

        # Determine object placement based on the selected distribution.
        if placement_distribution == "gaussian":
            mu_x = fx + 0.5 * (tx - fx)
            sigma_x = ((tx - fx) / 4) * (94 / 100)
            x = int(np.random.normal(mu_x, sigma_x))
            mu_y = bt + (1 - (1 / 8 + 1 / 16)) * (bm - bt)
            sigma_y = ((bm - bt) / 16) * (99 / 100)
            y = int(np.random.normal(mu_y, sigma_y))
            # Clamp the coordinates within the allowed range.
            x = np.clip(x, fx, tx)
            y = np.clip(y, fy, ty)
        elif placement_distribution == "uniform":
            x = np.random.randint(fx, tx)
            y = np.random.randint(fy, ty)
        else:
            x, y = 0, 0
        # Adjust scaling based on depth if required.
        if depth:
            scale_rate = 0.1 + (1 - 0.1) * y / bm
            new_size = (int(ow * scale_rate), int(oh * scale_rate))
            h, w, channels = scaled_obj.shape

            if channels == 4:
                bgr = scaled_obj[:, :, :3]
                alpha = scaled_obj[:, :, 3]

                # INTER_AREA
                bgr_resized = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
                alpha_resized = cv2.resize(alpha, new_size, interpolation=cv2.INTER_AREA)

                scaled_obj = np.dstack((bgr_resized, alpha_resized))

            elif channels == 3:
                scaled_obj = cv2.resize(scaled_obj, new_size, interpolation=cv2.INTER_AREA)
            # scaled_obj = cv2.resize(scaled_obj,
            #                         (int(ow * (0.1 + (1 - 0.1) * y / bm)), int(oh * (0.1 + (1 - 0.1) * y / bm))))
            #
            new_oh, new_ow = scaled_obj.shape[:2]

            if allow_out_of_bounds:
                percent_x, percent_y = ImageUtils.calculate_out_of_bounds_percentage(x, y, bw, bh, ow, oh)
                if (x + ow > bw) or (x < 0) or (y < 0) or (y + oh > bh):
                    if x + ow > bw:
                        delta_x = (1 - percent_x) * (ow - new_ow)
                        x += delta_x
                    elif x < 0:
                        delta_x = percent_x * (ow - new_ow)
                        x += delta_x
                    if y < 0:
                        delta_y = percent_y * (oh - new_oh)
                        y += delta_y
                    x = int(x)
                    y = int(y)
                x = np.clip(x, -int(new_ow * percent_x), bw - int((1 - percent_x) * new_ow))
                y = np.clip(y, -int(new_oh * percent_y), bh - int((1 - percent_y) * new_oh))

        return x, y, scaled_obj

    @staticmethod
    def generate_img(bg_img, generate_obj_point, angle):
        """
        Overlays a transformed object onto the background.

        Args:
            bg_img (np.ndarray): Background image.
            generate_obj_point (tuple): (x, y, scaled_obj) from generate_point.
            angle (float): Rotation angle for the object.

        Returns:
            tuple: (combined_img, output) where output is coordinates (classification) or mask (segmentation).

        Raises:
            ValueError: If output_format is invalid.
        """
        x, y, scaled_obj = generate_obj_point
        # Optionally flip the object image horizontally.

        flipped_obj = ImageUtils.flip_img(scaled_obj, synthetic_flip_probability)
        # Rotate the object if foreground rotation is enabled.
        if foreground_rotation:
            _, flipped_obj = ImageUtils.image_rotation(flipped_obj, angle, borderMode=None ,interpolation_type=cv2.INTER_LINEAR, save_original_shape = False)

        # Copy the object image for alpha processing.
        blured_img = flipped_obj.copy()
        # Extract alpha channel and normalize it.
        alpha = (blured_img[:, :, -1] / 255.0).astype(np.float32)
        # Apply blur to the alpha channel.
        blurred_alpha = ImageUtils.add_alpha_blur(alpha, alpha_blur_type, synthetic_alpha_blur_kernel_range)
        # Combine the original and blurred alpha values.
        blured_img[:, :, -1] = (blurred_alpha * alpha * 255).astype(np.uint8)

        # Overlay the object onto the background using the 'put' method.
        img, bg_alpha = ImageGeneratorUtils.put(x, y, blured_img, bg_img)
        output = None
        # Generate output based on desired format.
        if output_format == 'classification':
            oh, ow = scaled_obj.shape[:2]
            bh, bw = bg_img.shape[:2]
            output = ImageUtils.get_coord_info(x, y, oh, ow, bh, bw)
        elif output_format == 'segmentation':
            output = bg_alpha

        return img, output

    @staticmethod
    def put(x, y, obj_img, bg_img):
        """
        Pastes an object onto the background, blending with alpha if present.

        Args:
            x (int): X-coordinate for placement.
            y (int): Y-coordinate for placement.
            obj_img (np.ndarray): Object image (3 or 4 channels).
            bg_img (np.ndarray): Background image (3 channels).

        Returns:
            tuple: (result_img, alpha_mask).

        Raises:
            Exception: If placement is completely out-of-bounds.
            ValueError: If image channels are incompatible.
        """
        bh, bw, c = bg_img.shape
        h, w = obj_img.shape[:2]
        # Check if the object lies completely out of the background boundaries.
        if 1 - w > x or x > bw - 1 or 1 - h > y or y > bh - 1:
            raise Exception("out of bounds")
        fh, th = 0, h
        fw, tw = 0, w
        # Adjust offsets if the object partially lies outside.
        if x < 0:
            fw = -x
        if y < 0:
            fh = -y
        if h > bh - y:
            th = bh - y
        if w > bw - x:
            tw = bw - x

        bg = bg_img.copy()
        # Extract the region of the background where the object will be pasted.
        paste = bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)]
        # Extract the corresponding region from the object (ignoring alpha channel).
        obj = obj_img[fh:th, fw:tw, :3]

        bg_alpha = None
        # If the object image has an alpha channel, blend using the alpha mask.
        if obj_img.shape[2] == 4:
            alpha = obj_img[fh:th, fw:tw, -1] / 255.0
            alpha_n = np.array([alpha] * 3).transpose((1, 2, 0))
            alpha_t = 1.0 - alpha_n
            # Blend the background and object images.
            bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] =  np.clip(paste * alpha_t + obj * alpha_n, 0, 255).astype(np.uint8)
            # Create an alpha mask for the pasted region.
            bg_alpha = np.zeros((bh, bw, c), dtype=np.uint8)
            bg_alpha[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = np.clip(alpha_n * 255, 0, 255).astype(np.uint8)
        else:
            # If no alpha channel, directly replace the background region with the object.
            bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = obj

        return bg, bg_alpha
