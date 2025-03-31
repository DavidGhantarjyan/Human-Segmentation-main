import cv2
import numpy as np

from other.parsing.train_args_parser import *
import random


# Import configuration parameters from train_args_parser, e.g. images_per_combination, result_dir, etc.

###############################################################################
# FileManager Class
###############################################################################
class FileManager:
    @staticmethod
    def get_amount(obj, bg):
        """
        Calculates the total number of images generated based on the number of objects and backgrounds.

        :param obj: List of object images.
        :param bg: List of background images.
        :return: Total number of generated images (integer).
        """
        # Multiply images_per_combination (a global parameter) by the number of objects and backgrounds.

        return images_per_combination * len(obj) * len(bg)

    @staticmethod
    def get_new_epoch_path():
        """
               Determines a new unique epoch directory name for saving generated results.

               :return: Path to a new epoch directory, ensuring no name conflicts in the result directory.
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
    def get_objects():
        """
                Loads all object images from the specified objects directory.

                :return: List of object images loaded via cv2.imread with unchanged flags (to preserve alpha channel).
        """
        obj_images = []
        for name in os.listdir(objects_dir):
            path = os.path.join(objects_dir, name)
            # Use cv2.IMREAD_UNCHANGED (-1) to load image along with alpha channel if present.
            obj_images.append(cv2.imread(path, -1))
        return obj_images

    @staticmethod
    def save_as_txt(text, path):
        """
               Appends the given text to a text file at the specified path.

               :param text: Text to save.
               :param path: File path (without extension) to which text is appended.
        """
        with open(path + ".txt", 'a') as f:
            f.write(text + '\n')

    @staticmethod
    def save_as_grayscale_img(img, path):
        """
               Saves a grayscale image to disk.

               If the image is in BGR format, it takes only one channel. It also ensures that the image
               data type is uint8 and values are clipped between 0 and 255.

               :param img: Image (as NumPy array) to be saved.
               :param path: Destination file path.
               :raises IOError: If the image fails to save.
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
    @staticmethod
    def get_rate(rate):
        """
                Determines a random rate value based on input.

                :param rate: Could be a tuple, list, or a fixed value.
                             - If tuple: returns a random float between the two values.
                             - If list: returns a random choice from the list.
                             - Otherwise, returns the given value.
                :return: A float or the original rate.
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
        Flips the image horizontally based on a given probability.

        :param img: Input image as a NumPy array.
        :param flip_chance: Probability threshold for performing a flip.
        :return: Flipped image if random chance passes; otherwise, original image.
        """
        if np.random.rand() < flip_chance:
            return img
        return cv2.flip(img, 1)

    @staticmethod
    def add_alpha_blur(alpha, alpha_blur_type, alpha_blur_kernel_range):
        """
        Applies blur to the alpha channel of an image.

        Currently supports Gaussian blur.

        :param alpha: Alpha channel as a single-channel image (NumPy array).
        :param alpha_blur_type: Type of blur (e.g., 'GaussianBlur').
        :param alpha_blur_kernel_range: Tuple indicating min and max kernel size.
        :return: Blurred alpha channel.
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
        Applies blur to the background image while preserving regions where label equals 1.

        :param img: Background image as a NumPy array.
        :param label: Label or segmentation mask (used to decide which areas to blur).
        :param bg_blur_chance: Probability threshold to apply the blur.
        :param bg_blur_type: Type of blur to apply (e.g., 'GaussianBlur').
        :param bg_blur_kernel_range: Tuple of kernel size range for the blur.
        :return: Image with selective background blur applied.
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
    def add_noise(img, noise_rate):
        """
        Adds random noise to an image. The noise can be uniform or Gaussian.

        :param img: Input image as a NumPy array.
        :param noise_rate: Rate parameter (can be a tuple/list or fixed value) defining noise intensity.
        :return: Noisy image.
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
        # Ensure that image values stay within valid range after adding noise.
        return np.clip(img, arg, 255 - arg) + noise

    @staticmethod
    def scale_img(img, scale_rate):
        """
        Scales the image by a given scale factor.

        :param img: Input image as a NumPy array.
        :param scale_rate: Scaling factor (can be generated randomly).
        :return: Resized image.
        :raises ValueError: If scale_rate is zero.
        """
        scale_rate = ImageUtils.get_rate(scale_rate)
        if scale_rate == 1:
            return img
        elif scale_rate == 0:
            raise ValueError("scale_rate cannot be zero")
        h, w, _ = img.shape
        scaled_img = cv2.resize(img, (int(w * scale_rate), int(h * scale_rate)))
        return scaled_img

    @staticmethod
    def calculate_out_of_bounds_percentage(x, y, bw, bh, ow, oh):
        """
        Calculates the percentage of the object that lies out of the background bounds.

        :param x: X-coordinate of the object's placement.
        :param y: Y-coordinate of the object's placement.
        :param bw: Width of the background.
        :param bh: Height of the background.
        :param ow: Original width of the object.
        :param oh: Original height of the object.
        :return: Tuple (percent_x, percent_y) representing the out-of-bounds percentage in each dimension.
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
        Generates a string containing normalized coordinates and dimensions.

        :param x: X-coordinate of the object placement.
        :param y: Y-coordinate of the object placement.
        :param h: Height of the object.
        :param w: Width of the object.
        :param bh: Height of the background.
        :param bw: Width of the background.
        :return: A string with normalized center coordinates and size: 'cx cy ax ay'.
        """
        cx = str((x + w / 2) / bw)
        cy = str((y + h / 2) / bh)
        ax = str(w / bw)
        ay = str(h / bh)
        return " ".join([cx, cy, ax, ay])

    @staticmethod
    def image_rotation(img, angle_rate, interpolation_type=cv2.INTER_LINEAR):
        """
        Rotates an image by a given angle.

        :param img: Input image as a NumPy array.
        :param angle_rate: Rotation angle or range (if tuple/list, a random angle is chosen).
        :param interpolation_type: Interpolation flag for cv2.warpAffine.
        :return: Tuple (angle, rotated_img), where 'angle' is the used rotation angle.
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        if isinstance(angle_rate, (list, tuple)):
            angle = int(ImageUtils.get_rate(angle_rate))
        else:
            angle = angle_rate
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT, flags=interpolation_type)
        return angle, rotated_img

    @staticmethod
    def apply_color_jitter(image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3),
                           saturation_range=(0.7, 1.5), hue_range=(-0.1, 0.1), jitter_probability=0.8):
        """
        Applies color jitter to the image, adjusting brightness, contrast, saturation, and hue.

        :param image: Input image as a NumPy array.
        :param brightness_range: Tuple specifying the range for brightness adjustment.
        :param contrast_range: Tuple specifying the range for contrast adjustment.
        :param saturation_range: Tuple specifying the range for saturation adjustment.
        :param hue_range: Tuple specifying the range for hue adjustment.
        :param jitter_probability: Probability threshold for applying jitter.
        :return: Color jittered image.
        """
        if np.random.rand() < jitter_probability:
            return image
        image_float = image.astype(np.float32)
        contrast_factor = ImageUtils.get_rate(contrast_range)
        mean_per_channel = np.mean(image_float, axis=(0, 1))
        contrasted = mean_per_channel + contrast_factor * (image_float - mean_per_channel)
        image_contrast = np.clip(contrasted, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2HSV).astype(np.float32)

        hsv[:, :, 2] *= ImageUtils.get_rate(brightness_range)
        hsv[:, :, 1] *= ImageUtils.get_rate(saturation_range)
        hsv[:, :, 0] += ImageUtils.get_rate(hue_range) * 180

        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        hsv_uint8 = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2BGR)


###############################################################################
# ImageGeneratorUtils Class
###############################################################################
class ImageGeneratorUtils:
    @staticmethod
    def generate_point(bg_img, obj_img):
        """
        Determines the placement coordinates and scales the object image for overlaying onto the background.

        :param bg_img: Background image as a NumPy array.
        :param obj_img: Object image as a NumPy array.
        :return: A tuple (x, y, scaled_obj) where (x, y) are the top-left coordinates and scaled_obj is the resized object.
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
            scaled_obj = cv2.resize(scaled_obj,
                                    (int(ow * (0.1 + (1 - 0.1) * y / bm)), int(oh * (0.1 + (1 - 0.1) * y / bm))))

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
        Overlays a processed object image onto the background at a given coordinate.

        Applies optional flipping and rotation to the object, blurs the alpha channel, and then overlays
        it on the background image.

        :param bg_img: Background image as a NumPy array.
        :param generate_obj_point: Tuple (x, y, scaled_obj) from generate_point.
        :param angle: Rotation angle to apply to the object.
        :return: Tuple (combined_img, output) where output is either coordinate information or a segmentation mask.
        """
        x, y, scaled_obj = generate_obj_point
        # Optionally flip the object image horizontally.

        flipped_obj = ImageUtils.flip_img(scaled_obj, flip_probability)
        # Rotate the object if foreground rotation is enabled.
        if foreground_rotation:
            _, flipped_obj = ImageUtils.image_rotation(flipped_obj, angle)

        # Copy the object image for alpha processing.
        blured_img = flipped_obj.copy()
        # Extract alpha channel and normalize it.
        alpha = (blured_img[:, :, -1] / 255.0).astype(np.float32)
        # Apply blur to the alpha channel.
        blurred_alpha = ImageUtils.add_alpha_blur(alpha, alpha_blur_type, alpha_blur_kernel_range)
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
        Pastes an object image (with potential alpha channel) onto a background image at position (x, y).

        Handles out-of-bound placements by clipping and blends the object with the background using the alpha mask.

        :param x: X-coordinate for object placement.
        :param y: Y-coordinate for object placement.
        :param obj_img: Object image with an alpha channel (if present).
        :param bg_img: Background image.
        :return: Tuple (result_img, alpha_mask) where alpha_mask represents the pasted region's transparency.
        :raises Exception: If the placement is completely out-of-bounds.
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
            bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = paste * alpha_t + obj * alpha_n
            # Create an alpha mask for the pasted region.
            bg_alpha = np.zeros((bh, bw, c), dtype=np.uint8)
            bg_alpha[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = alpha_n * 255
        else:
            # If no alpha channel, directly replace the background region with the object.
            bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = obj

        return bg, bg_alpha
