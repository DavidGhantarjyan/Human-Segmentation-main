import cv2
from other.parsing.train_args_parser import *
import random

class FileManager:
    @staticmethod
    def get_amount(obj, bg):
        return images_per_combination * len(obj) * len(bg)

    @staticmethod
    def get_new_epoch_path():
        m = 0
        p = os.listdir(result_dir)
        while True:
            ep_dir = f"epoch{m}"
            if ep_dir in p:
                m += 1
            else:
                break
        ep_dir = os.path.join(result_dir, ep_dir) + os.sep
        return ep_dir

    @staticmethod
    def get_objects():
        obj_images = []
        for name in os.listdir(objects_dir):
            path = os.path.join(objects_dir, name)
            obj_images.append(cv2.imread(path, -1))
        return obj_images

    @staticmethod
    def save_as_txt(text, path):
        with open(path + ".txt", 'a') as f:
            f.write(text + '\n')


    @staticmethod
    def save_as_grayscale_img(img, path):
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, 0]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        success = cv2.imwrite(path, img)
        if not success:
            raise IOError(f"Failed to save image to {path}")


class ImageUtils:
    @staticmethod
    def get_rate(rate):
        if isinstance(rate, tuple):
            return np.random.rand() * (rate[1] - rate[0]) + rate[0]
        elif isinstance(rate, list):
            return np.random.choice(rate, 1)
        else:
            return rate

    @staticmethod
    def flip_img(img, flip_chance):
        if np.random.rand() < flip_chance:
            return img
        return cv2.flip(img, 1)

    @staticmethod
    def add_alpha_blur(alpha,alpha_blur_type,alpha_blur_kernel_range):
        if alpha_blur_type == 'GaussianBlur':
            k_min,k_max = alpha_blur_kernel_range
            ksize = random.randrange(k_min,k_max+1, 2)
            blured_alpha = cv2.GaussianBlur(alpha,ksize=(ksize,ksize),sigmaX=0)
            return blured_alpha
        return alpha

    @staticmethod
    def add_background_blur(img,label,bg_blur_chance,bg_blur_type,bg_blur_kernel_range):
        blured_img = img
        if bg_blur_chance:
            if np.random.rand() < (bg_blur_chance):
                return blured_img
        if bg_blur_type == 'GaussianBlur':
            k_min, k_max = bg_blur_kernel_range
            ksize = random.randrange(k_min,k_max+1, 2)
            bg_blured = cv2.GaussianBlur(blured_img,(ksize,ksize),sigmaX=0)
            bg_blured = np.where(label == 1, blured_img, bg_blured)
            return bg_blured

    @staticmethod
    def add_noise(img, noise_rate):
        noise_rate = ImageUtils.get_rate(noise_rate)
        arg = int(noise_rate * 127)
        if arg == 0:
            return img

        if noise_type == 'uniform':
            noise = np.random.randint(-arg, arg, img.shape)
        elif noise_type == 'gaussian':
            noise = np.random.randn(*img.shape) * arg
            noise = np.clip(noise, -arg, arg)
        return np.clip(img, arg, 255 - arg) + noise

    @staticmethod
    def scale_img(img, scale_rate):
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
        cx = str((x + w / 2) / bw)
        cy = str((y + h / 2) / bh)
        ax = str(w / bw)
        ay = str(h / bh)
        return " ".join([cx, cy, ax, ay])

    @staticmethod
    def image_rotation(img, angle_rate):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        if isinstance(angle_rate, (list, tuple)):
            angle = int(ImageUtils.get_rate(angle_rate))
        else:
            angle = angle_rate
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        rotated_resized_img = cv2.resize(rotated_img, (w, h))
        return angle, rotated_resized_img

    @staticmethod
    def apply_color_jitter(image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3),
                           saturation_range=(0.7, 1.5), hue_range=(-0.1, 0.1), jitter_probability=0.8):

        if np.random.rand() < jitter_probability:
            return image
        image_float = image.astype(np.float32)
        contrast_factor = ImageUtils.get_rate(contrast_range)
        mean_per_channel  = np.mean(image_float, axis=(0, 1))
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


class ImageGeneratorUtils:
    @staticmethod
    def generate_point(bg_img, obj_img):
        # Scale object image according to config
        oh,ow = obj_img.shape[:2]
        bh,bw = bg_img.shape[:2]
        if allow_out_of_bounds:
            out_of_bounds_rate_w = ImageUtils.get_rate(out_of_bounds_range)
            out_of_bounds_rate_h = ImageUtils.get_rate(out_of_bounds_range)
            fx = -int(out_of_bounds_rate_w * ow)
            fy = 0
            # fy = -int(out_of_bounds_rate_h * oh)
            tx = bw - int((1 - out_of_bounds_rate_w) * ow)
            ty = bh - int((1 - out_of_bounds_rate_h) * oh)
        else:
            fx, fy = 0, 0
            tx = max(1, bw - ow)
            ty = max(1, bh - oh)

        if depth:
            bm = bh
            bt = 0
            if allow_out_of_bounds:
                bm = bh - int(oh * (1 - out_of_bounds_range[1]))
                bt = 0
        else:
            bm, bt = bh, 0

        scaled_obj = ImageUtils.scale_img(obj_img, object_scale_range)


        # Placement distribution based on Gaussian or Uniform
        if placement_distribution == "gaussian":
            mu_x = fx + 0.5 * (tx - fx)
            sigma_x = ((tx - fx) / 4) * (94 / 100)
            x = int(np.random.normal(mu_x, sigma_x))
            mu_y = bt + (1 - (1 / 8 + 1/16 )) * (bm - bt)
            sigma_y = ((bm - bt) / 16) * (99 / 100)
            y = int(np.random.normal(mu_y, sigma_y))
            x = np.clip(x, fx, tx)
            y = np.clip(y, fy, ty)
        elif placement_distribution == "uniform":
            x = np.random.randint(fx, tx)
            y = np.random.randint(fy, ty)
        else:
            x, y = 0, 0

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
                x = np.clip(x, -int(new_ow * percent_x), bw - int((1-percent_x) * new_ow))
                y = np.clip(y, -int(new_oh * percent_y), bh - int((1-percent_y) * new_oh))

        return x, y, scaled_obj

    @staticmethod
    def generate_img(bg_img, generate_obj_point, angle):
        x, y, scaled_obj = generate_obj_point

        flipped_obj = ImageUtils.flip_img(scaled_obj, flip_probability)
        if foreground_rotation:
            _, flipped_obj = ImageUtils.image_rotation(flipped_obj, angle)

        blured_img = flipped_obj.copy()
        alpha = (blured_img[:, :, -1] / 255.0).astype(np.float32)
        blurred_alpha = ImageUtils.add_alpha_blur(alpha,alpha_blur_type,alpha_blur_kernel_range)
        blured_img[:, :, -1] = (blurred_alpha * alpha * 255).astype(np.uint8)

        img, bg_alpha = ImageGeneratorUtils.put(x, y, blured_img, bg_img)
        output = None
        if output_format == 'classification':
            oh, ow = scaled_obj.shape[:2]
            bh, bw = bg_img.shape[:2]
            output = ImageUtils.get_coord_info(x, y, oh, ow, bh, bw)
        elif output_format == 'segmentation':
            output = bg_alpha

        return img, output

    @staticmethod
    def put(x, y, obj_img, bg_img):
        bh, bw, c = bg_img.shape
        h, w = obj_img.shape[:2]
        if 1 - w > x or x > bw - 1 or 1 - h > y or y > bh - 1:
            raise Exception("out of bounds")
        fh, th = 0, h
        fw, tw = 0, w
        if x < 0:
            fw = -x
        if y < 0:
            fh = -y
        if h > bh - y:
            th = bh - y
        if w > bw - x:
            tw = bw - x

        bg = bg_img.copy()
        paste = bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)]
        obj = obj_img[fh:th, fw:tw, :3]

        bg_alpha = None
        if obj_img.shape[2] == 4:
            alpha = obj_img[fh:th, fw:tw, -1] / 255.0
            alpha_n = np.array([alpha] * 3).transpose((1, 2, 0))
            alpha_t = 1.0 - alpha_n
            bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = paste * alpha_t + obj * alpha_n

            bg_alpha = np.zeros((bh, bw, c), dtype=np.uint8)
            bg_alpha[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = alpha_n * 255
        else:
            bg[max(0, y):min(y + h, bh), max(0, x):min(x + w, bw)] = obj

        return bg, bg_alpha
