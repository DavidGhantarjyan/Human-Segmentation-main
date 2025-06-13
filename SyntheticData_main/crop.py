import cv2
import os
from progressbar import ProgressBar
from other.parsing.train_args_parser import objects_dir

"""
Script to crop RGBA images based on their alpha channel, removing transparent borders.

Reads images from objects_dir, identifies non-transparent regions, crops with padding,
and saves results to objects_dir_cut. Displays progress using a progress bar.
"""

# Define input and output directories
path = objects_dir
result = path[:-1] + "_cut/"

# Create output directory if it doesn't exist
if not os.path.isdir(result):
    os.makedirs(result)

# Initialize lists to store images and filenames
obj_images = []
file_names = []

# Load all images from input directory
for name in os.listdir(path):
    file_path = os.path.join(path, name)
    img = cv2.imread(file_path, -1)   # Load with alpha channel

    # Validate image and ensure 4 channels (RGBA)
    if img is not None and img.shape[2] == 4:
        obj_images.append(img)
        file_names.append(name)
    else:
        print(f"Warning: Skipped {name} (invalid image or missing alpha channel)")

# Initialize progress bar for image processing
with ProgressBar(max_value=len(obj_images)) as bar:
    for idx, (img, name) in enumerate(zip(obj_images, file_names)):
        """
        Crop image based on non-transparent region in alpha channel.

        Args:
            img (np.ndarray): RGBA image, shape (H, W, 4).
            name (str): Filename for saving cropped image.
        """
        ly, lx, _ = img.shape  # Image dimensions: height (ly), width (lx), channels

        # Initialize cropping bounds
        maxY, maxX = -1, -1
        minY, minX = -1, -1

        # Find top (minY) where alpha channel becomes non-zero
        for y in range(ly):
            if sum(img[y, :, 3]) != 0:
                minY = y
                break

        # Find bottom (maxY) where alpha channel becomes non-zero
        for y in range(1, ly):
            if sum(img[-y, :, 3]) != 0:
                maxY = ly - y + 1
                break

        # Find left (minX) where alpha channel becomes non-zero
        for x in range(lx):
            if sum(img[:, x, 3]) != 0:
                minX = x
                break

        # Find right (maxX) where alpha channel becomes non-zero
        for x in range(1, lx):
            if sum(img[:, -x, 3]) != 0:
                maxX = lx - x + 1
                break

        # Crop image with 1-pixel padding around non-transparent region
        cropped_img = img[max(0, minY - 1):maxY + 2, max(0, minX - 1):maxX + 2, :]

        # Save cropped image to output directory
        save_path = os.path.join(result, name)
        cv2.imwrite(save_path, cropped_img)

        # Update progress bar
        bar.update(idx + 1)

