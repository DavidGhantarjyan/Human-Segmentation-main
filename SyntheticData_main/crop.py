import cv2
import os
from progressbar import ProgressBar
from config import OBJECTS_DIR

path = OBJECTS_DIR
result = path[:-1] + "_cut/"

if not os.path.isdir(result):
    os.makedirs(result)

obj_images = []
file_names = []

for name in os.listdir(path):
    file_path = os.path.join(path, name)
    img = cv2.imread(file_path, -1)

    if img is not None:
        obj_images.append(img)
        file_names.append(name)

with ProgressBar(max_value=len(obj_images)) as bar:
    for idx, (img, name) in enumerate(zip(obj_images, file_names)):
        ly, lx, _ = img.shape
        maxY, maxX = -1, -1
        minY, minX = -1, -1

        for y in range(ly):
            if sum(img[y, :, 3]) != 0:
                minY = y
                break
        for y in range(1, ly):
            if sum(img[-y, :, 3]) != 0:
                maxY = ly - y + 1
                break

        for x in range(lx):
            if sum(img[:, x, 3]) != 0:
                minX = x
                break
        for x in range(1, lx):
            if sum(img[:, -x, 3]) != 0:
                maxX = lx - x + 1
                break

        cropped_img = img[max(0, minY - 1):maxY + 2, max(0, minX - 1):maxX + 2, :]

        save_path = os.path.join(result, name)
        cv2.imwrite(save_path, cropped_img)

        bar.update(idx + 1)

