import cv2
import numpy as np
import math
import cupy as cp
import cupyx.scipy.ndimage as ndimage


def remove_letterbox_padding(pred_mask, original_h, original_w, padded_size):
    padded_h, padded_w = padded_size

    aspect_ratio = original_w / original_h
    padded_aspect = padded_w / padded_h

    # w > 16/9 * h
    if aspect_ratio > padded_aspect:
        scale = padded_w / original_w
        new_w = padded_w
        new_h = int(original_h * scale)
    # w < 16/9 * h
    elif aspect_ratio < padded_aspect:
        scale = padded_h / original_h
        new_h = padded_h
        new_w = int(original_w * scale)
    else:
        new_h = padded_h
        new_w = padded_w

    pad_left = (padded_w - new_w) // 2
    pad_top = (padded_h - new_h) // 2

    cropped = pred_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    output_resized = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return output_resized


def letterbox_resize_gpu(image, target_size):
    target_w, target_h = target_size
    h, w = image.size()[:2]
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    if aspect_ratio > target_aspect:
        scale = target_w / w
        new_w = target_w
        new_h = int(h * scale)
    elif aspect_ratio < target_aspect:
        scale = target_h / h
        new_h = target_h
        new_w = int(w * scale)
    else:
        new_h = target_h
        new_w = target_w

    resized_gpu = cv2.cuda.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    padded_gpu = cv2.cuda.copyMakeBorder(
        resized_gpu,
        pad_top, pad_bottom,
        pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    pad_info = (pad_top, pad_bottom, pad_left, pad_right, new_h, new_w)
    return padded_gpu, pad_info


def gpumat_to_cupy(gpu_mat):
    # address of memory
    ptr = int(gpu_mat.cudaPtr())
    width, height = gpu_mat.size()
    step = gpu_mat.step
    unowned = cp.cuda.UnownedMemory(ptr, step * height, gpu_mat)
    memptr = cp.cuda.MemoryPointer(unowned, 0)
    arr = cp.ndarray(
        shape=(height, width, 3),
        dtype=cp.uint8,
        memptr=memptr,
        strides=(step, 3, 1)
    )
    return arr


def preprocess_frame_gpu(frame: np.ndarray, target_h: int, target_w: int):
    gpu_frame = cv2.cuda.GpuMat()
    # CV_8UC3 <=> uint8
    gpu_frame.upload(frame)

    padded_gpu, pad_info = letterbox_resize_gpu(
        gpu_frame, (target_w, target_h)
    )
    rgb_gpu = cv2.cuda.cvtColor(padded_gpu, cv2.COLOR_BGR2RGB)

    rgb_gpu_np = gpumat_to_cupy(rgb_gpu)

    hwc = cp.asarray(rgb_gpu_np, dtype=cp.float16)

    hwc_normalized = hwc / 255.0

    nchw = cp.ascontiguousarray(
        cp.transpose(hwc_normalized, (2, 0, 1))[cp.newaxis, :]
    )
    return nchw, pad_info


def remove_letterbox_padding_from_pad_info(mask, pad_info, original_size):
    pad_top, pad_bottom, pad_left, pad_right, new_h, new_w = pad_info
    original_h, original_w = original_size

    cropped = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    scale_y = original_h / new_h
    scale_x = original_w / new_w
    restored = ndimage.zoom(cropped, (scale_y, scale_x), order=0, mode='nearest')
    return restored


def select_main_person(mask, prioritize_size=True, center_weight=0.5):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    h, w = mask.shape
    center_x, center_y = w / 2, h / 2
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        distance_to_center = math.hypot(cx - center_x, cy - center_y)
        components.append({'label': i, 'area': area, 'distance': distance_to_center})

    max_area = max(comp['area'] for comp in components)
    max_distance = max(comp['distance'] for comp in components) or 1

    selected = max(
        components,
        key=lambda c: (c['area'] / max_area) * (1 - center_weight) + (1 - c['distance'] / max_distance) * center_weight
    )
    new_mask = np.zeros_like(mask)
    new_mask[labels == selected['label']] = 1
    return new_mask
def blur_background(image, mask, kernel_size=(31, 31)):
    image_gpu = cv2.cuda.GpuMat()
    image_gpu.upload(image)
    gaussian_filter = cv2.cuda.createGaussianFilter(
        srcType=cv2.CV_8UC3,
        dstType=cv2.CV_8UC3,
        ksize=kernel_size,
        sigma1=0
    )
    blurred_gpu = gaussian_filter.apply(image_gpu)

    image_cupy = gpumat_to_cupy(image_gpu)
    blurred_cupy = gpumat_to_cupy(blurred_gpu)

    alpha = cp.clip(cp.asarray(mask[..., None], dtype=cp.float32), 0, 1)

    result_gpu = alpha * image_cupy.astype(cp.float32) + \
                 (1 - alpha) * blurred_cupy.astype(cp.float32)
    result_gpu = cp.clip(result_gpu, 0, 255)
    result_gpu = result_gpu.astype(cp.uint8)
    result = cp.asnumpy(result_gpu)

    return result
