import cv2
import numpy as np
import math
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import torchvision.transforms.functional as F
import torch

def remove_letterbox_padding(pred_mask, original_h, original_w, padded_size):
    """
    Remove letterbox padding from a mask and resize to original dimensions using CPU processing.

    Args:
        pred_mask (np.ndarray): Binary mask with padding, shape (padded_h, padded_w).
        original_h (int): Original height of the image.
        original_w (int): Original width of the image.
        padded_size (tuple): Padded dimensions (padded_h, padded_w).

    Returns:
        np.ndarray: Resized mask, shape (original_h, original_w).
    """
    padded_h, padded_w = padded_size

    aspect_ratio = original_w / original_h
    padded_aspect = padded_w / padded_h

    # Calculate scaling based on aspect ratio
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
    else:  # Same aspect ratio
        new_h = padded_h
        new_w = padded_w

    # Compute padding offsets
    pad_left = (padded_w - new_w) // 2
    pad_top = (padded_h - new_h) // 2

    # Crop padded region
    cropped = pred_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]

    # Resize to original dimensions using nearest-neighbor interpolation
    output_resized = cv2.resize(cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    return output_resized


def letterbox_resize_gpu(image, target_size):
    """
    Resize an image on GPU with letterbox padding to maintain aspect ratio.

    Args:
        image (cv2.cuda.GpuMat): Input image in BGR format, uint8.
        target_size (tuple): Target dimensions (width, height).

    Returns:
        tuple: (Padded GPU image (GpuMat), padding info (pad_top, pad_bottom, pad_left, pad_right, new_h, new_w)).
    """
    target_w, target_h = target_size
    h, w = image.size()[:2]
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    # Calculate scaling based on aspect ratio
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

    # Resize image on GPU with linear interpolation
    resized_gpu = cv2.cuda.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding amounts
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    # Add letterbox padding with black borders
    padded_gpu = cv2.cuda.copyMakeBorder(
        resized_gpu,
        pad_top, pad_bottom,
        pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # Return padded image and padding info
    pad_info = (pad_top, pad_bottom, pad_left, pad_right, new_h, new_w)
    return padded_gpu, pad_info


def gpumat_to_cupy(gpu_mat):
    """
    Convert an OpenCV CUDA GpuMat to a CuPy array for GPU processing.

    Args:
        gpu_mat (cv2.cuda.GpuMat): Input GpuMat, shape (height, width, 3), uint8.

    Returns:
        cp.ndarray: CuPy array, shape (height, width, 3), uint8.
    """
    # Get memory address and dimensions
    ptr = int(gpu_mat.cudaPtr())
    width, height = gpu_mat.size()
    step = gpu_mat.step
    # Create unowned memory and pointer
    unowned = cp.cuda.UnownedMemory(ptr, step * height, gpu_mat)
    memptr = cp.cuda.MemoryPointer(unowned, 0)
    # Create CuPy array with appropriate shape and strides
    arr = cp.ndarray(
        shape=(height, width, 3),
        dtype=cp.uint8,
        memptr=memptr,
        strides=(step, 3, 1)
    )
    return arr


def preprocess_frame_gpu(frame: np.ndarray, target_h: int, target_w: int):
    """
    Preprocess a frame on GPU for model input (resize, normalize, convert to NCHW).

    Args:
        frame (np.ndarray): Input frame in BGR format, shape (height, width, 3), uint8.
        target_h (int): Target height for resizing.
        target_w (int): Target width for resizing.

    Returns:
        tuple: (CuPy array in NCHW format, shape (1, 3, target_h, target_w), float16;
                padding info (pad_top, pad_bottom, pad_left, pad_right, new_h, new_w)).
    """
    # Upload frame to GPU
    gpu_frame = cv2.cuda.GpuMat()
    # CV_8UC3 (uint8, 3 channels)
    gpu_frame.upload(frame)

    # Resize with letterbox padding
    padded_gpu, pad_info = letterbox_resize_gpu(
        gpu_frame, (target_w, target_h)
    )
    # Convert BGR to RGB
    rgb_gpu = cv2.cuda.cvtColor(padded_gpu, cv2.COLOR_BGR2RGB)

    # Convert GpuMat to CuPy array
    rgb_gpu_np = gpumat_to_cupy(rgb_gpu)

    # Convert to float16 and normalize to [0, 1]

    hwc = cp.asarray(rgb_gpu_np, dtype=cp.float16)
    hwc_normalized = hwc / 255.0

    # Transpose to NCHW format (1, C, H, W)
    nchw = cp.ascontiguousarray(
        cp.transpose(hwc_normalized, (2, 0, 1))[cp.newaxis, :]
    )
    return nchw, pad_info


def remove_letterbox_padding_from_pad_info(mask, pad_info, original_size):
    """
    Remove letterbox padding from a mask using padding info and resize to original dimensions.

    Args:
        mask (np.ndarray or cp.ndarray): Binary mask with padding, shape (padded_h, padded_w).
        pad_info (tuple): Padding info (pad_top, pad_bottom, pad_left, pad_right, new_h, new_w).
        original_size (tuple): Original dimensions (height, width).

    Returns:
        np.ndarray or cp.ndarray: Resized mask, shape (original_h, original_w).
    """
    pad_top, pad_bottom, pad_left, pad_right, new_h, new_w = pad_info
    original_h, original_w = original_size

    # Crop padded region
    cropped = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    # Calculate scaling factors
    scale_y = original_h / new_h
    scale_x = original_w / new_w
    # Resize using nearest-neighbor interpolation (GPU-compatible)
    restored = ndimage.zoom(cropped, (scale_y, scale_x), order=0, mode='nearest')
    return restored


def select_main_person(mask, prioritize_size=True, center_weight=0.5):
    """
    Select the main person in a binary mask based on area and proximity to center.

    Args:
        mask (np.ndarray): Binary mask, shape (height, width).
        prioritize_size (bool): If True, prioritize larger components (unused in current logic).
        center_weight (float): Weight for distance-to-center in selection (0 to 1).

    Returns:
        np.ndarray: Binary mask with only the selected component, shape (height, width).
    """
    # Convert mask to binary uint8
    mask = (mask > 0).astype(np.uint8)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    # Get image center
    h, w = mask.shape
    center_x, center_y = w / 2, h / 2
    # Collect component stats
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        distance_to_center = math.hypot(cx - center_x, cy - center_y)
        components.append({'label': i, 'area': area, 'distance': distance_to_center})

    # Normalize area and distance
    max_area = max(comp['area'] for comp in components)
    max_distance = max(comp['distance'] for comp in components) or 1

    # Select component based on weighted area and distance
    selected = max(
        components,
        key=lambda c: (c['area'] / max_area) * (1 - center_weight) + (1 - c['distance'] / max_distance) * center_weight
    )
    # Create new mask with only the selected component
    new_mask = np.zeros_like(mask)
    new_mask[labels == selected['label']] = 1
    return new_mask

# def blur_background(image, alpha, kernel_size=(31, 31)):
#     """
#     Apply Gaussian blur to the background using OpenCV CUDA (commented out).
#
#     Args:
#         image (np.ndarray): Input image, shape (height, width, 3), uint8.
#         alpha (np.ndarray or cp.ndarray): Mask, shape (height, width, 1), float32.
#         kernel_size (tuple): Gaussian kernel size, default (31, 31).
#
#     Returns:
#         np.ndarray: Image with blurred background, shape (height, width, 3), uint8.
#     """
#     # Upload image to GPU
#     image_gpu = cv2.cuda.GpuMat()
#     image_gpu.upload(image)
#     # Create Gaussian filter
#     gaussian_filter = cv2.cuda.createGaussianFilter(
#         srcType=cv2.CV_8UC3,
#         dstType=cv2.CV_8UC3,
#         ksize=kernel_size,
#         sigma1=0
#     )
#     # Apply blur
#     blurred_gpu = gaussian_filter.apply(image_gpu)
#
#     # Convert to CuPy arrays
#     image_cupy = gpumat_to_cupy(image_gpu)
#     blurred_cupy = gpumat_to_cupy(blurred_gpu)
# #      Blend foreground and blurred background
#     result_gpu = alpha * image_cupy.astype(cp.float32) + \
#                  (1 - alpha) * blurred_cupy.astype(cp.float32)
#     result_gpu = cp.clip(result_gpu, 0, 255)
#     result_gpu = result_gpu.astype(cp.uint8)
#     result = cp.asnumpy(result_gpu)
#
#     return result

def blur_background(image, alpha, kernel_size=(31, 31)):
    """
    Apply Gaussian blur to the background using PyTorch.

    Args:
        image (torch.Tensor): Input image, shape (1, 3, height, width), uint8.
        alpha (torch.Tensor): Mask, shape (1, 3, height, width), float32.
        kernel_size (tuple): Gaussian kernel size, default (31, 31).

    Returns:
        np.ndarray: Image with blurred background, shape (height, width, 3), uint8.
    """
    # Apply Gaussian blur to image
    blurred_tensor = F.gaussian_blur(image, kernel_size=kernel_size, sigma=[10.0, 10.0])
    # Convert to float32 for blending
    image_float = image.to(torch.float32)
    blurred_float = blurred_tensor.to(torch.float32)
    # Blend foreground and blurred background
    result_gpu = alpha * image_float + (1 - alpha) * blurred_float
    result_gpu = torch.clip(result_gpu, 0, 255)
    result_gpu = result_gpu.to(torch.uint8)
    # Convert to NumPy array (HWC format)
    result = result_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return result
