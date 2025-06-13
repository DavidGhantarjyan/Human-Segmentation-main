import cv2
import numpy as np
import onnxruntime as ort
import time
import onnx
import torch
import cupy as cp
import torch.utils.dlpack

# Import hyperparameters and utility functions
from other.parsing.train_args_parser import real_resize_height, real_resize_width
from evaluation_utils import (
    remove_letterbox_padding_from_pad_info as remove_letterbox_padding,
    preprocess_frame_gpu,
    select_main_person,
    blur_background
)
from other.models.guided_filter import GuidedFilter

# Define constants for ONNX model and processing
ONNX_PATH = 'train_results/UNet_MobileNetv3/UNet_MobileNetv3_fp16.onnx'  # Path to ONNX model
INPUT_SIZE = (180, 320)  # Input size for model (height, width)
THRESHOLD = 0.5  # Threshold for binarizing model output
BLUR_KERNEL = (31, 31)  # Kernel size for background blur

providers = [
    # ('TensorrtExecutionProvider', {  # Commented-out TensorRT provider
    #     'device_id': 0,
    #     'trt_fp16_enable': True,
    #     # 'trt_int8_enable': True,
    #     'trt_engine_cache_enable': True,
    #     'trt_engine_cache_path': './trt_cache',
    # }),
    'CUDAExecutionProvider',
]

# Initialize ONNX runtime session
so = ort.SessionOptions()
# so.log_severity_level = 0  # Optionally set logging level (commented out)
session = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=providers)
input_name = session.get_inputs()[0].name

# Initialize webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {orig_w}x{orig_h}")

# Initialize FPS counter
fps = 0
while True:
    # Record start time for FPS calculation
    start_time = time.time()

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture a frame from the camera.")
        break

    # -------------------------------
    # Preprocess Frame and Run Inference
    # -------------------------------
    # Preprocess frame on GPU (resize, normalize, convert to tensor)
    input_tensor, pad_info = preprocess_frame_gpu(frame, real_resize_width, real_resize_height)
    output_shape = (1, 1, real_resize_width, real_resize_height)  # Expected output shape
    output_buffer = cp.zeros(output_shape, dtype=cp.float16)  # Initialize CuPy output buffer

    # Set up IO binding for ONNX inference on GPU
    io_binding = session.io_binding()
    device_ptr = input_tensor.data.ptr
    io_binding.bind_input(
        name=input_name,
        device_type="cuda",
        device_id=0,
        element_type=onnx.TensorProto.FLOAT16,
        shape=input_tensor.shape,
        buffer_ptr=device_ptr
    )

    io_binding.bind_output(
        name="output",
        device_type="cuda",
        device_id=0,
        element_type=onnx.TensorProto.FLOAT16,
        shape=output_shape,
        buffer_ptr=output_buffer.data.ptr
    )

    # Run ONNX model inference
    session.run_with_iobinding(io_binding)
    output = output_buffer  # CuPy float16 output
    output = 1 / (1 + cp.exp(-output))  # Apply sigmoid to get probabilities
    # CuPy uint8
    binary_output = (output > THRESHOLD).astype(cp.uint8).squeeze() * 255  # Binarize and scale to 0–255

    # CuPy uint8
    # -------------------------------
    # Postprocess Mask
    # -------------------------------
    # Remove letterbox padding and resize mask to original frame size
    mask_cleaned = remove_letterbox_padding(binary_output, pad_info, (orig_h, orig_w))
    alpha = cp.clip(cp.asarray(mask_cleaned[..., None], dtype=cp.float32), 0, 1)  # Convert to float32 alpha mask

    # Convert frame and mask to PyTorch tensors for further processing
    frame_torch_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to('cuda')
    mask_cleaned_torch_tensor = torch.from_dlpack(alpha.__dlpack__()).permute(2, 0, 1).unsqueeze(0).repeat(1, 3, 1, 1)

    # Apply GuidedFilter to refine the mask
    up_scaled_mask_guided = torch.clip(GuidedFilter(r=5, eps=1e-6)(frame_torch_tensor,
                                                                   mask_cleaned_torch_tensor), 0.0, 1.0)

    # -------------------------------
    # Generate Display Image
    # -------------------------------
    # Apply background blur using the refined mask
    binary_display = blur_background(frame_torch_tensor, up_scaled_mask_guided)
    binary_display = np.ascontiguousarray(binary_display)

    # mask_cleaned = select_main_person(mask_cleaned.get(), center_weight=0.5)
    # binary_display = blur_background(frame, mask_cleaned)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Add FPS text to display image
    cv2.putText(binary_display, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text for color image

    # Display original frame and processed output
    cv2.imshow("Original Video", frame)
    cv2.imshow("Model Output (Thresholded)", binary_display)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
