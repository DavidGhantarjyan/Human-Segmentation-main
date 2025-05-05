import cv2
import numpy as np
import onnxruntime as ort
import time
import onnx
import torch
import cupy as cp

from other.parsing.train_args_parser import real_resize_height, real_resize_width
from evaluation_utils import (
    remove_letterbox_padding_from_pad_info as remove_letterbox_padding,
    preprocess_frame_gpu,
    select_main_person,
    blur_background
)

ONNX_PATH = 'train_results/UNet_MobileNet/UNet_MobileNet_fp16.onnx'

INPUT_SIZE = (180, 320)
THRESHOLD = 0.3
BLUR_KERNEL = (31, 31)

providers = [
    # ('TensorrtExecutionProvider', {
    #     'device_id': 0,
    #     'trt_fp16_enable': True,
    #     # 'trt_int8_enable': True,
    #     'trt_engine_cache_enable': True,
    #     'trt_engine_cache_path': './trt_cache',
    # }),
    'CUDAExecutionProvider',
]

so = ort.SessionOptions()
# so.log_severity_level = 0
session = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=providers)
input_name = session.get_inputs()[0].name

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {orig_w}x{orig_h}")

# def apply_guided_filter(I_g, I_l_up, radius=4, eps=1e-6):
#     I_g = I_g.astype(np.float32) / 255.0
#     I_l_up = I_l_up.astype(np.float32)
#
#     if I_g.ndim == 3:
#         I_g = cv2.cvtColor(I_g, cv2.COLOR_BGR2GRAY)
#
#     q = cv2.ximgproc.guidedFilter(guide=I_g, src=I_l_up, radius=radius, eps=eps)
#     return np.clip(q, 0, 1)

fps = 0
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture a frame from the camera.")
        break

    input_tensor, pad_info = preprocess_frame_gpu(frame, real_resize_width, real_resize_height)
    output_shape = (1, 1, real_resize_width, real_resize_height)
    output_buffer = cp.zeros(output_shape, dtype=cp.float16)

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

    session.run_with_iobinding(io_binding)
    output = output_buffer  # CuPy float16
    output = 1 / (1 + cp.exp(-output))
    # CuPy uint8
    binary_output = (output > THRESHOLD).astype(cp.uint8).squeeze() * 255

    # CuPy uint8
    mask_cleaned = remove_letterbox_padding(binary_output, pad_info, (orig_h, orig_w))


    # mask_cleaned = select_main_person(mask_cleaned.get(), center_weight=0.5)
    # binary_display = blur_background(frame, mask_cleaned)
    # up_scaled_mask_guided = apply_guided_filter(frame, main_person_mask, radius=3, eps=1e-6)

    binary_display = blur_background(frame, mask_cleaned.get())

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    cv2.putText(binary_display, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Original Video", frame)
    cv2.imshow("Model Output (Thresholded)", binary_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

