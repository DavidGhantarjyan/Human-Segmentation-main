import os
import torch
from other.models.models_handler import MODELS
from other.utils import find_last_model_in_tree
import onnx
from collections import Counter
from onnxsim import simplify
from onnxconverter_common import float16

"""
Script to convert a PyTorch UNet_MobileNet model to ONNX, simplify it, quantize to FP16,
validate the model, and print input/output and node statistics.

Saves original, simplified, and FP16 ONNX models to the script directory.
"""

# Get script directory and define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
train_res_dir = os.path.join(script_dir, '../../train_results')
model_name = r'UNet_MobileNetv3'
onnx_filename = model_name + '_original.onnx'
onnx_simplified_filename = model_name + '_simplified.onnx'
onnx_quantized_filename = model_name + '_fp16.onnx'
optimized_model_path = model_name +  '_optimized.onnx'

# Find the latest model checkpoint
model_trains_tree_dir = os.path.join(train_res_dir, model_name)
_, model_path = find_last_model_in_tree(model_trains_tree_dir)
if model_path is None:
    raise Exception(f"No model was found at {model_trains_tree_dir}")
print(f"Model was found at {model_path}")

# Load and prepare the model
try:
    model = MODELS[model_name]()
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Convert model to FP16 for quantization
    model.half()
except Exception as e:
    raise RuntimeError(f"Failed to load or prepare model: {e}")

# Create FP16 dummy input for ONNX export
dummy_input = torch.randn(1, 3, 320, 180, dtype=torch.float16)

# Export model to ONNX
onnx_path = os.path.join(script_dir, onnx_filename)
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"ONNX model saved at: {onnx_path}")
except Exception as e:
    raise RuntimeError(f"ONNX export failed: {e}")

# Simplify ONNX model
print("Simplifying ONNX model...")
try:
    model_onnx = onnx.load(onnx_path)
    model_simp, check = simplify(model_onnx)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
    onnx_simplified_path = os.path.join(script_dir, onnx_simplified_filename)
    onnx.save(model_simp, onnx_simplified_path)
    print(f"Simplified ONNX saved at: {onnx_simplified_path}")
except Exception as e:
    raise RuntimeError(f"ONNX simplification failed: {e}")

# Quantize to FP16
try:
    model = onnx.load(onnx_simplified_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, onnx_quantized_filename)
    # torch.onnx.dynamo_export(model, dummy_input,save_path, opset_version=12)
except Exception as e:
    raise RuntimeError(f"FP16 quantization failed: {e}")

# Validate final model
try:
    model_path = onnx_quantized_filename
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("✅ ONNX model is valid")
except Exception as e:
    raise RuntimeError(f"ONNX model validation failed: {e}")

# Print input details
print("\nInputs:")
for inp in model.graph.input:
    tp = inp.type.tensor_type
    shape = [d.dim_value if (d.dim_value>0) else '?' for d in tp.shape.dim]
    print(f"  • {inp.name} : {tp.elem_type=} , shape={shape}")

# Print output details
print("\nOutputs:")
for out in model.graph.output:
    tp = out.type.tensor_type
    shape = [d.dim_value if (d.dim_value>0) else '?' for d in tp.shape.dim]
    print(f"  • {out.name} : {tp.elem_type=} , shape={shape}")

# Print top-10 most common node types
node_types = [n.op_type for n in model.graph.node]
counts = Counter(node_types)
print("\nTop-10 most common node types:")
for op, cnt in counts.most_common(10):
    print(f"  {op}: {cnt}")

# Print initializer data types
init_types = [init.data_type for init in model.graph.initializer]
cnt_init = Counter(init_types)
print("\nInitializer data types:")
for dt, cnt in cnt_init.items():
    # ONNX elem_type: 10=float16, 1=float32, etc.
    print(f"  elem_type={dt} : count={cnt}")
