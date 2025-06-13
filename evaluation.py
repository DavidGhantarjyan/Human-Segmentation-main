import os
import torch
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.amp import autocast

# Import model, dataset, and utility functions
from other.models.models_handler import MODELS
from other.utils import find_last_model_in_tree
from other.data.datasets import val_input_to_tensor_transform, EvalDataset
from other.losses_utils import ImageProcessor
from other.parsing.train_args_parser import real_resize_width, real_resize_height

from evaluation_utils import remove_letterbox_padding

# Define paths and model name
test_dir = r"./CocoData/model_test/"  # Directory containing test images
model_name = r'UNet_MobileNetv3'  # Name of the model to evaluate
train_res_dir = "train_results"  # Directory containing training results and checkpoints

if __name__ == '__main__':
    # -------------------------------
    # Model Setup
    # -------------------------------
    # Find the latest model checkpoint in the training results directory
    model_trains_tree_dir = os.path.join(train_res_dir, model_name)
    model_new_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)
    if model_path is None:
        raise Exception(f"No model was found at {model_trains_tree_dir}")

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the model checkpoint (weights only)
    checkpoint = torch.load(model_path, weights_only=True)
    model = MODELS[model_name]().to(device)  # Initialize and move model to device
    model.load_state_dict(checkpoint['model_state_dict'])  # Load pre-trained weights
    model.eval()  # Set model to evaluation mode

    # -------------------------------
    # Dataset Setup
    # -------------------------------
    # Create evaluation dataset with validation transformations
    eval_dataset = EvalDataset(test_dir, transform=val_input_to_tensor_transform)
    print(f"{len(eval_dataset)} files will be evaluated")
    if len(eval_dataset) == 0:
        raise ValueError("No files were found")

    # Create output directory for saving predictions
    output_dir = os.path.join(test_dir, 'target', model_name)
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Initialize DataLoader for batch processing
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,  # Process 32 images per batch
        shuffle=False,  # Maintain order for consistent file naming
        num_workers=8,  # Use 8 workers for data loading
    )

    # Define output dimensions for resizing
    out_height = real_resize_height
    out_width = real_resize_width

    # -------------------------------
    # Inference Loop
    # -------------------------------
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_inputs, file_names, original_sizes in tqdm(eval_loader, desc="Evaluating"):
            # Move inputs to device and specify memory format for faster transfer
            batch_inputs = batch_inputs.to(device, memory_format=torch.channels_last)
            # Run model inference and apply sigmoid activation
            if device.type == 'cuda':
                # Use autocast with explicit device_type
                with autocast(device_type='cuda'):
                    out = model(batch_inputs)
                    out = torch.sigmoid(out)
            else:
                # On CPU or other devices, run normally
                out = model(batch_inputs)
                out = torch.sigmoid(out)
            # Binarize output with threshold and scale to [0, 255]
            out_bin = ImageProcessor.binarize_array(out, threshold=0.5) * 255.0
            out_bin = out_bin.byte()  # Convert to uint8
            # Extract original image sizes for resizing
            heights = original_sizes[0]
            widths = original_sizes[1]
            sizes = list(zip(heights.tolist(), widths.tolist()))

            # -------------------------------
            # Save Predictions
            # -------------------------------
            for i, (file_name, (original_h, original_w)) in enumerate(zip(file_names, sizes)):
                # Extract single mask from batch and convert to NumPy
                output_np = out_bin[i].squeeze(0).squeeze(0).cpu().numpy()

                # Resize mask to double the output size
                output_full = cv2.resize(output_np, (out_width * 2, out_height * 2), interpolation=cv2.INTER_NEAREST)

                # Remove letterbox padding to match original image dimensions
                output_resized = remove_letterbox_padding(output_full, original_h, original_w,
                                                          (out_height * 2, out_width * 2))

                # Save binary mask as PNG
                file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
                cv2.imwrite(file_path, output_resized)
