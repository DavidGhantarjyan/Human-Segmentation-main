import os
import torch
import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.amp import autocast

from other.models.models_handler import MODELS
from other.utils import find_last_model_in_tree
from other.data.datasets import val_input_to_tensor_transform, EvalDataset
from other.losses_utils import ImageProcessor
from other.parsing.train_args_parser import real_resize_width,real_resize_height


from evaluation_utils import remove_letterbox_padding

# Define paths and model name
test_dir = r"./CocoData/model_test/"
model_name = r'UNet_MobileNet'
train_res_dir = "train_results"

if __name__ == '__main__':


    # Find the latest model checkpoint in the training tree
    model_trains_tree_dir = os.path.join(train_res_dir, model_name)
    model_new_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)
    if model_path is None:
        raise Exception(f"No model was found at {model_trains_tree_dir}")

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the model checkpoint (weights only)
    checkpoint = torch.load(model_path, weights_only=True)
    model = MODELS[model_name]().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    # Create evaluation dataset and DataLoader
    eval_dataset = EvalDataset(test_dir, transform=val_input_to_tensor_transform)
    print(f"{len(eval_dataset)} files will be evaluated")
    if len(eval_dataset) == 0:
        raise ValueError("No files were found")

    # Create output directory for saving predictions
    output_dir = os.path.join(test_dir, 'target', model_name)
    os.makedirs(output_dir, exist_ok=True)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
    )

    out_height = real_resize_height
    out_width = real_resize_width

    # Perform inference without gradient computation
    with torch.no_grad():
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
            out_bin = ImageProcessor.binarize_array(out, threshold=0.5) * 255.0
            out_bin = out_bin.byte()
            heights = original_sizes[0]
            widths = original_sizes[1]
            sizes = list(zip(heights.tolist(), widths.tolist()))


            for i, (file_name, (original_h, original_w)) in enumerate(zip(file_names, sizes)):
                output_np = out_bin[i].squeeze(0).squeeze(0).cpu().numpy()
                output_full  = cv2.resize(output_np, (out_width * 2, out_height * 2), interpolation=cv2.INTER_NEAREST)
                output_resized = remove_letterbox_padding(output_full, original_h, original_w, (out_height * 2, out_width * 2))

                file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
                cv2.imwrite(file_path, output_resized)
