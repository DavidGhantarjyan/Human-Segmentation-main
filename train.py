import torch
import time
import shutil
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

# Data processing and dataset creation
from other.data.processing import get_train_val_dataloaders
from other.data.datasets import (CocoDataset, SyntheticDataset, MixedDataset, TripletTransform,
                                 base_transform, apply_custom_transform, input_to_tensor_transform,
                                 val_base_transform, val_input_to_tensor_transform)

# Model selection and utility functions for counting parameters and estimating VRAM usage
from other.models.models_handler import MODELS, count_parameters, estimate_vram_usage , get_flops

# Utility functions for file management, logging, and plotting training history
from other.utils import (find_model_in_dir_or_path, find_last_model_in_tree, print_as_table,
                         create_new_model_trains_dir, save_history_plot)

# Custom loss functions and image processing utilities
from other.losses import (BoundaryLossCalculator, BlurBoundaryLoss)
from other.losses_utils import ImageProcessor

# Profiling modules (optional, commented out for now)
import cProfile
import pyprof2calltree
from torch.profiler import profile, record_function, ProfilerActivity

# Automatic mixed precision (AMP) for FP16 training
from torch.amp import autocast, GradScaler
from segmentation_models_pytorch.losses import DiceLoss , FocalLoss
from monai.losses import GeneralizedDiceLoss, TverskyLoss

if __name__ == '__main__':
    # Load training arguments from parser. This includes all hyperparameters and directory paths.
    from other.parsing.train_args_parser import *

    # Print a separator to indicate the start of a training run in the console log.
    print(f"\n{'=' * 100}\n")
    # -------------------------------
    # Data Preparation
    # -------------------------------
    # Create a COCO-format dataset for training with a specified set of transformations.
    train_coco_dataset = CocoDataset(
        cocodataset_path=train_coco_data_dir,
        transform=TripletTransform(
            transform=apply_custom_transform,  # Custom augmentation function for background images.
            base_transform=base_transform,  # Base transformation applied to the input images.
            input_to_tensor_transform=input_to_tensor_transform  # Converts inputs to PyTorch tensors.
        )
    )
    # Initialize Mapillary dataset for training with the same transformations as COCO
    train_mapillary_dataset = CocoDataset(
        cocodataset_path=train_mapillary_data_dir,
        transform=TripletTransform(
            transform=apply_custom_transform,
            base_transform=base_transform,
            input_to_tensor_transform=input_to_tensor_transform
        )
    )

    # Create a synthetic dataset.
    synthetic_dataset = SyntheticDataset(
        length=None,  # Use default or full dataset length if None
        test_safe_to_desk=False,  # Do not persist generated data to disk
    )

    # Create a validation COCO dataset with validation-specific transformations.
    val_coco_dataset = CocoDataset(
        cocodataset_path=val_coco_data_dir,
        transform=TripletTransform(
            transform=None,  # No extra augmentation on validation images.
            base_transform=val_base_transform,  # Base validation transformation.
            input_to_tensor_transform=val_input_to_tensor_transform  # Convert validation inputs to tensors.
        )
    )
    combined_dataset = ConcatDataset([train_coco_dataset, train_mapillary_dataset])

    # Combine the training COCO dataset and synthetic dataset with a scale factor to form a mixed dataset.
    mixed_dataset = MixedDataset(combined_dataset, synthetic_dataset, scale_factor=scale_factor)
    print(len(mixed_dataset))

    # -------------------------------
    # Model Setup
    # -------------------------------
    # Select the device for computation (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create an instance of the chosen model and move it to the selected device.
    model = MODELS[model_name]().to(device)

    # Optionally compile the model (PyTorch 2.0+) for performance improvements.
    # model = torch.compile(model)
    # Create an optimizer (Adam) with the learning rate specified by the configuration.
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if scheduler_available:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=8, threshold=0.001, min_lr=1e-6,
        )

    # Determine the directory structure for saving training results. The model's name is used for folder organization.
    model_trains_tree_dir = os.path.join(train_res_dir, model_name)

    # Initialize model directory and checkpoint variables.
    model_dir, model_path = None, None
    last_weights_path = None

    # -------------------------------
    # Checkpoint Handling
    # -------------------------------
    # If a path for pre-trained weights is provided, attempt to load from that location.
    if weights_load_from is not None:
        # Example: train_results/model_name/2025-02-19_12-00-00/res_10/weights.pt
        # Find the first available weights file in the provided path.
        last_weights_path = find_model_in_dir_or_path(weights_load_from)
    else:
        # If not creating a new model, try to locate the last saved checkpoint.
        if (create_new_model is None) or (not create_new_model):
            model_dir, model_path = find_last_model_in_tree(model_trains_tree_dir)
            last_weights_path = model_path
            if model_path is None:
                print(f"Couldn't find any model in {model_trains_tree_dir} so a new model will be created")

    # Initialize dataloaders and seeds.
    train_dataloader, val_dataloader, train_seed, validation_sed = [None] * 4
    if last_weights_path is not None:
        # Load checkpoint state.
        checkpoint = torch.load(last_weights_path, weights_only=True)
        train_seed = checkpoint["train_seed"]
        validation_seed = checkpoint["validation_seed"]
        # Retrieve dataloaders with fixed seeds for reproducibility.
        train_dataloader, val_dataloader, _, _ = get_train_val_dataloaders(
            train_dataset=mixed_dataset,
            val_dataset=val_coco_dataset,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            val_num_workers=val_num_workers,
            train_seed=train_seed,
            val_seed=validation_seed
        )
        # Load saved model and optimizer state.
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if enable_lr_reset:
            for param_group in optimizer.param_groups:
                old_param_lr = param_group['lr']
                param_group['lr'] = lr
                print(f"Learning rate {old_param_lr} has been reset to: {param_group['lr']}")
        for idx, param_group in enumerate(optimizer.param_groups):
            print(
                f"Parameter group {idx} - Current learning rate: {param_group['lr']}")
        if 'scheduler_state_dict' in checkpoint and scheduler_available:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state has been successfully loaded.")

        # Set the current epoch to resume training.
        curr_run_start_global_epoch = checkpoint['epoch'] + 1

        print(f"Successfully loaded {last_weights_path} with optimizer {checkpoint['optimizer']}")
        print(f"Continuing training from epoch {curr_run_start_global_epoch} on {device} device")
    else:
        # No checkpoint found; start training from scratch.
        curr_run_start_global_epoch = 1
        print(f"New model of {str(type(model))} type has been created, will be trained on {device} device")

        # Print model parameter count and estimated VRAM usage for debugging and monitoring.
    print(f"Estimated [parameters: {count_parameters(model)}, vram: {estimate_vram_usage(model):.4f} GB (if float32)]")
    try:
        gflops = get_flops(model, device=device)
        print(f"Estimated FLOPs: {gflops:.2f} GFLOPs")
    except Exception as e:
        print(f"FLOPs estimation failed: {e}")
    # -------------------------------
    # History Logging Initialization
    # -------------------------------
    try:
        # Try loading previous training history if available.
        loss_history_table = pd.read_csv(os.path.join(model_dir, 'loss_history.csv'), index_col="global_epoch")
        accuracy_history_table = pd.read_csv(os.path.join(model_dir, 'accuracy_history.csv'),
                                             index_col="global_epoch")
    except:
        # If not found, create new DataFrames for logging losses and accuracies.
        train_dataloader, val_dataloader, train_seed, validation_seed = get_train_val_dataloaders(
            train_dataset=mixed_dataset, val_dataset=val_coco_dataset, batch_size=batch_size,
            val_batch_size=val_batch_size, num_workers=num_workers, val_num_workers=val_num_workers)
        loss_history_table = pd.DataFrame(columns=['global_epoch', 'train_loss', 'val_loss'])
        accuracy_history_table = pd.DataFrame(columns=['global_epoch', 'train_accuracy', 'val_accuracy',
                                                       'train_precision', 'val_precision', 'train_recall', 'val_recall',
                                                       'train_f1', 'val_f1', 'train_iou', 'val_iou',
                                                       ])
        loss_history_table.set_index('global_epoch', inplace=True)
        accuracy_history_table.set_index('global_epoch', inplace=True)

        print(f"Checkpoints(for this run): {save_frames}")

    # # -------------------------------
    # # Optional: Setup Profiler (commented out)
    # # -------------------------------
    # profiler_iterations = 30
    # train_profiler = profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(
    #         wait=1,
    #         warmup=1,
    #         active=profiler_iterations,
    #         repeat=1
    #     ),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         r'C:\Users\Admin\PycharmProjects\PythonProject\Human-Segmentation-main\logs'
    #     ),
    #     record_shapes=True,
    #     with_stack=True,
    #     profile_memory=True,
    # )

    # Create a GradScaler for automatic mixed precision (AMP) training on CUDA.
    scaler = GradScaler(device='cuda')

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(1, do_epoches + 1):
        global_epoch = curr_run_start_global_epoch + epoch - 1

        w_base = torch.tensor(1 - ((global_epoch - 1) - 82) / 100, dtype=torch.float16, device=device)
        w_boundary = torch.tensor(((global_epoch - 1) - 82), dtype=torch.float16, device=device)

        # Set model to training mode.
        model.train()
        # Initialize running metrics as scalar tensors on the selected device.
        running_loss = torch.scalar_tensor(0.0, device=device)
        running_correct_count = torch.scalar_tensor(0.0, device=device)
        running_whole_count = torch.scalar_tensor(0.0, device=device)
        running_tp = torch.scalar_tensor(0.0, device=device)
        running_fp = torch.scalar_tensor(0.0, device=device)
        running_fn = torch.scalar_tensor(0.0    , device=device)

        # with train_profiler:
        # Iterate over training batches.
        for batch_idx, (batch_inputs, batch_targets, mask) in enumerate(
                tqdm(train_dataloader, desc=f"Training epoch: {global_epoch} ({epoch}\\{do_epoches}) | ")):
            # -------------------------------\n
            # Data Transfer: CPU to GPU\n   
            # -------------------------------
            with record_function("## CPU Data Transfer ##"):
                batch_inputs = batch_inputs.to(device, non_blocking=True, memory_format=torch.channels_last).detach()
                batch_targets = batch_targets.to(device, non_blocking=True).detach()
                mask = mask.to(device, non_blocking=True).detach()

            # -------------------------------\n
            # Forward Pass & Loss Computation\n
            # -------------------------------
            with (((autocast(device_type='cuda')))):
                with record_function("## GPU Forward Pass ##"):
                    out = model(batch_inputs)
                    out = out.contiguous(memory_format=torch.contiguous_format)
                    # Calculate the comp    osite loss from three components:
                    #   1. Boundary Loss (weighted by alpha)
                    #   2. Blur Boundary Loss (weighted by beta)
                    #   3. Binary Cross-Entropy
                    #   Loss (weighted by gamma)

                    with record_function("## Loss Calculation ##"):

                        loss =(
                                (alpha * BoundaryLossCalculator(device=device)(out, batch_targets, mask).mean()  * w_boundary)+
                                (
                                # beta * BlurBoundaryLoss()(out, batch_targets) +
                                gamma * F.binary_cross_entropy_with_logits(out, batch_targets.unsqueeze(1),
                                                                           reduction='mean') +
                                # delta * DiceLoss(mode='binary', log_loss=False)(F.sigmoid(out), batch_targets)
                                # eta * FocalLoss(mode='binary',alpha=0.75, gamma=1)(F.sigmoid(out), batch_targets)
                                # sigma * GeneralizedDiceLoss(sigmoid=True)(out, batch_targets.unsqueeze(1))
                                epsilon * TverskyLoss(alpha=0.3, beta=0.7, sigmoid=True,smooth_nr=1, smooth_dr=1)(out, batch_targets.unsqueeze(1))
                                ) * w_base
                        )

                    # Normalize loss by the number of accumulation steps.
                    loss = loss / accumulation_steps

                    # -------------------------------\n
                    # Update Running Metrics\n
                    # -------------------------------
                    batch_samples_count = mask.size(0)
                    running_loss += loss.detach() * batch_samples_count * accumulation_steps
                    running_whole_count += batch_samples_count
                    # Binarize the output using a predefined threshold.
                    binary_out = ImageProcessor.binarize_array(torch.sigmoid(out).detach(), threshold=threshold)
                    binary_target = batch_targets.unsqueeze(1).detach()

                    pred_correct = ((binary_out == binary_target)).int()
                    running_correct_count += torch.sum(
                        pred_correct).detach() * batch_samples_count / pred_correct.numel()

                    running_tp += (((binary_out == 1) & (binary_target == 1)).sum()).detach()
                    running_fp += (((binary_out == 1) & (binary_target == 0)).sum()).detach()
                    running_fn += (((binary_out == 0) & (binary_target == 1)).sum()).detach()

            # -------------------------------\n
            # Backward Pass & Optimization\n
            # -------------------------------
            with record_function("## GPU Backward ##"):
                # Use GradScaler for FP16 backward pass.
                scaler.scale(loss).backward()
                # Perform an optimizer step every 'accumulation_steps' batches.
                if (batch_idx + 1) % accumulation_steps == 0:
                    with record_function("## GPU Optim Step ##"):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        # Detach output and loss to prevent memory leaks.
                        out = out.detach()
                        loss = loss.detach()
            # Uncomment the following lines for profiling details:
            # train_profiler.step()
            # print("TOP-10 time-consuming operations CUDA:", train_profiler.key_averages().table(sort_by='cuda_time_total', row_limit=10))
            # print("\nTOP-10 time-consuming operations CPU:", train_profiler.key_averages().table(sort_by='cpu_time_total', row_limit=10))
        # -------------------------------\n
        # End of Training Epoch: Calculate and Log Metrics\n
        # -------------------------------
        with torch.no_grad():
            train_loss = (running_loss / running_whole_count).item()
            train_accuracy = (running_correct_count / running_whole_count).item()
            train_precision = (running_tp / (running_tp + running_fp)).item() if (running_tp + running_fp) > 0 else 0
            train_recall = (running_tp / (running_tp + running_fn)).item() if (running_tp + running_fn) > 0 else 0
            train_f1 = (2 * train_precision * train_recall) / (train_precision + train_recall) if (
                    train_precision + train_recall) else 0
            train_iou = (running_tp / (running_tp + running_fp + running_fn)).item() if (
                                                                                                running_tp + running_fp + running_fn) > 0 else 0

            # Prepare a dictionary to store training metrics for logging.
            row_loss_values = {
                'global_epoch': global_epoch,
                'train_loss': train_loss
            }
            row_acc_values = {
                'global_epoch': global_epoch,
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'train_iou': train_iou,
            }
            if print_level > 0:
                time.sleep(0.25)
                print(
                    f"Training | loss: {train_loss:.4f} | train_accuracy: {train_accuracy:.4f} | train_precision: {train_precision:.4f} |"
                    f"train_recall: {train_recall:.4f} | train_f1: {train_f1:.4f} | train_iou: {train_iou:.4f}")
                time.sleep(0.25)

            # -------------------------------\n
            # Validation Phase\n
            # -------------------------------
            val_loss, val_acc = None, None
            if val_every != 0 and epoch % val_every == 0:
                model.eval()
                val_running_loss = torch.scalar_tensor(0.0, device=device)
                val_correct_count = torch.scalar_tensor(0.0, device=device)
                val_whole_count = torch.scalar_tensor(0.0, device=device)
                val_running_tp = torch.scalar_tensor(0.0, device=device)
                val_running_fp = torch.scalar_tensor(0.0, device=device)
                val_running_fn = torch.scalar_tensor(0.0, device=device)
                print()
                time.sleep(0.25)
                # Iterate over validation batches.
                for (batch_inputs, batch_targets, mask) in tqdm(val_dataloader,
                                                                desc=f"Calculating validation scores: "):
                    batch_inputs = batch_inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                    batch_targets = batch_targets.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    val_samples_count = mask.size(0)
                    with autocast(device_type='cuda'):
                        out = model(batch_inputs)
                        out = out.contiguous(memory_format=torch.contiguous_format)
                        loss =(
                                (alpha * BoundaryLossCalculator(device=device)(out, batch_targets, mask).mean() * w_boundary)+
                                (
                                # beta * BlurBoundaryLoss()(out, batch_targets) +
                                gamma * F.binary_cross_entropy_with_logits(out, batch_targets.unsqueeze(1),
                                                                           reduction='mean') +
                                # delta * DiceLoss(mode='binary', log_loss=False)(F.sigmoid(out), batch_targets)
                                # eta * FocalLoss(mode='binary',alpha=0.75, gamma=1)(F.sigmoid(out), batch_targets)
                                # sigma * GeneralizedDiceLoss(sigmoid=True)(out, batch_targets.unsqueeze(1))
                                epsilon * TverskyLoss(alpha=0.3, beta=0.7, sigmoid=True,smooth_nr=1, smooth_dr=1)(out, batch_targets.unsqueeze(1))
                                ) * w_base
                        )
                    val_running_loss += loss.detach() * val_samples_count
                    val_whole_count += val_samples_count

                    binary_out = ImageProcessor.binarize_array(torch.sigmoid(out).detach(), threshold=threshold)
                    binary_target = batch_targets.unsqueeze(1).detach()
                    pred_correct = (binary_out == binary_target).int()
                    val_correct_count += torch.sum(pred_correct).detach() * val_samples_count / pred_correct.numel()

                    val_running_tp += (((binary_out == 1) & (binary_target == 1)).sum()).detach()
                    val_running_fp += (((binary_out == 1) & (binary_target == 0)).sum()).detach()
                    val_running_fn += (((binary_out == 0) & (binary_target == 1)).sum()).detach()

                # Calculate validation loss and accuracy.
                val_loss = (val_running_loss / val_whole_count).item()
                if scheduler:
                    scheduler.step(val_loss)
                for idx, param_group in enumerate(optimizer.param_groups):
                    print(f"Parameter Group {idx}: Updated learning rate by scheduler = {param_group['lr']}")

                val_accuracy = (val_correct_count / val_whole_count).item()
                val_precision = (val_running_tp / (val_running_fp + val_running_tp)).item() if (
                                                                                                       val_running_fp + val_running_tp) > 0 else 0
                val_recall = (val_running_tp / (val_running_tp + val_running_fn)).item() if (
                                                                                                    val_running_tp + val_running_fn) > 0 else 0
                val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall) if (
                                                                                                    val_precision + val_recall) > 0 else 0
                val_iou = (val_running_tp / (val_running_tp + val_running_fp + val_running_fn)).item() if (
                                                                                                                  val_running_tp + val_running_fp + val_running_fn) > 0 else 0

                # Append validation metrics to the logging dictionaries.
                row_loss_values['val_loss'] = val_loss if val_loss is not None else float('nan')
                row_acc_values['val_accuracy'] = val_accuracy if val_accuracy is not None else float('nan')
                row_acc_values['val_precision'] = val_precision if val_precision is not None else float('nan')
                row_acc_values['val_recall'] = val_recall if val_recall is not None else float('nan')
                row_acc_values['val_f1'] = val_f1 if val_f1 is not None else float('nan')
                row_acc_values['val_iou'] = val_iou if val_iou is not None else float('nan')

                # Log the metrics for the current global epoch.
                loss_history_table.loc[global_epoch] = row_loss_values
                accuracy_history_table.loc[global_epoch] = row_acc_values

            # Optionally print full history if verbose level is high.
            if val_every != 0 and print_level > 1 and epoch % val_every == 0:
                print(f"\nLoss history")
                print_as_table(loss_history_table)
                print(f"\nAccuracy history")
                print_as_table(accuracy_history_table)

            # -------------------------------\n
            # Checkpoint Saving\n
            # -------------------------------
            if epoch in save_frames:
                # Create a new directory for saving the current checkpoint if it doesn't exist.
                if model_dir is None:
                    model_dir, model_path = create_new_model_trains_dir(model_trains_tree_dir)
                    print(f"\nCreated {model_dir}")
                # Backup previous weights if applicable.
                if last_weights_path is not None:
                    old = os.path.join(model_dir, "old")
                    os.makedirs(old, exist_ok=True)
                    new_weights_path = os.path.join(old, f"weights{global_epoch}.pt")
                    shutil.copy(last_weights_path, new_weights_path)

            # Save loss and accuracy history to CSV files.
            loss_history_table.to_csv(os.path.join(model_dir, 'loss_history.csv'))
            accuracy_history_table.to_csv(os.path.join(model_dir, 'accuracy_history.csv'))

            # Optionally save plots of training history.
            if plot:
                save_history_plot(loss_history_table, 'global_epoch', 'Loss history', 'Epoch', 'Loss',
                                  os.path.join(model_dir, 'loss.png'))
                save_history_plot(accuracy_history_table, 'global_epoch', 'Accuracy history', 'Epoch', 'Accuracy',
                                  os.path.join(model_dir, 'accuracy.png'))

            # -------------------------------\n
            # Save Current Model Checkpoint\n
            # -------------------------------
            checkpoint = {
                'train_seed': train_seed,
                'validation_seed': validation_seed,
                'epoch': global_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': type(optimizer).__name__,
                'optimizer_state_dict': optimizer.state_dict()
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, model_path)
            print(f"\nModel saved (global epoch: {global_epoch}, checkpoint: {epoch})")

            # Update last weights path for the next iteration.
            last_weights_path = model_path
            # Trigger any additional actions post-model save (e.g., notifications).
            model_has_been_saved()
