
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import auc
from other.utils import find_model_in_dir_or_path
from torch.utils.data import DataLoader
from other.models.models_handler import MODELS
from other.data.datasets import CocoDataset, TripletTransform, val_base_transform, val_input_to_tensor_transform
from other.losses_utils import ImageProcessor
import pandas as pd
from torch.amp import autocast

if __name__ == '__main__':
    from other.parsing.metrix_args_parser import *

    metrix_dataset = CocoDataset(
        cocodataset_path=metrix_data_path,
        transform=TripletTransform(
            transform=None,  # No extra augmentation on validation images.
            base_transform=val_base_transform,  # Base validation transformation.
            input_to_tensor_transform=val_input_to_tensor_transform  # Convert validation inputs to tensors.
        )
    )
    dataloader = DataLoader(metrix_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[model_name]().to(device)
    model.eval()
    weights_path = find_model_in_dir_or_path(load_from)
    output_dir = os.path.dirname(weights_path)
    print(f"Output directory: {output_dir}")
    checkpoint = torch.load(weights_path, weights_only=True)
    # seed = checkpoint["seed"]
    model.load_state_dict(checkpoint['model_state_dict'])
    global_epoch = checkpoint['epoch'] + 1
    print(f"Successfully loaded {weights_path}")
    print(f"Metrix will be applied to {global_epoch} epoches trained model on {device} device")
    print(f"\n{'=' * 100}\n")

    whole_count = torch.scalar_tensor(0, device=device)
    num_thresholds = len(thresholds)
    tp_to_thold = torch.zeros(num_thresholds, dtype=torch.float32, device=device)
    fp_to_thold = torch.zeros(num_thresholds, dtype=torch.float32, device=device)
    fn_to_thold = torch.zeros(num_thresholds, dtype=torch.float32, device=device)
    tn_to_thold = torch.zeros(num_thresholds, dtype=torch.float32, device=device)
    # tp_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)
    # fn_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)
    # fp_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)
    # tn_to_thold = torch.tensor([0] * len(thresholds), dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch_inputs, batch_targets, _ in tqdm(dataloader):
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            if device.type == 'cuda':
                with autocast(device_type='cuda'):
                    out = model(batch_inputs)
                    out = torch.sigmoid(out)
            else:
                # On CPU or other devices, run normally
                out = model(batch_inputs)
                out = torch.sigmoid(out)

            # binary_outs = torch.stack([ImageProcessor.binarize_array(out.detach(), threshold=thold)
            #                            for thold in thresholds], dim=0)
            # binary_target = batch_targets.unsqueeze(1).detach()
            # binary_target = binary_target.expand(-1, num_thresholds, -1, -1)
            #
            # tp_to_thold += ((binary_outs == 1) & (binary_target == 1)).sum(dim=(1, 2, 3))
            # fp_to_thold += ((binary_outs == 1) & (binary_target == 0)).sum(dim=(1, 2, 3))
            # fn_to_thold += ((binary_outs == 0) & (binary_target == 1)).sum(dim=(1, 2, 3))
            # tn_to_thold += ((binary_outs == 0) & (binary_target == 0)).sum(dim=(1, 2, 3))

            for i, thold in enumerate(thresholds):
                binary_out = ImageProcessor.binarize_array(out.detach(), threshold=thold)
                binary_target = batch_targets.unsqueeze(1).detach()
                tp_to_thold[i] += ((binary_out == 1) & (binary_target == 1)).sum().item()
                fp_to_thold[i] += ((binary_out == 1) & (binary_target == 0)).sum().item()
                fn_to_thold[i] += ((binary_out == 0) & (binary_target == 1)).sum().item()
                tn_to_thold[i] += ((binary_out == 0) & (binary_target == 0)).sum().item()

        test_precision = tp_to_thold / (tp_to_thold + fp_to_thold)
        test_recall = tp_to_thold / (tp_to_thold + fn_to_thold)
        test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall)
        test_iou = tp_to_thold / (tp_to_thold + fp_to_thold + fn_to_thold)

        test_precision = torch.nan_to_num(test_precision, nan=1.0, posinf=1.0).cpu().numpy()
        test_recall = torch.nan_to_num(test_recall, nan=1.0, posinf=1.0).cpu().numpy()
        test_f1 = torch.nan_to_num(test_f1, nan=0.0, posinf=0.0).cpu().numpy()
        test_iou = torch.nan_to_num(test_iou, nan=0.0, posinf=0.0).cpu().numpy()


        roc_x = (fp_to_thold / (fp_to_thold + tn_to_thold))
        roc_y = (tp_to_thold / (tp_to_thold + fn_to_thold))

        roc_x = torch.nan_to_num(roc_x, nan=0.0, posinf=0.0).cpu().numpy()
        roc_y = torch.nan_to_num(roc_y, nan=0.0, posinf=0.0).cpu().numpy()

        roc_auc = auc(roc_x[::-1], roc_y[::-1])
        pr_auc = auc(test_recall, test_precision)

        best_th_by_f1 = thresholds[np.argmax(test_f1)]
        best_th_by_iou = thresholds[np.argmax(test_iou)]
        metrics_df = pd.DataFrame({
            'Threshold': thresholds,
            'F1': test_f1,
            'IoU': test_iou
        })
        metrics_df.set_index('Threshold', inplace=True)
        csv_path = os.path.join(output_dir, 'metrics.csv')
        metrics_df.to_csv(csv_path)
        print(f"Metrics saved to {csv_path}")

        plt.figure()
        plt.plot(roc_x[::-1], roc_y[::-1], color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.005])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_png_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_png_path)
        plt.close()
        print(f"ROC curve saved to {roc_png_path}")
        plt.figure()
        plt.plot(test_recall, test_precision, color='red', label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.005])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        pr_png_path = os.path.join(output_dir, 'pr_curve.png')
        plt.savefig(pr_png_path)
        plt.close()
        print(f"PR curve saved to {pr_png_path}")

        plt.figure()
        plt.plot(thresholds, test_f1, label='F1')
        plt.plot(thresholds, test_iou, label='IoU')
        plt.vlines(best_th_by_f1, 0, 1, color='gray', linestyle='--',
                   label=f'Best th: F1={best_th_by_f1:.2f}, IoU={best_th_by_iou:.2f}')
        plt.text(best_th_by_f1 + 0.02, np.max(test_f1), f'Max F1={np.max(test_f1):.2f}',
                 fontsize=8, color='gray')
        plt.text(best_th_by_iou + 0.02, np.max(test_iou), f'Max IoU={np.max(test_iou):.2f}',
                 fontsize=8, color='blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.005])
        plt.xlabel('Thresholds')
        plt.title('Metrics vs Thresholds')
        plt.legend(loc="lower right")
        metrics_png_path = os.path.join(output_dir, 'metrics_vs_thresholds.png')
        plt.savefig(metrics_png_path)
        plt.close()
        print(f"Metrics vs Thresholds saved to {metrics_png_path}")
