import torch
from torch.utils.data import DataLoader, Subset

from other.data.datasets  import *


def get_train_val_dataloaders(train_dataset, val_dataset, batch_size, val_batch_size, num_workers,
                              val_num_workers, train_seed=None, val_seed=None):
    if train_seed is None:
        train_seed = torch.randint(low=0, high=2 ** 32, size=(1,)).item()
    generator1 = torch.Generator()
    generator1.manual_seed(train_seed)

    if val_seed is None:
        val_seed = torch.randint(low=0, high=2 ** 32, size=(1,)).item()
    generator2 = torch.Generator()
    generator2.manual_seed(val_seed)

    train_indices = torch.randperm(len(train_dataset), generator=generator1)
    train_subset = Subset(train_dataset, train_indices)
    val_indices = torch.randperm(len(val_dataset), generator=generator2)
    val_subset = Subset(val_dataset, val_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers)

    return train_dataloader, val_dataloader, train_seed, val_seed



#
#
# if __name__ == '__main__':
#     ###############################################################################
#     # Dataset Initialization and Visualization
#     ###############################################################################
#     synthetic_dataset = SyntheticDataset(
#         test_safe_to_desk=False,
#         target_transform=base_transform,
#         input_transform=input_transform_synthetic
#     )
#
#     coco_dataset = CocoDataset(
#         cocodataset_path=train_coco_data_dir,
#         transform=TripletTransform(
#             input_transform=input_transform,
#             target_transform=target_transform,
#             mask_transform=mask_transform
#         ),
#         # pre_computing=natural_data_mask_saving
#     )
#
#     val_dataset = CocoDataset(
#         cocodataset_path=val_coco_data_dir,
#         transform=TripletTransform(
#             input_transform=base_transform,
#             target_transform=base_transform,
#             mask_transform=base_transform
#         ),
#         # pre_computing=natural_data_mask_saving
#     )
#     dataset = MixedDataset(coco_dataset, synthetic_dataset, scale_factor=1.5)
#
#
#     train_dataloader, val_dataloader, _,_ = get_train_val_dataloaders(train_dataset=dataset,val_dataset=val_dataset, batch_size=
#                                                                         batch_size,
#                                                                         val_batch_size = val_batch_size,
#                                                                         num_workers = num_workers, val_num_workers = val_num_workers,
#                                                                     )
#     for batch in (train_dataloader):
#         # torch.Size([12, 3, 640, 360])
#         # torch.Size([12, 640, 360])
#         # torch.Size([12, 640, 360])
#         inputs, targets,masks = batch
#         print(inputs.shape)
#         print(targets.shape)
#         print(masks.shape)