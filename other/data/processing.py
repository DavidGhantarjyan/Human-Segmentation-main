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
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    # train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    val_dataloader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, num_workers=val_num_workers,pin_memory=True)

    return train_dataloader, val_dataloader, train_seed, val_seed




