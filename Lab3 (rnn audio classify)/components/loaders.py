from torch.utils.data import random_split, DataLoader


def loaders(dataset, bs):
    train_set, val_set = random_split(dataset, [.8,.2])

    loaders = {
        'train': DataLoader(train_set, shuffle=True,  batch_size=bs, drop_last=True),
        'val':   DataLoader(val_set,   shuffle=False, batch_size=bs)
    }

    print(f'train: {len(train_set)}' + '\n' + f'val: {len(val_set)}')

    return loaders