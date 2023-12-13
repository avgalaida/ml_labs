from torch.utils.data import random_split, DataLoader


def create_loaders(dataset, proportions, batch_size):
    train_set, val_set, test_set = random_split(dataset, proportions)

    print(f'train: {len(train_set)}' + '\n' + f'valid: {len(val_set)}' + '\n' + f'test: {len(test_set)}')

    loaders = {
        'train': DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=True),
        'val':   DataLoader(val_set,   shuffle=True, batch_size=batch_size),
        'test':  DataLoader(test_set,  shuffle=True, batch_size=batch_size)
    }

    return loaders
