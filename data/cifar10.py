import torch
import numpy as np
from PIL import Image
import os

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform


def load_data(root, num_seen, batch_size, num_workers):
    """
    Load cifar10 dataset.

    Args
        root(str): Path of dataset.
        num_seen(int): Number of seen classes.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    CIFAR10.init(root, num_seen)
    query_dataset = CIFAR10('query', transform=query_transform())
    seen_dataset = CIFAR10('seen', transform=train_transform())
    unseen_dataset = CIFAR10('unseen', transform=train_transform())
    retrieval_dataset = CIFAR10('retrieval', transform=train_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
      )

    seen_dataloader = DataLoader(
        seen_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    unseen_dataloader = DataLoader(
        unseen_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader


class CIFAR10(Dataset):
    """
    Cifar10 dataset.
    """
    @staticmethod
    def init(root, num_seen):
        # Load data
        CIFAR10.QUERY_DATA = np.load(os.path.join(root, 'cifar10_1000_query_data.npy'))
        CIFAR10.QUERY_TARGETS = np.load(os.path.join(root, 'cifar10_1000_query_onehot_targets.npy'))
        CIFAR10.RETRIEVAL_DATA = np.load(os.path.join(root, 'cifar10_59000_retrieval_data.npy'))
        CIFAR10.RETRIEVAL_TARGETS = np.load(os.path.join(root, 'cifar10_59000_retrieval_onehot_targets.npy'))

        # Split seen data
        seen_index = 5900 * num_seen
        CIFAR10.SEEN_DATA = CIFAR10.RETRIEVAL_DATA[:seen_index, :]
        CIFAR10.SEEN_TARGETS = CIFAR10.RETRIEVAL_TARGETS[:seen_index, :]
        CIFAR10.UNSEEN_DATA = CIFAR10.RETRIEVAL_DATA[seen_index:, :]
        CIFAR10.UNSEEN_TARGETS = CIFAR10.RETRIEVAL_TARGETS[seen_index:, :]

        unseen_index = np.array(list(set(range(CIFAR10.RETRIEVAL_DATA.shape[0])) - set(range(seen_index))), dtype=np.int)
        CIFAR10.UNSEEN_INDEX = unseen_index

    def __init__(self, mode,
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'seen':
            self.data = CIFAR10.SEEN_DATA
            self.targets = CIFAR10.SEEN_TARGETS
        elif mode == 'unseen':
            self.data = CIFAR10.UNSEEN_DATA
            self.targets = CIFAR10.UNSEEN_TARGETS
        elif mode == 'query':
            self.data = CIFAR10.QUERY_DATA
            self.targets = CIFAR10.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = CIFAR10.RETRIEVAL_DATA
            self.targets = CIFAR10.RETRIEVAL_TARGETS
        else:
            raise ValueError('Mode error!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.targets)
