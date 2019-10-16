import torch
import os
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_seen, batch_size, num_workers):
    """
    Loading nus-wide dataset.

    Args:
        root(str): Path of image files.
        num_seen(str): Number of classes of seen.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
       query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    NusWideDatasetTC21.init(root, num_seen)
    query_dataset = NusWideDatasetTC21(
        root,
        'query',
        transform=query_transform(),
    )

    retrieval_dataset = NusWideDatasetTC21(
        root,
        'retrieval',
        transform=train_transform(),
    )

    unseen_dataset = NusWideDatasetTC21(
        root,
        'unseen',
        transform=train_transform(),
    )

    seen_dataset = NusWideDatasetTC21(
        root,
        'seen',
        transform=train_transform(),
    )

    query_dataloader = DataLoader(
        query_dataset,
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

    unseen_dataloader = DataLoader(
        unseen_dataset,
        shuffle=True,
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

    return query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader


class NusWideDatasetTC21(Dataset):
    """
    Nus-wide dataset, 21 classes.

    Args
        root(str): Path of image files.
        mode(str): 'query', 'seen', 'unseen', 'retrieval'.
        transform(callable, optional): Image transform.
    """
    @staticmethod
    def init(root, num_seen):
        """
        Initialization.

        Args
            root(str): Path of dataset.
            num_seen(str): Number of classes of seen.
        """
        retrieval_img_txt_path = os.path.join(root, 'database_img.txt')
        retrieval_label_txt_path = os.path.join(root, 'database_label_onehot.txt')
        query_img_txt_path = os.path.join(root, 'test_img.txt')
        query_label_txt_path = os.path.join(root, 'test_label_onehot.txt')

        # Read files
        with open(retrieval_img_txt_path, 'r') as f:
            NusWideDatasetTC21.RETRIEVAL_DATA = np.array([i.strip() for i in f])
        NusWideDatasetTC21.RETRIEVAL_TARGETS = np.loadtxt(retrieval_label_txt_path, dtype=np.float32)

        with open(query_img_txt_path, 'r') as f:
            NusWideDatasetTC21.QUERY_DATA = np.array([i.strip() for i in f])
        NusWideDatasetTC21.QUERY_TARGETS = np.loadtxt(query_label_txt_path, dtype=np.float32)

        # Split seen, unseen
        unseen_index = np.array([])
        num_retrieval = NusWideDatasetTC21.RETRIEVAL_TARGETS.shape[0]
        for i in range(num_seen, 21):
            unseen_index = np.concatenate((unseen_index, (NusWideDatasetTC21.RETRIEVAL_TARGETS[:, i] == 1).nonzero()[0]))
        unseen_index = set([idx for idx in unseen_index])
        seen_index = np.array(list(set(range(num_retrieval)) - unseen_index), dtype=np.int)
        unseen_index = np.array(list(unseen_index), dtype=np.int)

        NusWideDatasetTC21.UNSEEN_DATA = NusWideDatasetTC21.RETRIEVAL_DATA[unseen_index]
        NusWideDatasetTC21.UNSEEN_TARGETS = NusWideDatasetTC21.RETRIEVAL_TARGETS[unseen_index, :]
        NusWideDatasetTC21.SEEN_DATA = NusWideDatasetTC21.RETRIEVAL_DATA[seen_index]
        NusWideDatasetTC21.SEEN_TARGETS = NusWideDatasetTC21.RETRIEVAL_TARGETS[seen_index, :]

        NusWideDatasetTC21.RETRIEVAL_DATA = np.concatenate((NusWideDatasetTC21.SEEN_DATA, NusWideDatasetTC21.UNSEEN_DATA))
        NusWideDatasetTC21.RETRIEVAL_TARGETS = np.concatenate((NusWideDatasetTC21.SEEN_TARGETS, NusWideDatasetTC21.UNSEEN_TARGETS))

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        self.mode = mode

        if mode == 'query':
            self.data = NusWideDatasetTC21.QUERY_DATA
            self.targets = NusWideDatasetTC21.QUERY_TARGETS
        elif mode == 'unseen':
            self.data = NusWideDatasetTC21.UNSEEN_DATA
            self.targets = NusWideDatasetTC21.UNSEEN_TARGETS
        elif mode == 'seen':
            self.data = NusWideDatasetTC21.SEEN_DATA
            self.targets = NusWideDatasetTC21.SEEN_TARGETS
        elif mode == 'retrieval':
            self.data = NusWideDatasetTC21.RETRIEVAL_DATA
            self.targets = NusWideDatasetTC21.RETRIEVAL_TARGETS
        else:
            raise ValueError('Mode error!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()
