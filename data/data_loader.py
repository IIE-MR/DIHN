import torch
import numpy as np
import data.cifar10 as cifar10

from data.transform import train_transform
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_seen, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_seen(int): Number of seen classes.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, seen_dataloader, unseen_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                                                                                       num_seen,
                                                                                                       batch_size,
                                                                                                       num_workers,
                                                                                                       )
    # elif dataset == 'nus-wide-tc21':
    #     query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader = nuswide.load_data(root,
    #                                                                                                    num_seen,
    #                                                                                                    batch_size,
    #                                                                                                    num_workers,
    #                                                                                                    )
    else:
        raise ValueError("Invalid dataset name!")

    return query_dataloader, seen_dataloader, unseen_dataloader, retrieval_dataloader


def sample_dataloader(dataloader, num_samples, batch_size, root, dataset):
    """
    Sample data from dataloder.

    Args
        dataloader(torch.utils.data.DataLoader): Dataloader.
        num_samples(int): Number of samples.
        batch_size(int): Batch size.
        root(str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        sample_loader(torch.utils.data.DataLoader): Sample dataloader.
        omega(torch.Tensor): Sample index.
        unseen_sample_in_unseen_index(torch.Tensor): Index of unseen samples in unseen dataset.
        unseen_sample_in_sample_index(torch.Tensor): Index of unseen samples in sampling dataset.
    """
    data = dataloader.dataset.data
    targets = dataloader.dataset.targets
    num_retrieval = len(data)

    omega = np.random.permutation(num_retrieval)[:num_samples]
    data = data[omega]
    targets = targets[omega]
    sample_loader = wrap_data(data, targets, batch_size, root, dataset)

    unseen_sample_in_unseen_index = torch.from_numpy(np.array([idx for idx in range(dataloader.dataset.UNSEEN_INDEX.shape[0]) if dataloader.dataset.UNSEEN_INDEX[idx] in omega], np.int))
    unseen_sample_in_sample_index = torch.from_numpy(np.array([idx for idx in range(omega.shape[0]) if omega[idx] in dataloader.dataset.UNSEEN_INDEX], np.int))

    return sample_loader, omega, unseen_sample_in_unseen_index, unseen_sample_in_sample_index


def wrap_data(data, targets, batch_size, root, dataset):
    """
    Wrap data into dataloader.

    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, data, targets, root, dataset):
            self.data = data
            self.targets = targets
            self.root = root
            self.transform = train_transform()
            self.dataset = dataset
            self.onehot_targets = self.targets

        def __getitem__(self, index):
            img = Image.fromarray(self.data[index])
            img = self.transform(img)
            return img, self.targets[index], index

        def __len__(self):
            return self.data.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.targets)

    dataset = MyDataset(data, targets, root, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
    )

    return dataloader
