""" Data loader definition. """

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class dataset_h5(Dataset):
    def __init__(self, h5_file, node_dim, mip_dim, var_dim):
        """
        :param h5_file: str, pathway to the data H5 file
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param var_dim: int, dimension of variable state
        """
        super(dataset_h5, self).__init__()

        # load the h5 file
        self.h5_file = h5py.File(h5_file, 'r')

        # define the dimensions of each feature
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.var_dim = var_dim

        # define the number of data points
        self.n_data = len(self.h5_file['dataset'])

    def __getitem__(self, index):
        x = self.h5_file['dataset'][index]
        return [torch.LongTensor([x[0]]),
                torch.FloatTensor(x[1:1 + self.node_dim]),
                torch.FloatTensor(x[1 + self.node_dim:1 + self.node_dim + self.mip_dim]),
                torch.FloatTensor(x[1 + self.node_dim + self.mip_dim:].reshape(-1, self.var_dim))
                ]

    def __len__(self):
        return self.n_data


def collate_fn(batch):
    batch_list = [item for item in batch]
    return batch_list
