import os
import glob

import torch
import h5py
import numpy as np


class OSG_VSD_DATASET(torch.utils.data.Dataset):
    def __init__(self, path_to_h5, device):
        self.path_to_h5 = path_to_h5
        self.device = device
        self.num_of_h5 = len(glob.glob(os.path.join(path_to_h5, "*.hdf5")))

    def __len__(self):
        return self.num_of_h5

    def __getitem__(self, idx):
        data = h5py.File(os.path.join(self.path_to_h5, f"{idx}.hdf5"), "r")

        return torch.tensor(
            data["x"][:], dtype=torch.float, device=self.device
        ), torch.tensor(data["t"][:], dtype=torch.float, device=self.device)


my_collate_err_msg_format = "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"


def my_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    if isinstance(batch[0], tuple):
        return (my_collate(samples) for samples in zip(*batch))

    if not isinstance(batch[0], torch.Tensor):
        raise TypeError(my_collate_err_msg_format.format(type(batch[0])))

    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-1)
