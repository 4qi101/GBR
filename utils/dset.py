import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
import scipy.io as sio


class PreDataset(Dataset):
    def __init__(self, data_path, data_split='train', dataname=None, flag='sup'):
        self.root = data_path
        if dataname == 'flickr25k':
            if flag == 'sup':
                self.dset = dataname + '_clip_split_sup'
            else:
                self.dset = dataname + '_clip_split'
        elif dataname == 'nuswide':
            if flag == 'sup':
                self.dset = dataname + '_clip_split_sup'
            else:
                self.dset = dataname + '_clip_split'
        elif dataname == 'coco':
            if flag == 'sup':
                self.dset = dataname + '_clip_split_sup'
            else:
                self.dset = dataname + '_clip_split'
        else:
            raise ValueError(f"Unsupported dataset name: {dataname}")

        self.data_path = osp.join(self.root, dataname, f'{self.dset}.mat')
        self.data = self._load_mat_file(self.data_path)

        if data_split == 'train':
            self.images = self._read_split('I_tr')
            self.texts = self._read_split('T_tr')
            self.labels = self._read_split('L_tr')
        elif data_split == 'test':
            self.images = self._read_split('I_te')
            self.texts = self._read_split('T_te')
            self.labels = self._read_split('L_te')
        elif data_split == 'retrieval':
            self.images = self._read_split('I_db')
            self.texts = self._read_split('T_db')
            self.labels = self._read_split('L_db')
        elif data_split == 'all':
            self.images = np.concatenate((self._read_split('I_db'), self._read_split('I_te')), axis=0)
            self.texts = np.concatenate((self._read_split('T_db'), self._read_split('T_te')), axis=0)
            self.labels = np.concatenate((self._read_split('L_db'), self._read_split('L_te')), axis=0)
        else:
            raise ValueError(f"Unsupported data split: {data_split}")

        self.images = self._ensure_sample_first(self.images)
        self.texts = self._ensure_sample_first(self.texts)
        self.labels = self._ensure_sample_first(self.labels)

        self.length = len(self.labels)

    def _load_mat_file(self, path):
        data = sio.loadmat(path)
        for meta_key in ['__header__', '__version__', '__globals__']:
            data.pop(meta_key, None)
        return data

    def _read_split(self, key):
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in {self.data_path}")
        array = self.data[key]

        array = np.asarray(array)
        array = np.squeeze(array)

        if array.ndim >= 2:
            array = array.T

        return array

    def _ensure_sample_first(self, array):
        array = np.asarray(array)
        if array.ndim != 2:
            return array
        if array.shape[0] < array.shape[1]:
            return array.T
        return array

    def __getitem__(self, index):
        img = torch.as_tensor(self.images[index], dtype=torch.float32)
        text = torch.as_tensor(self.texts[index], dtype=torch.float32)
        label = torch.as_tensor(self.labels[index], dtype=torch.float32)
        return img, text, label, index

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass
