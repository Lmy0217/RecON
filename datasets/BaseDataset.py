import abc
import os
import time

import torch
from torch.utils.data import Dataset

import configs
from utils import common, Logger

__all__ = ['BaseDataset', 'BaseSplit']


class BaseDataset(Dataset, metaclass=abc.ABCMeta):

    logger: Logger

    def __init__(self, cfg, **kwargs):
        self.name = os.path.splitext(os.path.split(cfg._path)[1])[0]
        self.cfg = self.more(self._more(cfg))
        self.data, self.cfg.data_count = self.load()

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _more(cfg):
        for name, value in configs.env.dataset.dict().items():
            setattr(cfg, name, getattr(cfg, name, value))
        return cfg

    @staticmethod
    def more(cfg):
        return cfg

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.cfg.data_count

    def split(self):
        self.trainset_length = int(self.cfg.series_per_data[0] * self.cfg.train_test_range[0])
        self.valset_length = int(self.cfg.series_per_data[1] * self.cfg.train_test_range[1])
        self.testset_length = len(self) - self.trainset_length - self.valset_length

        index_range_trainset = [[0, self.trainset_length]]
        index_range_valset = [[self.trainset_length, self.trainset_length + self.valset_length]]
        index_range_testset = [[self.trainset_length + self.valset_length, len(self)]]

        return BaseSplit(self, index_range_trainset), BaseSplit(self, index_range_valset), BaseSplit(self, index_range_testset)

    def get_idx(self, index):
        if index < self.trainset_length:
            idx = torch.div(index, self.cfg.series_per_data[0], rounding_mode='floor')
            common.set_seed(int(time.time() * 1000) % (1 << 32) + index)
        elif index < self.trainset_length + self.valset_length:
            idx = torch.div(index - self.trainset_length, self.cfg.series_per_data[1], rounding_mode='floor') \
                  + torch.div(self.trainset_length, self.cfg.series_per_data[0], rounding_mode='floor')
            common.set_seed(index * 3)
        else:
            idx = torch.div(index - self.trainset_length - self.valset_length, self.cfg.series_per_data[2], rounding_mode='floor') \
                  + torch.div(self.valset_length, self.cfg.series_per_data[1], rounding_mode='floor') \
                  + torch.div(self.trainset_length, self.cfg.series_per_data[0], rounding_mode='floor')
            common.set_seed(index * 3)
        return idx


class BaseSplit(Dataset):

    def __init__(self, dataset, index_range_set):
        self.dataset = dataset
        self.indexset = self._index(index_range_set)
        self.count = len(self.indexset)

        if hasattr(self.dataset, 'logger'):
            self.logger = self.dataset.logger

    def _index(self, index_range_set):
        indexset = []
        for index_range in index_range_set:
            indexset.extend(range(index_range[0], index_range[1]))
        return indexset

    def __getitem__(self, index):
        return self.dataset[self.indexset[index]][0], index

    def __len__(self):
        return self.count
