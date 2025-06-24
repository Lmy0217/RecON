import abc
import os

import torch
import torch.nn as nn

import configs
from datasets import BaseDataset
from utils import Logger
from utils.common import get_path

__all__ = ['BaseModel']


class _ProcessHook(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train(self, epoch_info: dict, sample_dict: dict):
        raise NotImplementedError

    def train_return_hook(self, epoch_info: dict, return_all: dict):
        return return_all

    @abc.abstractmethod
    def test(self, epoch_info: dict, sample_dict: dict):
        raise NotImplementedError

    def test_return_hook(self, epoch_info: dict, return_all: dict):
        return return_all


class BaseModel(_ProcessHook, metaclass=abc.ABCMeta):

    dataset: BaseDataset
    logger: Logger
    main_msg: dict

    def __init__(self, cfg, data_cfg, run, **kwargs):
        self.name = os.path.splitext(os.path.split(cfg._path)[1])[0]
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.run = run
        self.path = get_path(cfg, data_cfg, run)
        self.device = self.run.device

        self._save_list = []

        for k, v in kwargs.items():
            setattr(self, k, v)

    def apply(self, fn):
        for name, value in self.__dict__.items():
            if isinstance(value, nn.Module):
                self.__dict__[name].apply(fn)

    def modules(self):
        m = {}
        for name, value in list(vars(self).items()):
            if isinstance(value, nn.Module):
                m[name] = value
        return m

    def train_return_hook(self, epoch_info: dict, return_all: dict):
        _count = torch.tensor(return_all.pop('_count'), dtype=torch.float32, device=self.device)
        _count_sum = torch.sum(_count)
        for key, value in return_all.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32, device=self.device)
            elif value.device != self.device:
                value = value.to(self.device)
            return_all[key] = _count @ value / _count_sum
        return return_all

    def load(self, start_epoch=None, path=None):
        assert start_epoch is None or (isinstance(start_epoch, int) and start_epoch >= 0)
        path = path or self.path
        if start_epoch is None:
            check_path = os.path.join(path, self.name + configs.env.paths.check_file)
            if os.path.exists(check_path):
                check_data = torch.load(check_path)
                start_epoch = check_data['epoch']
                self.main_msg = check_data['main_msg']
            else:
                start_epoch = 0
        if start_epoch > 0:
            for name, value in self.__dict__.items():
                if isinstance(value, (nn.Module, torch.optim.Optimizer)) or name in self._save_list:
                    load_path = os.path.join(path, self.name + '_' + name + '_' + str(start_epoch) + '.pth')
                    if not os.path.exists(load_path) and isinstance(value, torch.optim.Optimizer):
                        self.logger.info(f"IGNORE! Optimizer weight `{load_path}` not found!")
                        continue
                    load_value = torch.load(load_path, map_location=self.device)
                    if isinstance(value, (nn.Module, torch.optim.Optimizer)):
                        self.__dict__[name].load_state_dict(load_value)
                    else:
                        self.__dict__[name] = load_value
        return start_epoch

    def save(self, epoch, path=None):
        path = path or self.path
        if not os.path.exists(path):
            os.makedirs(path)
        for name, value in self.__dict__.items():
            if isinstance(value, (nn.Module, torch.optim.Optimizer)) or name in self._save_list:
                save_value = value.state_dict() if isinstance(value, (nn.Module, torch.optim.Optimizer)) else value
                torch.save(save_value, os.path.join(path, self.name + '_' + name + '_' + str(epoch) + '.pth'))
        torch.save(dict(epoch=epoch, main_msg=self.main_msg),
                   os.path.join(path, self.name + configs.env.paths.check_file))
