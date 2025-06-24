import os

import torch

import configs

__all__ = ['Run']


class Run(configs.BaseConfig):

    epochs: int
    save_step: int
    batch_size: int
    test_batch_size: int

    def __init__(self, cfg, gpus: str = '0', **kwargs):
        super(Run, self).__init__(cfg, gpus=gpus, **kwargs)
        self._more()

    def _more(self):
        self._set_gpus()
        if self.gpus:
            self.cuda = torch.cuda.is_available() and getattr(self, 'cuda', True)
            self.device = torch.device("cuda", 0) if self.cuda else torch.device("cpu")
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def _set_gpus(self):
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            self.gpus = os.environ['CUDA_VISIBLE_DEVICES']

        if self.gpus.lower() == 'cpu':
            self.gpus = []
        elif self.gpus == '':
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = [int(g) for g in self.gpus.split(',')]

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in self.gpus])
