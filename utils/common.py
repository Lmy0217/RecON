import os

import configs
import random

import numpy as np
import torch

__all__ = ['set_seed', 'merge_dict', 'get_filename', 'get_path', 'real_config_path']


def set_seed(seed=0):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def merge_dict(dst: dict, src: dict):
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(-1)
            if key in dst.keys():
                dst[key] = torch.cat([dst[key], value.detach()])
            else:
                dst[key] = value.detach()
        else:
            if key in dst.keys():
                dst[key].append(value)
            else:
                dst[key] = [value]


def get_filename(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def get_path(model_cfg, dataset_cfg, run_cfg):
    dirname = get_filename(model_cfg._path) + '-' + get_filename(run_cfg._path) + '-' + get_filename(dataset_cfg._path)
    return os.path.join(configs.env.getdir(configs.env.paths.save_folder), dirname)


def real_config_path(config_path, set_folder):
    if os.path.exists(config_path):
        return os.path.abspath(config_path)
    else:
        return configs.env.getdir(os.path.join(set_folder, config_path + '.json'))
