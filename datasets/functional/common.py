import datasets

__all__ = ['more', 'find']


def more(cfg):
    dataset = getattr(datasets, cfg.name, None)
    return dataset.more(dataset._more(cfg)) if dataset else cfg


def find(name):
    dataset = getattr(datasets, name, None)
    return dataset if dataset is not None and issubclass(dataset, datasets.BaseDataset) else None
