import json
from typing import Union

__all__ = ['BaseConfig']


class BaseConfig(object):

    _path: str
    name: str

    def __init__(self, cfg: Union[dict, str], **kwargs):
        self._space = 0
        self._load(cfg) if isinstance(cfg, dict) else self._fromfile(cfg)

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _values(key: str):
        return not (key.startswith('_') or key == 'name')

    def dict(self):
        value_dict = dict(vars(self))
        for key in list(value_dict.keys()):
            if not self._values(key):
                value_dict.pop(key, None)
        return value_dict

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            for key in vars(other).keys():
                if other._values(key) and not hasattr(self, key):
                    return False
            for key, value in vars(self).items():
                if self._values(key) and (not hasattr(other, key) or value != getattr(other, key)):
                    return False
            return True
        else:
            return False

    def __repr__(self):
        s, sp = '', ' ' * self._space
        if self._space == 0:
            s += self.__class__.__name__
            if hasattr(self, 'name'):
                s += ' (' + str(self.name) + ')'
            s += ': '
        s += '{\n'
        v = list(vars(self).items())
        v.sort()
        for key, value in v:
            if not self._values(key):
                continue
            s += sp + '  ' + key + ': '
            if issubclass(value.__class__, BaseConfig):
                value._space = self._space + 2
            if isinstance(value, str):
                s += "'"
            s += str(value)
            if isinstance(value, str):
                s += "'"
            s += '\n'
        s += sp + '}'
        if self._space == 0:
            s += '\n'
        return s

    def _load(self, config: dict):
        for key, value in config.items():
            setattr(self, key, BaseConfig(value) if isinstance(value, dict) else value)

    def _fromfile(self, path: str):
        with open(path, 'r') as f:
            self._load(json.load(f))
        setattr(self, '_path', path)
