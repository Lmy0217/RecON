from collections.abc import Iterable
from itertools import repeat


""" Copied from torch.nn.modules.utils """


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
