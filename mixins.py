"""mixins"""
from abc import ABCMeta as _ABCMeta

class MixinNameSpace(_ABCMeta):
    def __init__(self, *args, **kwargs):  # noqa
        raise NotImplementedError('This class is intended as a namespace and cannot be directly created')

    def __new__(cls, *args, **kwargs):  # noqa
        raise NotImplementedError('This class is intended as a namespace and cannot be directly created')
