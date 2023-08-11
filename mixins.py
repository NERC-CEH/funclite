"""mixins"""


class MixinNameSpace:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('This class is intended as a namespace and cannot be directly created')

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError('This class is intended as a namespace and cannot be directly created')
