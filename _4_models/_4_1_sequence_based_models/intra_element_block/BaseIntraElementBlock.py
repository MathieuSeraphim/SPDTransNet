from torch.nn import Module


class BaseIntraElementBlock(Module):

    def __init__(self, **init_args):
        super(BaseIntraElementBlock, self).__init__()

    def setup(self, **setup_args):
        raise NotImplementedError

    def forward(self, **inputs):
        raise NotImplementedError
