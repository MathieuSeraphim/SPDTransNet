from torch.nn import Module


class BaseClassificationBlock(Module):

    def __init__(self, **init_args):
        super(BaseClassificationBlock, self).__init__()

    def setup(self, **setup_args):
        raise NotImplementedError

    def forward(self, **inputs):
        raise NotImplementedError
