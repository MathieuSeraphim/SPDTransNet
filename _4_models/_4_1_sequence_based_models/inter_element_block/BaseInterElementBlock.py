from torch.nn import Module


class BaseInterElementBlock(Module):

    def __init__(self, **init_args):
        super(BaseInterElementBlock, self).__init__()

    def setup(self, **setup_args):
        raise NotImplementedError

    def forward(self, **inputs):
        raise NotImplementedError
