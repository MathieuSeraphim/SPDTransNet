from torch.nn import Module


class BaseDataFormattingBlock(Module):

    def __init__(self, **init_args):
        super(BaseDataFormattingBlock, self).__init__()

    def setup(self, **setup_args):
        raise NotImplementedError

    def forward(self, **inputs):
        raise NotImplementedError
