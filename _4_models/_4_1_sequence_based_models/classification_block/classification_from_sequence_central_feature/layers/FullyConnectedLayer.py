import torch
from torch.nn import Module, Linear, ReLU, Dropout


class FullyConnectedLayer(Module):

    def __init__(self):
        super(FullyConnectedLayer, self).__init__()
        self.__setup_done_flag = False

        self.fully_connected_layer = None
        self.activation_function = None
        self.dropout_layer = None

    def setup(self, in_features: int, out_features: int, dropout_rate: float):
        assert not self.__setup_done_flag

        self.fully_connected_layer = Linear(in_features, out_features)
        self.activation_function = ReLU()
        self.dropout_layer = Dropout(p=dropout_rate)

        self.__setup_done_flag = True

    # x of shape (*, in_features)
    # output of shape (*, out_features)
    def forward(self, x):
        assert self.__setup_done_flag

        x = self.fully_connected_layer(x)
        x = self.activation_function(x)
        x = self.dropout_layer(x)

        return x
