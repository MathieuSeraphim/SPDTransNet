from torch.nn import Module, TransformerEncoderLayer, TransformerEncoder


# Non-SPD stand-in for SPDTransformerEncoderLayer, used in earlier versions of our model.
class StandardTransformerEncoderLayer(Module):

    def __init__(self):
        super(StandardTransformerEncoderLayer, self).__init__()
        self.__setup_done_flag = False
        self.encoder = None

    def setup(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int):
        assert not self.__setup_done_flag

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=num_layers)

        self.__setup_done_flag = True

    # x of shape (batch_size, sequence_length, vector_length=d_model)
    def forward(self, x):
        assert self.__setup_done_flag
        return self.encoder(x)
