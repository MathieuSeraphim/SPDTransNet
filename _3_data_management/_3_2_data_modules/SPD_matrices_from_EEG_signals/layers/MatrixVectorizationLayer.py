from typing import Union

import torch
from torch.nn import Module
from _4_models.utils import matrix_log


class MatrixVectorizationLayer(Module):

    def __init__(self):
        super(MatrixVectorizationLayer, self).__init__()
        self.__setup_done_flag = False

        self.matrix_size = None
        self.vector_size = None
        self.svd_epsilon_if_any = None

    def setup(self, matrix_size: int, svd_singular_value_minimum: Union[float, None] = None):
        assert not self.__setup_done_flag
        self.matrix_size = matrix_size
        self.vector_size = int((matrix_size * (matrix_size + 1)) / 2)

        if svd_singular_value_minimum is not None:
            assert svd_singular_value_minimum > 0
        self.svd_epsilon_if_any = svd_singular_value_minimum

        self.__setup_done_flag = True
        return self.vector_size

    # spd_matrices_to_vectorize of shape (..., matrix_size, matrix_size)
    # output of shape (..., matrix_size * (matrix_size + 1) / 2)
    def forward(self, spd_matrices_to_vectorize: torch.Tensor):
        assert self.__setup_done_flag

        matrices_shape = spd_matrices_to_vectorize.shape
        assert matrices_shape[-2] == matrices_shape[-1] == self.matrix_size
        output_shape = [*matrices_shape[:-2], self.vector_size]

        symmetric_matrices_to_vectorize = matrix_log(spd_matrices_to_vectorize, epsilon=self.svd_epsilon_if_any)
        assert symmetric_matrices_to_vectorize.shape == matrices_shape

        upper_triangular_mask = torch.triu(torch.ones(matrices_shape)) == 1
        vectorized_matrices = symmetric_matrices_to_vectorize[upper_triangular_mask].view(output_shape)

        return vectorized_matrices



