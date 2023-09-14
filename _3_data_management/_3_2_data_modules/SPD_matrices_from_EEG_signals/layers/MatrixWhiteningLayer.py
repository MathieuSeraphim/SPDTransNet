import warnings
import torch
from torch.nn import Module
from _4_models.utils import matrix_pow


class MatrixWhiteningLayer(Module):

    def __init__(self):
        super(MatrixWhiteningLayer, self).__init__()
        self.__setup_done_flag = False
        self.__inactive_flag = True

        self.matrix_size = None
        self.extra_dimensions = None
        self.matrix_multiplication_factor_if_any = None

    def setup(self, matrix_size, operate_whitening: bool = True, extra_dimensions: int = 0,
              matrix_multiplication_factor: float = 1.):
        assert not self.__setup_done_flag
        self.__inactive_flag = not operate_whitening

        self.matrix_size = matrix_size

        assert extra_dimensions >= 0
        self.extra_dimensions = extra_dimensions

        if matrix_multiplication_factor != 1.:
            assert matrix_multiplication_factor > 0
            self.matrix_multiplication_factor_if_any = matrix_multiplication_factor

        self.__setup_done_flag = True

    # sdp_matrices_to_whiten of shape (<first dims>, <extra dims>, matrix_size, matrix_size)
    # sdp_whitening_matrices of shape (<first dims>, matrix_size, matrix_size)
    # output of shape (<first dims>, <extra dims>, matrix_size, matrix_size)
    def forward(self, sdp_matrices_to_whiten, sdp_whitening_matrices):
        assert self.__setup_done_flag

        if self.__inactive_flag:
            assert sdp_matrices_to_whiten.shape[-1] == sdp_matrices_to_whiten.shape[-2] == self.matrix_size
            warnings.warn("The MatrixWhiteningLayer is currently being bypassed.")
            return sdp_matrices_to_whiten

        # Raised to power -1/2
        sdp_whitening_matrices = matrix_pow(sdp_whitening_matrices, -.5)

        if self.extra_dimensions == 0:
            assert sdp_matrices_to_whiten.shape == sdp_whitening_matrices.shape
            assert sdp_matrices_to_whiten.shape[-1] == sdp_matrices_to_whiten.shape[-2] == self.matrix_size

        else:
            sdp_matrices_shape = sdp_matrices_to_whiten.shape
            whitening_matrices_shape = sdp_whitening_matrices.shape

            assert len(whitening_matrices_shape) + self.extra_dimensions == len(sdp_matrices_shape)
            assert sdp_matrices_shape[-2] == sdp_matrices_shape[-1] == self.matrix_size
            assert sdp_matrices_shape[-2:] == whitening_matrices_shape[-2:]
            if len(whitening_matrices_shape) > 2:
                assert whitening_matrices_shape[:-2] == sdp_matrices_shape[:-2 - self.extra_dimensions]

            for i in range(self.extra_dimensions):
                sdp_whitening_matrices = sdp_whitening_matrices.unsqueeze(-3)
            sdp_whitening_matrices = sdp_whitening_matrices.expand(sdp_matrices_shape)

        whitened_spd_matrices = torch.matmul(sdp_whitening_matrices, torch.matmul(sdp_matrices_to_whiten, sdp_whitening_matrices))
        assert whitened_spd_matrices.shape == sdp_matrices_to_whiten.shape

        if self.matrix_multiplication_factor_if_any is not None:
            whitened_spd_matrices = whitened_spd_matrices * self.matrix_multiplication_factor_if_any

        return whitened_spd_matrices
