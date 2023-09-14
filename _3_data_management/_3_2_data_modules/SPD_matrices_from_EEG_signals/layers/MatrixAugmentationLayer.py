import warnings
import torch
import torch.nn as nn
from torch.nn import Module, Parameter


class MatrixAugmentationLayer(Module):

    def __init__(self):
        super(MatrixAugmentationLayer, self).__init__()
        self.__setup_done_flag = False
        self.__inactive_flag = True

        self.augmentation_factor = None
        self.matrix_size = None
        self.matrix_augmentation_size = None
        self.augmented_matrix_size = None
        self.corner_identity_matrix = None

    def setup(self, matrix_size: int, augmentation_size: int, initial_augmentation_factor: float = 1.,
              augmentation_factor_learnable: bool = False):
        assert not self.__setup_done_flag
        assert matrix_size >= 2
        assert augmentation_size >= 0
        augmentation_factor = float(initial_augmentation_factor)

        self.matrix_size = matrix_size
        self.matrix_augmentation_size = augmentation_size
        self.augmented_matrix_size = self.matrix_size + self.matrix_augmentation_size
        self.augmentation_factor = nn.Parameter(torch.tensor(augmentation_factor),
                                                requires_grad=augmentation_factor_learnable)
        self.corner_identity_matrix = Parameter(torch.eye(self.matrix_augmentation_size), requires_grad=False)

        self.__setup_done_flag = True
        self.__inactive_flag = self.augmentation_factor == 0
        return not self.__inactive_flag, self.augmented_matrix_size

    # spd_matrices of shape (..., matrix_size, matrix_size)
    # augmentation_matrices of shape (..., matrix_size, matrix_augmentation_size)
    # output of shape (..., augmented_matrix_size, augmented_matrix_size)
    def forward(self, spd_matrices, augmentation_matrices):
        assert self.__setup_done_flag

        if self.__inactive_flag:
            assert spd_matrices.shape[-1] == spd_matrices.shape[-2]
            warnings.warn("The MatrixAugmentationLayer is currently being bypassed.")
            return spd_matrices

        spd_matrices_shape = spd_matrices.shape
        augmentation_matrices_shape = augmentation_matrices.shape
        assert spd_matrices_shape[:-2] == augmentation_matrices_shape[:-2]
        assert spd_matrices_shape[-2] == spd_matrices_shape[-1] == augmentation_matrices_shape[-2] == self.matrix_size
        assert augmentation_matrices_shape[-1] == self.matrix_augmentation_size

        augmentation_matrices = augmentation_matrices * self.augmentation_factor
        augmentation_matrices_transposed = torch.transpose(augmentation_matrices, -2, -1)

        semi_sdp_matrices_from_augmentation_matrices = torch.matmul(augmentation_matrices,
                                                                    augmentation_matrices_transposed)
        assert semi_sdp_matrices_from_augmentation_matrices.shape == spd_matrices_shape

        central_matrices = spd_matrices + semi_sdp_matrices_from_augmentation_matrices

        corner_identity_matrices = self.corner_identity_matrix.expand(*spd_matrices_shape[:-2], self.matrix_augmentation_size, self.matrix_augmentation_size)

        upper_tensor = torch.cat((central_matrices, augmentation_matrices), dim=-1)
        lower_tensor = torch.cat((augmentation_matrices_transposed, corner_identity_matrices), dim=-1)
        final_matrices = torch.cat((upper_tensor, lower_tensor), dim=-2)

        final_matrices_shape = final_matrices.shape
        assert final_matrices_shape[:-2] == spd_matrices_shape[:-2]
        assert final_matrices_shape[-1] == final_matrices_shape[-2] == self.augmented_matrix_size

        return final_matrices


