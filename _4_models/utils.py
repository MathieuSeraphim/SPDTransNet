import datetime
import warnings
from os import mkdir, getenv
from os.path import dirname, realpath, join, isdir
from jsonargparse import ArgumentParser
from typing import Any, Union, Dict, List
import torch
from torch import Tensor
from torch.linalg import LinAlgError
from torch.nn import Module
from torch.autograd import Function
from torch.nn import functional as F


# **********************************************************************************************************************
# Adapted from: https://github.com/KingJamesSong/DifferentiableSVD/blob/main/src/representation/SVD_Taylor.py


# Taylor polynomial to approximate SVD gradients (See the derivation in the paper.)


def taylor_polynomial(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dtype = s.dtype
    I = torch.eye(s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p)
    a1 = s.view(s.shape[0],s.shape[1],1).repeat(1, 1, s.shape[1])
    a1_t = a1.transpose(1,2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones(s.shape[1], s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1) - I
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(100):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


# SVD Step
class Eigen_decomposition(Function):

    @staticmethod
    def forward(ctx, input, epsilon=None):
        p = input

        try:
            _, eig_diag, eig_vec_transposed = torch.linalg.svd(p, full_matrices=False)
            eig_vec = eig_vec_transposed.transpose(-2, -1)  # Equivalent result to the commented line

        except LinAlgError as error:
            save_error_tensor(p, error, "torch.linalg.svd", Eigen_decomposition.forward, Eigen_decomposition)
            raise error

        dtype = eig_diag.dtype
        if epsilon is None:
            epsilon = torch.finfo(dtype).eps
        else:
            epsilon = max(epsilon, torch.finfo(dtype).eps)

        eig_diag[eig_diag <= epsilon] = epsilon  # Zero-out eigenvalues smaller than epsilon
        ctx.save_for_backward(eig_vec, eig_diag)
        return eig_vec, eig_diag

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        eig_vec, eig_diag = ctx.saved_tensors
        eig_diag = eig_diag.diag_embed()
        eig_vec_deri, eig_diag_deri = grad_output1, grad_output2
        k = taylor_polynomial(eig_diag)

        # Gradient Overflow Check;
        k[k == float('inf')] = k[k != float('inf')].max()
        k[k == float('-inf')] = k[k != float('-inf')].min()
        k[k != k] = k.max()
        grad_input = (k.transpose(1, 2) * (eig_vec.transpose(1, 2).bmm(eig_vec_deri))) + torch.diag_embed(eig_diag_deri)

        # Gradient Overflow Check;
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        grad_input = eig_vec.bmm(grad_input).bmm(eig_vec.transpose(1, 2))

        # Gradient Overflow Check;
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()

        return grad_input, None  # Must have as many outputs as the forward() method has inputs


# End of adapted section
# **********************************************************************************************************************

taylorsvd = Eigen_decomposition.apply


# No special checks in this section. Only call with items you know are batched / unbatched SPD matrices.
def batch_spd_matrices_operation(spd_matrices: torch.Tensor, operation_on_vectorized_diagonal: Any,
                                 epsilon: Union[float, None] = None):
    # non_finite_values_check(spd_matrices, batch_spd_matrices_operation)

    spd_matrices_shape = spd_matrices.shape
    spd_matrices = spd_matrices.view(-1, spd_matrices_shape[-2], spd_matrices_shape[-1])

    # negative_eigenvalues_check(spd_matrices)  # Very costly time-wise!
    eigenvectors, eigenvalues_diagonal = taylorsvd(spd_matrices, epsilon)

    # non_finite_values_check(eigenvectors, batch_spd_matrices_operation)
    # non_finite_values_check(eigenvalues_diagonal, batch_spd_matrices_operation)

    transformed_eigenvalues_diagonal = operation_on_vectorized_diagonal(eigenvalues_diagonal)
    transformed_eigenvalues = transformed_eigenvalues_diagonal.diag_embed()
    transformed_output = eigenvectors @ transformed_eigenvalues @ eigenvectors.transpose(-2, -1)

    non_finite_values_check(transformed_output, batch_spd_matrices_operation)

    transformed_output = transformed_output.view(spd_matrices_shape)
    return transformed_output


# Projects SPD matrices into the set of symmetric matrices
def matrix_log(spd_matrices: torch.Tensor, epsilon: Union[float, None] = None):
    return batch_spd_matrices_operation(spd_matrices, torch.log, epsilon)


def matrix_pow(spd_matrices: torch.Tensor, pow: float, epsilon: Union[float, None] = None):
    def my_pow(vectorized_matrices: torch.Tensor):
        return torch.pow(vectorized_matrices, pow)
    return batch_spd_matrices_operation(spd_matrices, my_pow, epsilon)


def combine_dict_outputs(outputs: List[Dict]):
    keys_list = list(outputs[0].keys())
    combined_outputs_dict = {}
    for key in keys_list:
        combined_outputs_dict[key] = torch.cat([tmp[key] for tmp in outputs])
    return combined_outputs_dict


supported_loss_functions = ["cross_entropy", "cross_entropy_with_label_smoothing"]
def get_loss_function_from_dict(loss_function_config_dict: Dict[str, Any], model: Any = None):
    loss_function_name = loss_function_config_dict["name"]
    assert loss_function_name in supported_loss_functions
    if loss_function_name == "cross_entropy":
        return F.cross_entropy
    if loss_function_name == "cross_entropy_with_label_smoothing":
        if "args" in loss_function_config_dict.keys():
            extra_args = loss_function_config_dict["args"]
            return define_cross_entropy_loss_with_label_smoothing(**extra_args)
        return define_cross_entropy_loss_with_label_smoothing()
    raise NotImplementedError


def define_cross_entropy_loss_with_label_smoothing(label_smoothing: float = 0.1):
    def cross_entropy_loss_with_label_smoothing(input: torch.Tensor, target: torch.Tensor):
        return F.cross_entropy(input, target, label_smoothing=label_smoothing)
    return cross_entropy_loss_with_label_smoothing


def check_block_import(block: Union[Module, Dict]):
    if isinstance(block, Module):
        return block

    class ModuleWrapper:
        def __init__(self, module: Module):
            self.module = module

    assert isinstance(block, dict)
    parser = ArgumentParser()
    wrapper_dict = {"wrapper": {"module": block}}
    parser.add_class_arguments(ModuleWrapper, "wrapper", fail_untyped=False)
    return parser.instantiate_classes(wrapper_dict).wrapper.module


def get_timestamp_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%dT%H:%M:%S") + ("-%02d" % (now.microsecond / 10000))


def save_problematic_tensor(tensor: Tensor, problem: str):
    current_script_directory = dirname(realpath(__file__))
    root_directory = dirname(current_script_directory)
    error_tensors_folder = join(root_directory, "error_tensors")
    if not isdir(error_tensors_folder):
        mkdir(error_tensors_folder)

    timestamp_string = get_timestamp_string()

    job_id_string = ""
    current_job_id = int(getenv("SLURM_ARRAY_TASK_ID"))
    if current_job_id is not None:
        job_id_string = "job_%04d_" % current_job_id

    filename = "%s%s_%s.pt" % (job_id_string, timestamp_string, problem)
    full_filename = join(error_tensors_folder, filename)

    torch.save(tensor, open(full_filename, "wb"))


def error_message_function_string(function: Any, class_if_any: Any = None):
    function_name = function.__name__
    if class_if_any is not None:
        class_name = class_if_any.__name__
        function_string = "method %s of class %s" % (function_name, class_name)
    else:
        function_string = "function %s" % function_name
    return function_string


def save_error_tensor(tensor: Tensor, error: Exception, error_reason: str, function: Any, class_if_any: Any = None):
    error_name = error.__class__.__name__
    function_substring = error_message_function_string(function, class_if_any)
    timestamp_string = get_timestamp_string()
    warnings.warn("%s - %s generated by %s in %s." % (timestamp_string, error_name, error_reason, function_substring))
    save_problematic_tensor(tensor, error_name)


def non_finite_values_check(tensor_to_check: Tensor, function: Any, class_if_any: Any = None):
    if not torch.isfinite(tensor_to_check).all():
        error = ValueError("Tensor has non-finite values.")
        save_error_tensor(tensor_to_check, error, "non-finite values in Tensor", function, class_if_any)
        raise error


# Very costly to use!
def negative_eigenvalues_check(supposedly_spd_matrices: Tensor):
    if (torch.linalg.eigvals(supposedly_spd_matrices).real < 0).any():
        # save_problematic_tensor(supposedly_spd_matrices, "negative_eigenvalues")
        timestamp_string = get_timestamp_string()
        warnings.warn("%s - Matrix passing through SVD has negative eigenvalues!" % timestamp_string)


