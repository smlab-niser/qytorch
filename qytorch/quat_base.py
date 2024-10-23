"""
Quaternion layers:
    - [x] QLinear
    - [x] QConv1d
    - [x] QConv2d
    - [x] QConv3d
    - [ ] QMultiheadAttention
"""

import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import math
def _construct_matrix(r: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r"""Constructs a matrix from the given quaternion components. To  to be used in quaternion multiplications (Hamilton product).

    Args:
        r,i,j,k (`torch.Tensor`): The real and imaginary parts of the quaternion, each of shape (in_features//4, out_features//4).

    Returns:
        `torch.Tensor`: A matrix constructed from the quaternion components. The shape of the matrix is (in_features, out_features).
    """    
    
    
    weight = torch.cat([torch.cat([r, -i, -j, -k], dim=0),
                        torch.cat([i,  r, -k,  j], dim=0),  # noqa: E241
                        torch.cat([j,  k,  r, -i], dim=0),  # noqa: E241
                        torch.cat([k, -j,  i,  r], dim=0)], dim=1)  # noqa: E241

    return weight


# def make_quaternion_mul(kernel):
#     """" The constructed 'hamilton' W is a modified version of the quaternion representation,
#         thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
#     dim = kernel.size(1) // 4
#     split_sizes = [dim] * 4
#     r, i, j, k = torch.split(kernel, split_sizes, dim=1)
#     r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
#     i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
#     j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
#     k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
#     hamilton = torch.cat([r2, i2, j2, k2], dim=1)
#     assert kernel.size(1) == hamilton.size(1)
#     return hamilton

