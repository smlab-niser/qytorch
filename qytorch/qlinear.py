from typing import Union, Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from quat_base import _construct_matrix
r"""Quaternion Linear Layer.
    
    # All features of torch.nn.Linear layer from PyTorch are supported.
    
    # You need to remember these things when using this layer:
    #     - in_features and out_features must be divisible by 4.
    #     - We do not use quaternion for bias. bias count is not reduced. Only weight count is reduced to 25%.

    Examples:
        >>> model = QLinear(20, 16)  # 20 and 16 are divisible by 4
        >>> x = torch.randn(128, 20)
        >>> output = model(x)
        >>> print(output.size())
        torch.Size([128, 16])
    """

class QLinear(nn.Module):  
    """Quaternion Linear Layer.
    
    All features of `torch.nn.Linear` layer from PyTorch are supported.
    
    You need to remember these things when using this layer:
        - `in_features` and `out_features` must be divisible by 4.
        - This resulting layer will have `in_features * out_features / 4` parameters.
        - We do not use quaternion for `bias`. bias count/size is not reduced. Only weight count/size is reduced to 25%.
    """    
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    r_weight: torch.Tensor
    i_weight: torch.Tensor
    j_weight: torch.Tensor
    k_weight: torch.Tensor

    def __init__(self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device = None, 
        dtype = None) -> None:
        """Initializing Quaternion Linear Layer.

        Args:
            in_features (`int`): The number of input features. Should be divisible by 4.
            out_features (`int`): The number of output features. Should be divisible by 4.
            bias (`bool`, optional): If `True`, adds a learnable bias to the output. Defaults to `True`.
            device (optional): The desired device of the parameters. Defaults to `None`.
            dtype (optional): The desired data type of the parameters. Defaults to `None`.
        
        Examples:
            >>> model = QLinear(20, 16)  # 20 and 16 are divisible by 4
            >>> x = torch.randn(128, 20)
            >>> output = model(x)
            >>> print(output.size())
            torch.Size([128, 16])
        """               
        torch.tensor
        assert in_features % 4 == 0 and out_features % 4 == 0, "in_channels and out_channels must be divisible by 4"
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        a, b = out_features//4, in_features//4
        self.r_weight = nn.Parameter(torch.empty((a, b), **factory_kwargs))
        self.i_weight = nn.Parameter(torch.empty((a, b), **factory_kwargs))
        self.j_weight = nn.Parameter(torch.empty((a, b), **factory_kwargs))
        self.k_weight = nn.Parameter(torch.empty((a, b), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the layer.
        """        
        nn.init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))
        if self.bias is not None:
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_in = self.in_features * 4
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer.

        Args:
            x (torch.Tensor): The input tensor. The shape of the tensor should be `(..., in_features)`.
        
        Returns:
            torch.Tensor: The output tensor. The shape of the tensor will be `(..., out_features)`.
        """        
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        """Returns additional information about the layer

        Returns:
            str: The extra representation of the layer in string format.
            
        Examples:
            >>> model = QLinear(20, 16)
            >>> print(model)
            QLinear(in_features=20, out_features=16, bias=True)
        """        
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def get_weight(self):
        """Constructs a matrix from the given quaternion components. To  to be used in quaternion multiplications (Hamilton product).
        """
        return _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)

if __name__ == '__main__':  # testing
    print("QLinear:")
    rmodel = nn.Linear(20, 16)
    qmodel = QLinear(20, 16)
    input_ = torch.randn(128, 20)
    routput = rmodel(input_)
    qoutput = qmodel(input_)
    print(f"{rmodel.weight.size() = }")
    print(f"{qmodel.r_weight.size() = }")
    print(f"{qmodel.i_weight.size() = }")
    print(f"{qmodel.j_weight.size() = }")
    print(f"{qmodel.k_weight.size() = }")
    print(f"{input_.size() = }\n{routput.size() = }\n{qoutput.size() = }")
    print("routput and qoutput should have same shape")
