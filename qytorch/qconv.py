import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, List, Tuple

from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

from quat_base import _construct_matrix





class _QConvNd(nn.Module):
    """Quaternion Convolution Base Class
    
    All features of torch.nn.ConvNd layer from PyTorch are supported.
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        """Initializing Quaternion Convolution Base Class

        Args:
            in_channels (int): `in_channels` for the input tensor.
            out_channels (int): `out_channels` for the output tensor.
            kernel_size (Tuple[int, ...]): `kernel_size` for the convolution.
            stride (Tuple[int, ...]): `stride` for the convolution.
            padding (Tuple[int, ...]): `padding` for the convolution.
            dilation (Tuple[int, ...]): `dilation` for the convolution.
            transposed (bool): True if this is a transposed convolution.
            output_padding (Tuple[int, ...]): `output_padding` for the convolution.
            groups (int): `groups` for the convolution.
            bias (bool): True if bias is used.
            padding_mode (str): `padding_mode` for the convolution.
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
        """        
        
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            # self.weight = Parameter(torch.empty((in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
            a, b = in_channels, out_channels // groups
            assert a%4 == 0 and b%4 == 0, f"in_channels={a}, (out_channels//groups)={b} must be divisible by 4"
            self.r_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
            self.i_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
            self.j_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
            self.k_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
        else:
            # self.weight = Parameter(torch.empty((out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
            a, b = out_channels, in_channels // groups
            assert a%4 == 0 and b%4 == 0, f"out_channels={a}, (in_channels//groups)={b} must be divisible by 4"
            self.r_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
            self.i_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
            self.j_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
            self.k_weight = Parameter(torch.empty((a//4, b//4, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.r_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.i_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.j_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.k_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.r_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        """Extra representation of the layer
        
        Returns:
            str: The extra representation of the layer in string format.
        """
        
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class QConv1d(_QConvNd):
    """Quaternion Convolution 1D
    
    All features of torch.nn.Conv1d layer from PyTorch are supported.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        """Initializing Quaternion Convolution 1D

        Args:
            in_channels (int): `in_channels` for the input tensor.
            out_channels (int): `out_channels` for the output tensor.
            kernel_size (_size_1_t): `kernel_size` for the convolution.
            stride (_size_1_t, optional): `stride` for the convolution. Defaults to 1.
            padding (Union[str, _size_1_t], optional): `padding` for the convolution. Defaults to 0.
            dilation (_size_1_t, optional): `dilation` for the convolution. Defaults to 1.
            groups (int, optional): `groups` for the convolution. Defaults to 1.
            bias (bool, optional): True if bias is used. Defaults to True.
            padding_mode (str, optional): `padding_mode` for the convolution. Defaults to 'zeros'.
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
        """        
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.get_weight()
        return self._conv_forward(input, weight, self.bias)

    def get_weight(self):
        return _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)


class QConv2d(_QConvNd):
    """Quaternion Convolution 2D
    
    All features of torch.nn.Conv2d layer from PyTorch are supported.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        """Initializing Quaternion Convolution 2D
        
        Args:
            in_channels (int): `in_channels` for the input tensor.
            out_channels (int): `out_channels` for the output tensor.
            kernel_size (_size_2_t): `kernel_size` for the convolution.
            stride (_size_2_t, optional): `stride` for the convolution. Defaults to 1.
            padding (Union[str, _size_2_t], optional): `padding` for the convolution. Defaults to 0.
            dilation (_size_2_t, optional): `dilation` for the convolution. Defaults to 1.
            groups (int, optional): `groups` for the convolution. Defaults to 1.
            bias (bool, optional): True if bias is used. Defaults to True.
            padding_mode (str, optional): `padding_mode` for the convolution. Defaults to 'zeros'.
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
        
        """
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.get_weight()
        return self._conv_forward(input, weight, self.bias)

    def get_weight(self):
        return _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)

class QConv3d(_QConvNd):
    """Quaternion Convolution 3D
    
    All features of torch.nn.Conv3d layer from PyTorch are supported.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        """Initializing Quaternion Convolution 3D
        
        Args:
            in_channels (int): `in_channels` for the input tensor.
            out_channels (int): `out_channels` for the output tensor.
            kernel_size (_size_3_t): `kernel_size` for the convolution.
            stride (_size_3_t, optional): `stride` for the convolution. Defaults to 1.
            padding (Union[str, _size_3_t], optional): `padding` for the convolution. Defaults to 0.
            dilation (_size_3_t, optional): `dilation` for the convolution. Defaults to 1.
            groups (int, optional): `groups` for the convolution. Defaults to 1.
            bias (bool, optional): True if bias is used. Defaults to True.
            padding_mode (str, optional): `padding_mode` for the convolution. Defaults to 'zeros'.
            device (optional): Defaults to None.
            dtype (optional): Defaults to None.
        
        """
        
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _triple(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: Tensor) -> Tensor:
        weight = self.get_weight()
        return self._conv_forward(input, weight, self.bias)
    
    def get_weight(self):
        return _construct_matrix(self.r_weight, self.i_weight, self.j_weight, self.k_weight)


if __name__ == '__main__':  # testing
    print("QConv1d:")
    rmodel = nn.Conv1d(20, 16, 3, stride=1, padding=1)
    qmodel = QConv1d(20, 16, 3, stride=1, padding=1)
    input_ = torch.randn(128, 20, 32)
    routput = rmodel(input_)
    qoutput = qmodel(input_)
    print(f"{rmodel.weight.size() = }")
    print(f"{qmodel.r_weight.size() = }")
    print(f"{qmodel.i_weight.size() = }")
    print(f"{qmodel.j_weight.size() = }")
    print(f"{qmodel.k_weight.size() = }")
    print(f"{input_.size() = }\n{routput.size() = }\n{qoutput.size() = }")
    print("routput and qoutput should have same shape")
    
    print("\nQConv2d:")
    rmodel = nn.Conv2d(20, 16, 3, stride=1, padding=1)
    qmodel = QConv2d(20, 16, 3, stride=1, padding=1)
    input_ = torch.randn(128, 20, 32, 32)
    routput = rmodel(input_)
    qoutput = qmodel(input_)
    print(f"{rmodel.weight.size() = }")
    print(f"{qmodel.r_weight.size() = }")
    print(f"{qmodel.i_weight.size() = }")
    print(f"{qmodel.j_weight.size() = }")
    print(f"{qmodel.k_weight.size() = }")
    print(f"{input_.size() = }\n{routput.size() = }\n{qoutput.size() = }")
    print("routput and qoutput should have same shape")
    
    print("\nQConv3d:")
    rmodel = nn.Conv3d(20, 16, 3, stride=1, padding=1)
    qmodel = QConv3d(20, 16, 3, stride=1, padding=1)
    input_ = torch.randn(128, 20, 32, 32, 32)
    routput = rmodel(input_)
    qoutput = qmodel(input_)
    print(f"{rmodel.weight.size() = }")
    print(f"{qmodel.r_weight.size() = }")
    print(f"{qmodel.i_weight.size() = }")
    print(f"{qmodel.j_weight.size() = }")
    print(f"{qmodel.k_weight.size() = }")
    print(f"{input_.size() = }\n{routput.size() = }\n{qoutput.size() = }")
    print("routput and qoutput should have same shape")
