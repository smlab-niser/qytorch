from typing import Union, Callable

import torch
from torch import Tensor

import torch.nn.functional as F
from torch.nn import Dropout, Linear, LayerNorm
from .qmha import QMultiheadAttention
from torch.nn.modules.transformer import TransformerEncoder, TransformerDecoder, _get_activation_fn



def get_qtransformer(
        d_model: int = 512, 
        nhead: int = 8, 
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5, 
        batch_first: bool = False, 
        norm_first: bool = False,
        bias: bool = True, 
        device=None, 
        dtype=None) -> None:
    """Function to get a Quaternion Transformer.

    Args:
        d_model (int, optional): Dimensionality of the input and output vectors used in each of the transformer’s layers. Defaults to 512.
        nhead (int, optional): The number of heads in the multi-head-attention models (required). Defaults to 8.
        num_encoder_layers (int, optional): The number of sub-encoder-layers in the encoder (required). Defaults to 6.
        num_decoder_layers (int, optional): The number of sub-decoder-layers in the decoder (required). Defaults to 6.
        dim_feedforward (int, optional): The dimension of the feedforward network model (required). Defaults to 2048.
        dropout (float, optional):  The dropout value (required). Defaults to 0.1.
        activation (Union[str, Callable[[Tensor], Tensor]], optional): The activation function of encoder/decoder intermediate layer, can be string or `torch.nn.functional`. Defaults to `F.relu`.
        layer_norm_eps (float, optional): The `eps` in `LayerNorm` (required). Defaults to 1e-5.
        batch_first (bool, optional): If `True`, then the input and output tensors are provided as (batch, seq, feature). Defaults to False.
        norm_first (bool, optional): If `True`, then the normalization is applied before the attention layer. Defaults to False.
        bias (bool, optional): If `True`, then the qtransformer layers will have learnable bias. Defaults to True.
        device (optional): The desired device of the parameters. Defaults to `None`.
        dtype (optional): The desired data type of the parameters. Defaults to `None`.

    Returns:
        `torch.nn.modules.transformer.Transformer`: A custom Quaternion Transformer model.
    """    
    
    factory_kwargs = {'device': device, 'dtype': dtype}
    
    encoder_layer = QTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation, layer_norm_eps, batch_first, norm_first,
                                            bias, **factory_kwargs)
    encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)



    decoder_layer = QTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation, layer_norm_eps, batch_first, norm_first,
                                            bias, **factory_kwargs)
    decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
    decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)


    return torch.nn.modules.transformer.Transformer(
        d_model, nhead, num_encoder_layers,
        num_decoder_layers, dim_feedforward, dropout,
        activation, encoder, decoder,
        layer_norm_eps, batch_first, norm_first,
        bias, device=None, dtype=None
    )


class QTransformerEncoderLayer(torch.nn.modules.transformer.TransformerEncoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        """Quaternion Transformer Encoder Layer.

        Args:
            d_model (int): The number of expected features in the input (required).
            nhead (int): The number of heads in the multiheadattention models (required).
            dim_feedforward (int, optional): The dimension of the feedforward network model (required). Defaults to 2048.
            dropout (float, optional): The dropout value (required). Defaults to 0.1.
            activation (Union[str, Callable[[Tensor], Tensor]], optional): The activation function of encoder/decoder intermediate layer, can be string or `torch.nn.functional`. Defaults to `F.relu`.
            layer_norm_eps (float, optional): The `eps` in `LayerNorm` (required). Defaults to 1e-5.
            batch_first (bool, optional): If `True`, then the input and output tensors are provided as (batch, seq, feature). Defaults to False.
            norm_first (bool, optional):  If `True`, then the normalization is applied before the attention layer. Defaults to False.
            bias (bool, optional): If `True`, then the qtransformer layers will have learnable bias. Defaults to True.
            device (optional): The desired device of the parameters. Defaults to `None`.
            dtype (optional): The desired data type of the parameters. Defaults to `None`.
            
        Returns:
            `torch.nn.modules.transformer.TransformerEncoderLayer`: A custom Quaternion Transformer Encoder Layer.
        """        
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = QMultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation


class QTransformerDecoderLayer(torch.nn.modules.transformer.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        """Quaternion Transformer Decoder Layer.
        
        Args:
            d_model (int): The number of expected features in the input (required).
            nhead (int): The number of heads in the multiheadattention models (required).
            dim_feedforward (int, optional): The dimension of the feedforward network model (required). Defaults to 2048.
            dropout (float, optional): The dropout value (required). Defaults to 0.1.
            activation (Union[str, Callable[[Tensor], Tensor]], optional): The activation function of encoder/decoder intermediate layer, can be string or `torch.nn.functional`. Defaults to `F.relu`.
            layer_norm_eps (float, optional): The `eps` in `LayerNorm` (required). Defaults to 1e-5.
            batch_first (bool, optional): If `True`, then the input and output tensors are provided as (batch, seq, feature). Defaults to False.
            norm_first (bool, optional): If `True`, then the normalization is applied before the attention layer. Defaults to False.
            bias (bool, optional): If `True`, then the qtransformer layers will have learnable bias. Defaults to True.
            device (optional): The desired device of the parameters. Defaults to `None`.
            dtype (optional): The desired data type of the parameters. Defaults to `None`.
            
        Returns:
            `torch.nn.modules.transformer.TransformerDecoderLayer`: A custom Quaternion Transformer Decoder Layer.
        """       
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = QMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn = QMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
