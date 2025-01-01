from typing import Sequence

import torch.nn as nn


def make_linear_layer(in_features: int, out_features: int, use_bias=True):
    """
    Create a linear layer
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    return fc_layer


def make_mlp_layer(feature_dims: Sequence[int] = None,
                   activation: nn.Module = nn.ELU,
                   activation_last: bool = True):
    """
        Create a MLP layer
    """
    # assert len(feature_dims) >= 2
    fc_layers = nn.Sequential()
    for i in range(len(feature_dims) - 1):
        input_dim = feature_dims[i]
        output_dim = feature_dims[i + 1]
        fc_layers.append(make_linear_layer(input_dim, output_dim))
        if i + 1 < len(feature_dims) - 1 or activation_last:
            fc_layers.append(activation())

    for m in fc_layers.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # 使用默认 ELU 初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return fc_layers


def make_action_heads(action_dims: Sequence[int] = None, feature_dims: Sequence[int] = None):
    action_heads = nn.ModuleList()
    for action_dim in action_dims:
        action_heads.append(
            make_mlp_layer(feature_dims + [action_dim], activation_last=False)
        )
    return action_heads


# def make_conv2d_layer(in_channels: int,
#                       out_channels: int,
#                       kernel_size: Tuple[int, int],
#                       stride: Tuple[int, int] = (1, 1),
#                       padding: Sequence[int] = 0,
#                       ):
#     conv_layer = nn.Conv2d(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         # dilation=1,
#         # groups=1,
#         # bias=True,
#     )
#     return conv_layer


if __name__ == '__main__':
    net = make_mlp_layer([1344, 2048, 768, 512, 512], activation_last=False)
    print(net)
