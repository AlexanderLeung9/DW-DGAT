import torch
import torch.nn as nn
import torch.nn.functional as f
import functools as ft
import arguments as ag
import networks as nw


def conv3x3(in_plane_num: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_plane_num, out_planes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False)


def conv1x1(in_plane_num: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_plane_num, out_planes, kernel_size=1, stride=stride, bias=False)


def conv7x7(in_plane_num: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_plane_num, out_planes, kernel_size=7, stride=stride, padding=1, dilation=1, bias=False)


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_plane_num: int, out_planes: int, stride: int = 1, down_sample_func=None):
        super().__init__()

        self.conv1 = conv3x3(in_plane_num, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.down_sample_func = down_sample_func
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample_func is not None:
            residual = self.down_sample_func(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck2D(nn.Module):
    expansion = 4

    def __init__(self, in_plane_num: int, out_planes: int, stride: int = 1, down_sample_func=None):
        super().__init__()

        self.conv1 = conv1x1(in_plane_num, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = conv1x1(out_planes, out_planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample_func = down_sample_func
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample_func is not None:
            residual = self.down_sample_func(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet2D(nw.NetworkBase):
    def __init__(self, model_depth: int, input_dim: int, output_dim: int,
                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                 max_pool: bool = True, shortcut_type: str = 'B', widen_factor: float = 1.0):
        """
        :params shortcut_type: A or B.
        """
        super().__init__(None)

        blocks_in_planes = [64, 128, 256, 512]
        blocks_in_planes = [int(block_n * widen_factor) for block_n in blocks_in_planes]
        self.in_plane_num = blocks_in_planes[0]

        self.conv1 = nn.Conv2d(input_dim, self.in_plane_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_plane_num)
        self.relu = nn.ReLU(inplace=True)
        if max_pool:
            self.max_pool_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool_layer = None
        self.layers = nn.ModuleList()

        if model_depth == 10:
            layers = [1, 1, 1, 1]
        elif model_depth == 18:
            layers = [2, 2, 2, 2]
        elif model_depth == 34:
            layers = [3, 4, 6, 3]
        elif model_depth == 50:
            layers = [3, 4, 6, 3]
        elif model_depth == 101:
            layers = [3, 4, 23, 3]
        elif model_depth == 152:
            layers = [3, 8, 36, 3]
        elif model_depth == 200:
            layers = [3, 24, 36, 3]
        else:
            raise NotImplementedError(f"model_depth={model_depth}")
        block = BasicBlock2D if model_depth < 50 else Bottleneck2D
        
        for i in range(len(layers)):
            stride = 1 if i == 0 else 2
            self.layers.append(self.__make_layers(block, blocks_in_planes[i], layers[i], shortcut_type, stride))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(blocks_in_planes[-1] * block.expansion, output_dim)

    def __make_layers(self, block_ctor, plane_num: int, layer_num: int, shortcut_type: str, stride: int = 1):
        down_sample_func = None
        expanding_plane_num = plane_num * block_ctor.expansion
        if stride != 1 or self.in_plane_num != expanding_plane_num:
            if shortcut_type == 'A':
                down_sample_func = ft.partial(ResNet2D.__down_sample_basic_block, plane_num=expanding_plane_num, stride=stride)
            else:
                down_sample_func = nn.Sequential(
                    conv1x1(self.in_plane_num, expanding_plane_num, stride),
                    nn.BatchNorm2d(expanding_plane_num))

        layers = [block_ctor(in_plane_num=self.in_plane_num, out_planes=plane_num, stride=stride, down_sample_func=down_sample_func)]
        self.in_plane_num = expanding_plane_num
        for i in range(1, layer_num):
            layers.append(block_ctor(self.in_plane_num, plane_num))

        return nn.Sequential(*layers)

    @staticmethod
    def __down_sample_basic_block(x, plane_num: int, stride: int):
        out = f.avg_pool2d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros((out.size(0), plane_num - out.size(1), out.size(2), out.size(3)), dtype=torch.float).to(ag.Arguments.device)

        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def forward(self, x: [[[float]]]) -> [[float]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.max_pool_layer is not None:
            x = self.max_pool_layer(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)
        scores = self.fc(x)
        return scores
