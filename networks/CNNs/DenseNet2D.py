import math
import torch
import torch.nn as nn
import torch.nn.functional as f


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = f.dropout(out, p=self.dropout_rate, training=self.training)

        # Dense connection.
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = f.dropout(out, p=self.dropout_rate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.dropout_rate > 0:
            out = f.dropout(out, p=self.dropout_rate, inplace=False, training=self.training)

        # Dense connection.
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = f.dropout(out, p=self.dropout_rate, inplace=False, training=self.training)
        return f.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropout_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = DenseBlock.__make_layer(block, in_planes, growth_rate, nb_layers, dropout_rate)

    @staticmethod
    def __make_layer(block, in_planes, growth_rate, nb_layers, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet2D(nn.Module):
    def __init__(self, n_input_channels: int, n_output_channels: int, num_classes, depth, growth_rate=12, reduction=0.5,
                 bottleneck=True, dropout_rate=0.0):
        """
        Refer to paper "Densely Connected Convolutional Networks".
        :param num_classes:
        :param depth: total number of layers: 40, 100, 250, 190
        :param growth_rate: `k` in the paper: 12, 24, 40
        :param reduction: for compression, `θ` in the paper: (0, 1]
        :param bottleneck: the concept in paper "ResNet"; it's especially effective for DenseNet
        :param dropout_rate: it is set as 0.2 in the SVHN task in the paper.
        """
        super(DenseNet2D, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(n_input_channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropout_rate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropout_rate=dropout_rate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropout_rate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropout_rate=dropout_rate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropout_rate)

        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_output_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x: []) -> [[float]]:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = f.avg_pool2d(out, 7)
        n = out.size(0)
        out = out.view(n, -1)

        scores = self.fc(out)
        return scores
