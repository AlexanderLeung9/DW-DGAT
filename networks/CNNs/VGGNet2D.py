import torch.nn as nn


class VGGNet2D(nn.Module):
    def __init__(self, class_num: int, input_channel_num: int, output_channel_num: int,
                 layer_num: int, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        if layer_num == 16:
            block_nums = [2, 2, 3, 3, 3]
        elif layer_num == 19:
            block_nums = [2, 2, 4, 4, 4]
        else:
            raise NotImplementedError(f"layer_num={layer_num}")

        self.stage1 = VGGNet2D.__make_layers(input_channel_num, 64, block_nums[0], kernel_size, stride, padding)
        self.stage2 = VGGNet2D.__make_layers(in_channels=64, out_channels=128, block_num=block_nums[1])
        self.stage3 = VGGNet2D.__make_layers(in_channels=128, out_channels=256, block_num=block_nums[2])
        self.stage4 = VGGNet2D.__make_layers(in_channels=256, out_channels=512, block_num=block_nums[3])
        self.stage5 = VGGNet2D.__make_layers(in_channels=512, out_channels=512, block_num=block_nums[4])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=output_channel_num, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=4096, out_features=class_num)
        )

        self._init_params()

    @staticmethod
    def __make_layers(in_channels, out_channels, block_num, kernel_size=3, stride=1, padding=1):
        layers = [VGGNet2D.__conv3x3BNReLU(in_channels, out_channels, kernel_size, stride, padding)]
        for i in range(1, block_num):
            layers.append(VGGNet2D.__conv3x3BNReLU(out_channels, out_channels))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False))
        return nn.Sequential(*layers)

    @staticmethod
    def __conv3x3BNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        return sequential

    def forward(self, x) -> [[float]]:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        scores = self.classifier(x)
        return scores
