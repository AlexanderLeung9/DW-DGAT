import torch
import torch.nn.functional as F
import networks as nw
import arguments as ag


class E2EBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, bias=False):
        super().__init__()
        self.d = ag.BDArguments.ROI_NUM
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        c = torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)
        return c


class BrainNetCNN(nw.NetworkBase):
    """
    @article{kawahara2017brainnetcnn,
      title={BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment},
      author={Kawahara, Jeremy and Brown, Colin J and Miller, Steven P and Booth, Brian G and Chau, Vann and Grunau, Ruth E and Zwicker, Jill G and Hamarneh, Ghassan},
      journal={NeuroImage},
      volume={146},
      pages={1038--1049},
      year={2017},
      publisher={Elsevier}
    }
    """
    def __init__(self, params: nw.NetworkParams):
        super().__init__(params)
        self.e2e_conv1 = E2EBlock(params.args.feature_num, 32, True)
        self.e2e_conv2 = E2EBlock(32, 64, True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, ag.BDArguments.ROI_NUM))
        self.N2G = torch.nn.Conv2d(1, 256, (ag.BDArguments.ROI_NUM, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, params.args.class_num)

    def forward(self, x):
        out = F.leaky_relu(self.e2e_conv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2e_conv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)

        return out
