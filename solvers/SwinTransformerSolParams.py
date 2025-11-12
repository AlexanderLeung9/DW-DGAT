import torch.nn as nn
import torch.optim as to
import torch.nn.functional as f
import torchvision.models as tvm
import logging as l
import typing as t
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class SwinTransformerSolParams(s.TorchVisionSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: t.Optional[ag.SwinTransformerArgs]):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = SwinTransformerFactory(create_cls_params)


class SwinTransformerFactory(nw.NetworkFactory):
    def create_network(self) -> (nn.Module, [to.Adam], [to.lr_scheduler.MultiStepLR]):
        network = tvm.swin_b()
        network.head = nn.Linear(network.head.in_features, self.create_params.args.class_num)
        network.loss = f.cross_entropy
        optimizer, scheduler = self._create_optimizer_and_scheduler(network.parameters(), self.create_params.args)
        return network, [optimizer], [scheduler] if scheduler is not None else []
