import torch.nn as nn
import torch.nn.functional as f
import torch.optim as to
import timm
import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as sol


class ViTSolParams(sol.TorchVisionSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.ViTArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = ViTFactory(create_cls_params)


class ViTFactory(nw.NetworkFactory):
    def create_network(self) -> (nn.Module, [to.Adam], [to.lr_scheduler.MultiStepLR]):
        network = timm.create_model("vit_small_patch16_224",
                                    img_size=ag.BDArguments.ROI_NUM,
                                    patch_size=10,
                                    num_classes=self.create_params.args.class_num)
        network.loss = f.cross_entropy
        optimizer, scheduler = self._create_optimizer_and_scheduler(network.parameters(), self.create_params.args)
        return network, [optimizer], [scheduler] if scheduler is not None else []
