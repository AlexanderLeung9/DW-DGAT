import logging as l
import typing as t
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class ChAdaViTSolParams(s.TorchVisionSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: t.Optional[ag.ChAdaViTArgs]):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
