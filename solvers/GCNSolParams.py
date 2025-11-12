import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class GCNSolParams(s.MLPSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.GCNArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        create_cls_params = nw.GCNParams(self.args, self.all_samples)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
