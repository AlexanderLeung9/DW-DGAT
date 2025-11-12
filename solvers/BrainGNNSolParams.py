import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class BrainGNNSolParams(s.LG_GNNSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.BrainGNNArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
