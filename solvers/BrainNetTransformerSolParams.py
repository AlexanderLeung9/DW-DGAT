import logging as l
import typing as t
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class BrainNetTransformerSolParams(s.TorchVisionSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: t.Optional[ag.BrainNetTransformerArgs]):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        assert isinstance(self.dataset, ds.CachedBatchDataset)
        length = len(self.dataset)
        iterations = length // self.args.batch_size
        if length % self.args.batch_size != 0:
            iterations += 1
        self.args.scheduler.T_max = self.args.stop_value * iterations

        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
