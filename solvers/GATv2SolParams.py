import logging as l
import arguments as ag
import solvers as s
import datasets as ds


class GATv2SolParams(s.GATSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.GATv2Args):
        super().__init__(all_samples, logger, args)
