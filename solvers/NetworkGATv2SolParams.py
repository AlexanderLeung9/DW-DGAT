import logging as l
import arguments as ag
import solvers as sol
import datasets as ds


class NetworkGATv2SolParams(sol.NetworkGATSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.NetworkGATv2Args):
        super().__init__(all_samples, logger, args)
