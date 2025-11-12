import torch
import torch_geometric.data as tgd
import numpy as np
import logging as l
import enums as es
import arguments as ag
import utils.GraphUtils as gu
import utils.BDGraphUtils as pu
import networks as nw
import solvers as s
import datasets as ds


class GATSolParams(s.MLPSolParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.GATArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_network(self):
        create_cls_params = nw.GCNParams(self.args, self.all_samples)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
