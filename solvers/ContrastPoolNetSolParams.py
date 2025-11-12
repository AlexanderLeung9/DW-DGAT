import torch
import torch.utils.data as tud
# import torch.nn as nn
# import torch.optim as to
# import typing as t
import tqdm
import logging as l
import dgl
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class ContrastPoolNetSolParams(s.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.ContrastPoolNetArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_data(self):
        dataset_params = ds.SyntheticDatasetParams(self.args, self.all_samples)
        dataset = ds.SyntheticDataset(dataset_params)
        data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"Loading all data {self.args.txt_indices}...")

        all_data = []
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            datum0 = data[0]
            assert isinstance(datum0, torch.Tensor), "Dismiss a warning."
            FA_network = datum0[0][0].to(ag.Arguments.device)
            FA_weights = data[2][0].to(ag.Arguments.device)

            src, dst = FA_network.nonzero(as_tuple=True)
            graph = dgl.graph((src, dst), num_nodes=ag.BDArguments.ROI_NUM)
            graph.ndata["feat"] = FA_weights
            all_data.append(graph)

        self.dataset = ds.CachedBatchDataset(all_data, self.all_samples)

    def _prepare_network(self):
        create_cls_params = nw.ContrastPoolNetParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)


# class ContrastPoolNetFactory(nw.NetworkFactory):
#     def __init__(self, create_params: nw.ContrastPoolNetParams, network_name: t.Optional[str] = None):
#         super().__init__(create_params, network_name)
#         self.create_params = create_params
#
#     def create_network(self) -> (nn.Module, [to.Adam], [to.lr_scheduler.MultiStepLR]):
#         network = nw.ContrastPoolNet(self.create_params)
#         optimizer = to.Adam(network.parameters(), lr=self.create_params.args.init_lr, weight_decay=self.create_params.args.weight_decay)
#         scheduler = to.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=self.create_params.args.lr_reduce_factor, patience=self.create_params.args.lr_schedule_patience, verbose=True)
