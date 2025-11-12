import tqdm
import torch
import torch.utils.data as tud
import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class NetworkGATSolParams(s.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.NetworkGATArgs):
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

            graph = (FA_network, FA_weights)

            all_data.append(graph)

        self.dataset = ds.CachedBatchDataset(all_data, self.all_samples)

    def _prepare_network(self):
        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
