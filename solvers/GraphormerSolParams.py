import logging as l
import datasets as ds
import arguments as ag
import networks as nw
import solvers as s
import torch
import tqdm
import torch.utils.data as tud


class GraphormerSolParams(s.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.GraphormerArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_data(self):
        dataset_params = ds.SyntheticDatasetParams(self.args, self.all_samples)
        dataset = ds.SyntheticDataset(dataset_params)
        data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)

        print(f"Loading all data {self.args.txt_indices}...")

        edges = torch.zeros((self.sample_num, ag.BDArguments.ROI_NUM, ag.BDArguments.ROI_NUM), dtype=torch.float32).to(ag.Arguments.device)
        nodes = torch.zeros((self.sample_num, ag.BDArguments.ROI_NUM, 3), dtype=torch.float32).to(ag.Arguments.device)

        for i, data in tqdm.tqdm(enumerate(data_loader)):
            datum0 = data[0]
            assert isinstance(datum0, torch.Tensor), "Dismiss a warning."
            networks = datum0[0]
            edges[i] = networks

            node2 = data[2]
            assert isinstance(node2, torch.Tensor), "Dismiss a warning."
            nodes[i] = node2[0]

        dataset = torch.cat([edges, nodes], dim=2)
        self.dataset = tud.TensorDataset(dataset, torch.from_numpy(self.all_labels).long())

    def _prepare_network(self):
        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
