import torch
import torch.utils.data as tud
import tqdm
import logging as l
import datasets as ds
import arguments as ag
import networks as nw
import solvers as sol


class GCN_MHSASolParams(sol.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.GCN_MHSAArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_data(self):
        dataset_params = ds.SyntheticDatasetParams(self.args, self.all_samples)
        dataset = ds.SyntheticDatasetW(dataset_params)
        data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"Loading all data {self.args.txt_indices}...")

        all_data = torch.zeros((self.sample_num, self.args.feature_num), dtype=torch.float32).to(ag.Arguments.device)
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            all_data[i] = datum[0]

        if not self.args.learning_mode:
            self.dataset = ds.CachedBatchDataset(all_data, self.all_samples)
        else:
            self.all_data = all_data

    def _prepare_network(self):
        create_cls_params = nw.GCN_MHSAParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
