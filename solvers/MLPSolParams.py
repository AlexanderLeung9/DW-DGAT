import torch
import torch.utils.data as tud
import tqdm
import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class MLPSolParams(s.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.MLPArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _prepare_data(self):
        dataset_params = ds.SyntheticDatasetParams(self.args, self.all_samples)
        dataset = ds.SyntheticDatasetW(dataset_params)
        data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)

        print(f"Loading all data {self.args.txt_indices}...")

        all_data = torch.zeros((self.sample_num, self.args.feature_num), dtype=torch.float32)
        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            all_data[i] = datum[0]

        if not self.args.learning_mode:
            self.dataset = ds.CachedBatchDataset(all_data, self.all_samples)
        else:
            self.all_data = all_data.to(ag.Arguments.device)

    def _prepare_network(self):
        create_cls_params = nw.MLPParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
