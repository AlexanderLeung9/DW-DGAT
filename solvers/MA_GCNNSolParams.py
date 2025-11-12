import typing as t
import tqdm
import numpy as np
import torch
import torch.utils.data as tud
import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s


class MA_GCNNSolParams(s.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.MA_GCNNArgs):
        super().__init__(all_samples, logger, args)
        self.args = args

    def _preprocess_data(self) -> t.Union[np.ndarray, torch.Tensor]:
        dataset_params = ds.MotifDatasetParams(
            self.args.dataset_dir, self.all_samples, self.args.txt_indices, self.args.center_num,
            self.args.slot_nums, self.args.motif_len, self.args.motif_similarity)
        dataset = ds.MotifDataset(dataset_params)

        dataloader = tud.DataLoader(dataset, batch_size=1, shuffle=False)
        col_num = sum(self.args.slot_nums) * self.args.motif_len
        all_data = torch.zeros((self.sample_num, self.args.feature_num, self.args.center_num, col_num), dtype=torch.float)
        print(f"Loading all data {self.args.txt_indices}...")
        for i, datum in tqdm.tqdm(enumerate(dataloader)):
            all_data[i] = datum[0]

        return all_data

    def _prepare_data(self):
        all_data = self._store_or_load_data()
        self.dataset = ds.CachedBatchDataset(all_data, self.all_samples)

    def _prepare_network(self):
        create_cls_params = nw.NetworkParams(self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
