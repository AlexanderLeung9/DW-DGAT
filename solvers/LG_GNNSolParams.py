import torch
import torch.utils.data as tud
import tqdm
import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as s
import numpy as np
import enums as es
import utils.GraphUtils as gu


class LG_GNNSolParams(s.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.LG_GNNArgs):
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
            FA_network = datum0[0][0]
            FA_weights = data[2][0].to(ag.Arguments.device)
            edge_indices, edge_weights = gu.get_edge_indices_and_weights(FA_network)
            graph = (FA_weights, edge_indices.to(ag.Arguments.device), edge_weights.to(ag.Arguments.device))
            all_data.append(graph)

        if not self.args.learning_mode:
            self.dataset = ds.CachedBatchDataset(all_data, self.all_samples)
        else:
            self.all_data = all_data

    def _prepare_network(self):
        vectors = []
        for sample in self.all_samples:
            vector = sample.to_vector()
            vectors.append(vector)
        non_img = np.array(vectors)

        phonetic_score = {}
        if self.args.business == es.EBusiness.PD:
            phonetic = ds.PDSample.field_names()
        elif self.args.business == es.EBusiness.AD:
            phonetic = ds.ADSample.field_names()
        else:
            raise NotImplementedError(f"business={self.args.business}")

        for i in range(len(phonetic)):
            p = phonetic[i]
            phonetic_score[p] = np.copy(non_img[:, i])

        create_cls_params = nw.LG_GNNParams(non_img, phonetic_score, self.args)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params)
