import sys
import enums as es
import typing as t
import tqdm
import numpy as np
import torch
import torch.utils.data as tud
import logging as l
import arguments as ag
import datasets as ds
import networks as nw
import solvers as sol


class DW_DGATSolParams(sol.SolverBaseParams):
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.DW_DGATArgs):
        super().__init__(all_samples, logger, args)
        self.args: ag.DW_DGATArgs = args
        self.analyze_ROI_sequence: bool = False

    def _preprocess_data(self) -> t.Union[np.ndarray, torch.Tensor]:
        if self.analyze_ROI_sequence:
            dataset_params = ds.MotifDatasetParams(
                self.args.dataset_dir, self.all_samples, self.args.txt_indices, self.args.center_num,
                self.args.motif_nums, self.args.motif_len, es.EMotifSimilarity.Gaussian, None)

            dataset = ds.ROIsDataset(dataset_params)
            data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)

            ROIs = [[], [], []]
            for i, (data, label) in tqdm.tqdm(enumerate(data_loader)):
                assert isinstance(data, torch.Tensor), "Dismiss a warning."
                datum = data[0]
                label = label.cpu().item()
                ROIs[label].append(datum)

            avg_ROIs = []
            for i in range(len(ROIs)):
                class_data = torch.stack(ROIs[i], dim=0)
                mean_data = torch.mean(class_data, dim=0)
                avg_ROIs.append(mean_data.cpu().numpy())

            for i, mean_ROIs in enumerate(avg_ROIs):
                ROI_sequence = ds.ROISequenceDataset.get_pooled_ROI_sequence(mean_ROIs, self.args.center_num)
                print(f"{i}: {ROI_sequence}")
            sys.exit(0)

        if self.args.single_graph_module.name.startswith("vit_"):
            dataset_params = ds.ROISequenceDatasetParams(self.all_samples, self.args)

            dataset = ds.ROISequenceDataset(dataset_params)
            data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)

            all_data = torch.zeros((self.sample_num, ag.BDArguments.ROI_NUM, self.args.feature_num), dtype=torch.float)
        else:
            dataset_params = ds.MotifDatasetParams(
                self.args.dataset_dir, self.all_samples, self.args.txt_indices, self.args.center_num,
                self.args.motif_nums, self.args.motif_len, es.EMotifSimilarity.Gaussian, None)

            dataset = ds.MotifDataset(dataset_params)
            data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)

            col_num = sum(self.args.motif_nums) * self.args.motif_len
            all_data = torch.zeros((self.sample_num, self.args.feature_num, self.args.center_num, col_num), dtype=torch.float)

        for i, datum in tqdm.tqdm(enumerate(data_loader)):
            all_data[i] = datum[0]

        return all_data

    def _prepare_data(self):
        print(f"Loading all data {self.args.txt_indices}...")
        assert isinstance(self.args.single_graph_module, ag.ESingleGraphModule)

        if self.args.single_graph_module == ag.ESingleGraphModule.none:
            dataset_params = ds.SyntheticDatasetParams(self.args, self.all_samples)
            dataset = ds.SyntheticDatasetW(dataset_params)

            self.all_data = torch.zeros((self.sample_num, self.args.feature_num), dtype=torch.float).to(ag.Arguments.device)
            data_loader = tud.DataLoader(dataset, batch_size=1, shuffle=False)
            for i, datum in tqdm.tqdm(enumerate(data_loader)):
                self.all_data[i] = datum[0]

        else:
            self.all_data = self._store_or_load_data().to(ag.Arguments.device)

        if not self.args.learning_mode:
            self.dataset = ds.CachedBatchDataset(self.all_data, self.all_samples)

    def _prepare_network(self):
        create_cls_params = nw.DW_DGATParams(self.args, self.all_samples)
        self.create_cls_factory = nw.NetworkFactory(create_cls_params, self.args.net_name)

        create_gen_params = create_cls_params
        self.create_gen_factory = nw.NetworkFactory(create_gen_params, self.args.net_name + "WeightGenerator")
