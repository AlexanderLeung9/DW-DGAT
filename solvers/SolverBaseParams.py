import os.path
import numpy as np
import pickle
import torch
import torch.utils.data as tud
import typing as t
import logging as l
import sklearn.model_selection as sms
import networks as nw
import arguments as ag
import datasets as ds


class SolverBaseParams(object):
    create_cls_factory: nw.NetworkFactory
    all_labels: np.ndarray
    
    def __init__(self, all_samples: [ds.SampleBase], logger: l.Logger, args: ag.Arguments):
        assert isinstance(all_samples, np.ndarray)
        self.all_samples: [ds.SampleBase] = all_samples
        self.logger: l.Logger = logger
        self.args: ag.Arguments = args
        self.__has_loaded_data: bool = False
        # a sequence index
        self.test_time: int = -1

        self.create_cls_factory: nw.NetworkFactory
        self.create_gen_factory: t.Optional[nw.NetworkFactory] = None
        # It's used for splitting training set, validation set, and test set.
        self.all_labels: np.ndarray
        self.train_vldtn_indices: t.Optional[np.ndarray] = None
        self.test_indices: t.Optional[np.ndarray] = None

        self.dataset: t.Optional[tud.Dataset] = None
        self.all_data: t.Optional[torch.Tensor] = None
        self.gen_scores: [[float]] = None
        self.solver_name: str = "BatchSolver" if not args.learning_mode else "WholeSolver"

    def initialize(self):
        self._initialize()
        self._prepare_data()
        self._split_train_and_test_parts()
        self._prepare_network()
        self._summary_networks()

    def _initialize(self):
        self.all_labels = np.array([sample.label for sample in self.all_samples])
        if not self.args.learning_mode and self.args.batch_size == 0:
            self.args.batch_size = len(self.all_samples)

    @property
    def sample_num(self) -> int:
        """
        Allow the label set is larger or smaller than the sample set.
        """
        N = self.all_samples.shape[0]
        return N

    def _prepare_data(self):
        raise NotImplementedError

    def _prepare_network(self):
        raise NotImplementedError

    def _summary_networks(self):
        if self.logger is None:
            return

        if self.all_data is not None:
            all_data = self.all_data
        else:
            if isinstance(self.dataset, ds.CachedBatchDataset):
                all_data = self.dataset.all_data
            else:
                print("No data to summarize the network.")
                return

        network, _, _ = self.create_cls_factory.create_network()
        network.to(ag.Arguments.device)
        shape = list(all_data.shape)
        N = shape[0]
        if self.args.vldtn_ratio_OR_k_fold < 1:
            N = int(N * (1 - self.args.vldtn_ratio_OR_k_fold))
        elif self.args.vldtn_ratio_OR_k_fold > 1:
            N = N // self.args.vldtn_ratio_OR_k_fold * (self.args.vldtn_ratio_OR_k_fold - 1)

        if isinstance(network, nw.NetworkBase):
            network.current_indices = list(range(N))
            if not self.args.learning_mode:
                network.batch_index = 0

        if self.args.cls_gen_train_ratio != 0:
            generator, _, _ = self.create_gen_factory.create_network()
            generator.to(ag.Arguments.device)
            if isinstance(generator, nw.NetworkBase):
                generator.current_indices = list(range(N))
                if not self.args.learning_mode:
                    generator.batch_index = 0
        else:
            generator = None

        if not self.args.learning_mode:
            shape[0] = self.args.batch_size

        try:
            import torchinfo as ti
            network_states = ti.summary(network, input_size=shape, col_names=["num_params"], device=ag.Arguments.device)
            self.logger.info("network:\n" + str(network_states))

            if generator is not None:
                generator_states = ti.summary(generator, input_size=shape, col_names=["num_params"], device=ag.Arguments.device)
                self.logger.info("generator:\n" + str(generator_states))

        except (ModuleNotFoundError, RuntimeError) as ex:
            self.logger.error(f"_summary_networks: {ex}")

            import thop
            flops, params = thop.profile(network, inputs=(torch.randn(*shape).to(ag.BDArguments.device),))
            flops, params = thop.clever_format([flops, params])
            self.logger.info("Network Params: {}B; FLOPs: {}.".format(params, flops))

            if generator is not None:
                flops, params = thop.profile(generator, inputs=(torch.randn(*shape).to(ag.BDArguments.device),))
                flops, params = thop.clever_format([flops, params])
                self.logger.info("Generator Params: {}B; FLOPs: {}.".format(params, flops))

    def __train_test_split_by_classes(self, labels: [int], shuffle: bool = True) -> ([int], [int]):
        class_indices = []
        for i in range(self.args.class_num):
            class_indices.append([])

        for j in range(len(labels)):
            for i in range(self.args.class_num):
                if labels[j] == i:
                    class_indices[i].append(j)
                    break

        train_indices = []
        test_indices = []
        train_num = 0
        test_num = 0
        for i in range(self.args.class_num):
            train_indices_i, test_indices_i = sms.train_test_split(class_indices[i], test_size=self.args.test_percent, shuffle=shuffle)
            train_indices.append(train_indices_i)
            test_indices.append(test_indices_i)
            train_num += len(train_indices_i)
            test_num += len(test_indices_i)

        train_indices = np.concatenate(train_indices, dtype=np.int32)
        test_indices = np.concatenate(test_indices, dtype=np.int32)

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        return train_indices, test_indices

    def _split_train_and_test_parts(self):
        """
        Split out a test set.
        """
        if self.args.test_percent is not None:
            shuffle = self.args.test_times > 1 and not self.__has_loaded_data
            if self.args.multi_samples:
                subjects_indices, repeated_indices = self.__get_subjects_and_repeated_indices()
                subjects_labels = [self.all_labels[i] for i in subjects_indices]
                train_vldtn_indices, test_indices = self.__train_test_split_by_classes(subjects_labels, shuffle)

                train_vldtn_indices_temp = [subjects_indices[i] for i in train_vldtn_indices]
                test_indices_temp = [subjects_indices[i] for i in test_indices]
                self.train_vldtn_indices, self.test_indices = self.__get_train_test_indices(
                    repeated_indices, train_vldtn_indices_temp, test_indices_temp)
            else:
                self.train_vldtn_indices, self.test_indices = self.__train_test_split_by_classes(self.all_labels, shuffle)
        else:
            if self.args.vldtn_ratio_OR_k_fold == 1:
                self.train_vldtn_indices = None
                self.test_indices = np.array(range(self.sample_num), dtype=np.int32)
            else:
                self.train_vldtn_indices = np.array(range(self.sample_num), dtype=np.int32)
                self.test_indices = None

    def __get_subjects_and_repeated_indices(self):
        """
        Create an empty set to store unrepeated Nos.
        """
        unique_nos = set()
        subjects_indices = []
        repeated_indices = []
        for index, subject in enumerate(self.all_samples):
            if subject.No in unique_nos:
                repeated_indices.append(index)
                continue
            unique_nos.add(subject.No)
            subjects_indices.append(index)
        return subjects_indices, repeated_indices

    def __get_train_test_indices(self, repeated_indices, train_vldtn_indices, test_indices):
        train_val_set = set([self.all_samples[i].No for i in train_vldtn_indices])
        test_set = set([self.all_samples[i].No for i in test_indices])
        for i in repeated_indices:
            if self.all_samples[i].No in test_set:
                test_indices.append(i)
            if self.all_samples[i].No in train_val_set:
                train_vldtn_indices.append(i)
        return train_vldtn_indices, test_indices

    def _store_or_load_data(self) -> t.Union[np.ndarray, torch.Tensor]:
        """
        It calls _preprocess_data() inside.
        """
        if self.args.load_OR_save_data is None:
            file_path = os.path.join(self.args.preprocessed_data_dir, self.args.data_file_name)
            data_existed = os.path.exists(file_path)
            
            if data_existed:
                with open(file_path, "rb") as f:
                    preprocessed_data = pickle.load(f)
                self.__has_loaded_data = True
            else:
                preprocessed_data = self._preprocess_data()
    
                with open(file_path, "wb") as f:
                    pickle.dump(preprocessed_data, f, pickle.HIGHEST_PROTOCOL)
        else:
            preprocessed_data = self._preprocess_data()
            
            if self.args.load_OR_save_data:
                file_path = os.path.join(self.args.preprocessed_data_dir, self.args.data_file_name)
                with open(file_path, "wb") as f:
                    pickle.dump(preprocessed_data, f, pickle.HIGHEST_PROTOCOL)

        return preprocessed_data

    def _preprocess_data(self) -> t.Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError("A virtual method.")
