import abc
import sys
import select
import os.path
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim.optimizer as too
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import sklearn.metrics as skm
import sklearn.model_selection as sms
import sklearn.manifold as slm
import typing as t
import logging as l
import warnings
import utils.CommonUtils as cu
import programs.JobManager as jm
import networks as nw
import solvers as s
import enums as es
import arguments as ag

l.getLogger("matplotlib").setLevel(l.WARNING)
warnings.filterwarnings("ignore")


class SolverBase(object):
    def __init__(self, params: s.SolverBaseParams):
        self.__best_folds = []
        self.__worst_folds = []

        self.args: ag.Arguments = params.args
        self.logger: l.Logger = params.logger
        self.params = params

        if params.args.vldtn_ratio_OR_k_fold > 1:
            k_folds = int(params.args.vldtn_ratio_OR_k_fold)
            self.skf = sms.StratifiedKFold(k_folds, shuffle=False)
            self.loss_histories = [[]] * k_folds
            self.gen_loss_histories = [[]] * k_folds
            if self.args.check_training:
                self.train_acc_histories = [[]] * k_folds
            else:
                self.train_acc_histories = []
            if self.args.check_validation:
                self.val_acc_histories = [[]] * k_folds
            else:
                self.val_acc_histories = []
            self.min_losses = np.full(k_folds, np.inf, dtype=np.float32)
            self.max_accuracies = np.full(k_folds, -1.0, dtype=np.float32)
            self.best_states = [{}] * k_folds
            self.best_epochs = [0] * k_folds
            self.history_split_epochs = [0] * k_folds
            self.vldtn_label_indices = [np.array([])] * k_folds
            self.train_times = [0] * k_folds

            self.ground_truth_labels = [np.array([])] * k_folds
            self.predicted_labels = [np.array([])] * k_folds
            self.predicted_scores = [np.array([[]])] * k_folds
        else:
            self.skf = None
            self.loss_histories = [[]]
            self.gen_loss_histories = [[]]
            if self.args.check_training:
                self.train_acc_histories = [[]]
            else:
                self.train_acc_histories = []
            if self.args.check_validation:
                self.val_acc_histories = [[]]
            else:
                self.val_acc_histories = []
            self.min_losses = np.array([np.inf], dtype=np.float32)
            self.max_accuracies = np.array([-1.0], dtype=np.float32)
            self.best_states = [{}]
            self.best_epochs = [0]
            self.history_split_epochs = [0]
            self.vldtn_label_indices = [np.array([])]
            self.train_times = [0]

            self.ground_truth_labels = [np.array([])]
            self.predicted_labels = [np.array([])]
            self.predicted_scores = [np.array([[]])]

    def clear_test_results(self):
        k_folds = int(self.args.vldtn_ratio_OR_k_fold)
        # Cooperate with __concat_arrays(), don't initialize with np.ndarray.
        self.ground_truth_labels = [[]] * k_folds
        self.predicted_labels = [[]] * k_folds
        self.predicted_scores = [[[]]] * k_folds

    def __get_subjects_and_repeated_indices(self):
        train_vldtn_indices = self.params.train_vldtn_indices
        all_train_val_samples = self.params.all_samples[train_vldtn_indices]
        subjects_indices = []
        subjects_to_train_vldtn_indices = []
        repeated_indices = []
        repeated_to_train_vldtn_indices = []
        unique_Nos = set()

        # Iterate the sample list to add every sample's No to unique_Nos.
        for index, sample in enumerate(all_train_val_samples):
            if sample.No in unique_Nos:
                repeated_indices.append(train_vldtn_indices[index])
                repeated_to_train_vldtn_indices.append(index)
                continue
            unique_Nos.add(sample.No)
            subjects_indices.append(train_vldtn_indices[index])
            subjects_to_train_vldtn_indices.append(index)
        return subjects_indices, repeated_indices, subjects_to_train_vldtn_indices, repeated_to_train_vldtn_indices

    def __get_vldtn_true_indices(self, subjects_indices, repeated_indices, train_indices, vldtn_indices,
                                 subjects_to_train_vldtn_indices, repeated_to_train_vldtn_indices):
        if self.skf is not None:
            vldtn_true_indices = [subjects_indices[i] for i in vldtn_indices]
            train_indices = [subjects_to_train_vldtn_indices[i] for i in train_indices]
            vldtn_indices = [subjects_to_train_vldtn_indices[i] for i in vldtn_indices]
        else:
            vldtn_true_indices = vldtn_indices

        vldtn_set = set([self.params.all_samples[i].No for i in vldtn_true_indices])
        for index, i in enumerate(repeated_indices):
            if self.params.all_samples[i].No in vldtn_set:
                if self.skf is not None:
                    vldtn_true_indices.append(i)
                    vldtn_indices.append(repeated_to_train_vldtn_indices[index])
                else:
                    vldtn_indices.append(i)
            else:
                if self.skf is not None:
                    train_indices.append(repeated_to_train_vldtn_indices[index])
                else:
                    train_indices.append(i)

        return vldtn_true_indices, train_indices, vldtn_indices

    def train(self) -> bool:
        result = False
        if not self.args.multi_samples:
            if self.skf is not None:
                train_vldtn_labels = self.params.all_labels[self.params.train_vldtn_indices]
                for fold, (train_indices, vldtn_indices) in enumerate(
                        self.skf.split(self.params.train_vldtn_indices, train_vldtn_labels)):
                    self.vldtn_label_indices[fold] = self.params.train_vldtn_indices[vldtn_indices]
                    result = self.__train(fold, train_indices, vldtn_indices)
                    if not result:
                        break
            else:
                if self.args.check_validation:
                    train_indices, vldtn_indices = sms.train_test_split(
                        self.params.train_vldtn_indices, test_size=self.args.vldtn_ratio_OR_k_fold,
                        shuffle=self.args.test_percent is None)
                    self.vldtn_label_indices[0] = vldtn_indices
                else:
                    train_indices = self.params.train_vldtn_indices
                    vldtn_indices = None

                result = self.__train(0, train_indices, vldtn_indices)
        else:
            if self.skf is not None:
                # train = []
                # val = []
                subjects_indices, repeated_indices, subjects_to_train_vldtn_indices, repeated_to_train_vldtn_indices \
                    = self.__get_subjects_and_repeated_indices()
                distinct_labels = np.array([self.params.all_samples[i].label for i in subjects_indices])

                for fold, (train_indices, vldtn_indices) in enumerate(self.skf.split(subjects_indices, distinct_labels)):
                    vldtn_true_indices, train_indices, vldtn_indices = self.__get_vldtn_true_indices(
                        subjects_indices, repeated_indices, train_indices, vldtn_indices,
                        subjects_to_train_vldtn_indices, repeated_to_train_vldtn_indices)
                    self.vldtn_label_indices[fold] = np.array(vldtn_true_indices)
                    # train.append(self.params.train_vldtn_indices[train_indices])
                    # val.append(self.params.train_vldtn_indices[vldtn_indices])
                    result = self.__train(fold, train_indices, vldtn_indices)
                    if not result:
                        break
                # region Test
                # for i in range(int(self.args.vldtn_ratio_OR_k_fold)):
                #     print(f"train[{i}]={train[i]}")
                #     print(f"val[{i}]={val[i]}")
                #     set1 = set(train[i])
                #     set2 = set(val[i])
                #     if set1 & set2:
                #         print("tain & val error")
                # length = 0
                # for i in range(int(self.args.vldtn_ratio_OR_k_fold)):
                #     length += len(val[i])
                #     j = i + 1
                #     while j < int(self.args.vldtn_ratio_OR_k_fold):
                #         set1 = set(val[i])
                #         set2 = set(val[j])
                #         if set1 & set2:
                #             print("val1 & val2 error")
                #         j = j + 1
                # print(length)
                # sys.exit(1)
                # endregion
            else:
                if self.args.check_validation:
                    subjects_indices, repeated_indices, subjects_to_train_vldtn_indices, repeated_to_train_vldtn_indices = self.__get_subjects_and_repeated_indices()

                    train_indices, vldtn_indices = sms.train_test_split(
                        subjects_indices, test_size=self.args.vldtn_ratio_OR_k_fold,
                        shuffle=self.args.test_percent is None)
                    vldtn_true_indices, train_indices, vldtn_indices = self.__get_vldtn_true_indices(
                        subjects_indices, repeated_indices, train_indices, vldtn_indices,
                        subjects_to_train_vldtn_indices, repeated_to_train_vldtn_indices)
                    self.vldtn_label_indices[0] = np.array(vldtn_true_indices)
                else:
                    train_indices = self.params.train_vldtn_indices
                    vldtn_indices = None

                result = self.__train(0, train_indices, vldtn_indices)

        return result

    def __train(self, fold: int, train_indices: [int], vldtn_indices: [int]) -> bool:
        try:
            train_times = 0
            while train_times < self.args.max_train_times:
                train_times += 1
                self.logger.info(f"{cu.get_sequence_no(train_times)} training for fold {fold + 1}...")
                tValdt_indices = None if vldtn_indices is None else self.params.train_vldtn_indices[vldtn_indices]

                self.min_losses[fold] = np.inf
                self.max_accuracies[fold] = 0.0
                self.best_states[fold] = {}
                self.best_epochs[fold] = 0

                success = self._one_fold_train(fold, self.params.train_vldtn_indices[train_indices], tValdt_indices)
                if success:
                    break

            if train_times == self.args.max_train_times and not self.args.NaN_error_continue:
                assert train_times < self.args.max_train_times, f"train times can not surpasses {self.args.max_train_times}!"
            self.train_times[fold] = train_times

            max_accuracy = self.max_accuracies[fold]
            min_loss = self.min_losses[fold]
            strFold = f"Fold {fold + 1}" if self.args.vldtn_ratio_OR_k_fold > 1 else "Total"
            strTrain_times = f"{train_times} times" if train_times > 1 else f"{train_times} time"

            if self.args.lowest_accuracy != 0:
                should_continue = bool(self.max_accuracies[fold] >= self.args.lowest_accuracy)
            else:
                should_continue = True

            if should_continue:
                self.logger.info(f"{self.args.net_name} of {strFold}: trained {strTrain_times}, min_loss={min_loss:.6f}, "
                                 f"max_accuracy={max_accuracy * 100:.2f}% at epoch {self.best_epochs[fold]}.")
            else:
                self.logger.info(f"{self.args.net_name} of {strFold}: trained {strTrain_times}, min_loss={min_loss}, "
                                 f"max_accuracy({self.max_accuracies[fold] * 100:.2f}%) < lowest_accuracy({self.args.lowest_accuracy * 100:.2f}%), finish training.")
        except SystemError as ex:
            if str(ex) != "exit":
                print(ex)
            should_continue = False

        return should_continue

    def _one_fold_train(self, fold_index: int, train_indices: [int], vldtn_indices: [int]) -> bool:
        if self.args.vldtn_ratio_OR_k_fold > 1:
            self.logger.info(f"-----------------------------FOLD {fold_index + 1}-----------------------------")
        size = len(train_indices)
        if size <= 500:
            self.logger.info(f"Training labels ({size}):\n{self.params.all_labels[train_indices]}")
        else:
            self.logger.info(f"Training labels number: {size}")

        if self.args.check_validation:
            size = len(vldtn_indices)
            if size <= 500:
                self.logger.info(f"Validation indices ({size}):\n{vldtn_indices}")
                self.logger.info(f"Validation labels ({size}):\n{self.params.all_labels[vldtn_indices]}")
            else:
                self.logger.info(f"Total validation number: {size}")

        loss_history = []
        gen_loss_history = []
        train_acc_history = []
        vldtn_acc_history = []
        self.loss_histories[fold_index] = loss_history
        self.gen_loss_histories[fold_index] = gen_loss_history
        if self.args.check_training:
            self.train_acc_histories[fold_index] = train_acc_history
        if self.args.check_validation:
            self.val_acc_histories[fold_index] = vldtn_acc_history
        epoch_No = 0

        classifier, cls_optimizers, cls_schedulers = self.params.create_cls_factory.create_network()
        classifier.to(ag.Arguments.device)

        if isinstance(classifier, nw.NetworkBase):
            if self.args.vldtn_ratio_OR_k_fold > 1:
                classifier.load_parameters(fold_index + 1)
            else:
                classifier.load_parameters(self.params.test_time)

        if self.params.create_gen_factory is not None and self.args.cls_gen_train_ratio > 0:
            generator, gen_optimizers, gen_schedulers = self.params.create_gen_factory.create_network()
            generator.to(ag.Arguments.device)
        else:
            generator = None
            gen_optimizers = []
            gen_schedulers = []

        # Start training.
        while True:
            try:
                classifier.current_indices = train_indices
                if generator is not None:
                    generator.current_indices = train_indices

                cls_loss_value, gen_loss_value = self._train(classifier, cls_optimizers, generator, gen_optimizers, epoch_No)
                epoch_No += 1

                if cls_loss_value is None:
                    return False

                for cls_scheduler in cls_schedulers:
                    cls_scheduler.step()
                for gen_scheduler in gen_schedulers:
                    gen_scheduler.step()

                strFold = f"Fold {fold_index + 1}, " if self.args.vldtn_ratio_OR_k_fold > 1 else ""
                msg = f"{strFold}Epoch {epoch_No}: cls_loss={cls_loss_value:.6f}"
                if gen_loss_value is not None:
                    msg += f", gen_loss={gen_loss_value:.6f}"
                if self.args.verbose_logs:
                    self.logger.info(msg)
                else:
                    print(msg)

                if self.args.check_training:
                    train_accuracy, train_f1_scores = self.__validate(classifier, train_indices, fold_index)
                    indices = self.predicted_labels[fold_index] != self.ground_truth_labels[fold_index]
                    error_indices = train_indices[indices]
                    msg1 = "Training accuracy=%.2f%%%s, error count: %d" % (train_accuracy * 100, train_f1_scores, len(error_indices))
                else:
                    msg1 = ""
                    train_accuracy = None

                if self.args.check_validation:
                    vldtn_accuracy, val_f1_scores = self.__validate(classifier, vldtn_indices, fold_index)
                    msg2 = "Validation accuracy=%.2f%%%s, predicted_labels:\n%s" % (vldtn_accuracy * 100, val_f1_scores, self.predicted_labels[fold_index])
                else:
                    msg2 = ""
                    vldtn_accuracy = None

                if msg1 != "" and msg2 != "":
                    msg = "\n".join([msg1, msg2])
                else:
                    msg = msg1 + msg2
                if msg != "":
                    if self.args.verbose_logs:
                        self.logger.info(msg)
                    else:
                        print(msg)

                loss_history.append(cls_loss_value)
                if train_accuracy is not None:
                    train_acc_history.append(train_accuracy)
                if vldtn_accuracy is not None:
                    vldtn_acc_history.append(vldtn_accuracy)
                if self.params.create_gen_factory is not None and self.args.cls_gen_train_ratio > 0:
                    if gen_loss_value is not None:
                        gen_loss_history.append(gen_loss_value)
                    else:
                        if len(gen_loss_history) == 0:
                            gen_loss_history.append(0.0)
                        else:
                            gen_loss_history.append(gen_loss_history[-1])

                if train_accuracy is not None or vldtn_accuracy is not None:
                    if self.args.accuracy_standard == es.EAccuracy.Train:
                        assert train_accuracy is not None
                        accuracy = train_accuracy
                    elif self.args.accuracy_standard == es.EAccuracy.Validate:
                        assert vldtn_accuracy is not None
                        accuracy = vldtn_accuracy
                    elif self.args.accuracy_standard == es.EAccuracy.Double:
                        assert train_accuracy is not None and vldtn_accuracy is not None
                        accuracy = (train_accuracy + vldtn_accuracy) / 2
                    else:
                        raise NotImplementedError(f"accuracy_standard={self.args.accuracy_standard.name}")
                else:
                    accuracy = -1.0

                # I only record a state as the best state in an epoch when it satisfies a specified strategy.
                record_best = False
                if self.args.best_strategy == es.EBestStrategy.Loss:
                    if cls_loss_value < self.min_losses[fold_index]:
                        record_best = True
                    elif cls_loss_value == self.min_losses[fold_index] and accuracy - 1e-5 > self.max_accuracies[fold_index]:
                        record_best = True
                elif self.args.best_strategy == es.EBestStrategy.Accuracy:
                    if accuracy > self.max_accuracies[fold_index]:
                        record_best = True
                    elif accuracy == self.max_accuracies[fold_index] and cls_loss_value + 1e-6 < self.min_losses[fold_index]:
                        # To get a better ROC curve.
                        record_best = True
                else:
                    raise NotImplementedError(f"best_strategy={self.args.best_strategy.name}")

                if record_best:
                    self.min_losses[fold_index] = cls_loss_value
                    self.max_accuracies[fold_index] = accuracy
                    self.best_states[fold_index] = copy.deepcopy(classifier.state_dict())
                    self.best_epochs[fold_index] = epoch_No

                if self.args.stop_strategy == es.EStopStrategy.FixedEpochs:
                    if epoch_No == int(self.args.stop_value):
                        break
                elif self.args.stop_strategy == es.EStopStrategy.MinLoss:
                    if cls_loss_value <= self.args.stop_value:
                        break
                elif self.args.stop_strategy == es.EStopStrategy.EarlyStop:
                    early_stopping = self.args.stop_value
                    assert isinstance(early_stopping, es.EarlyStopping)
                    if not early_stopping.standard:
                        if cls_loss_value < early_stopping.best_value - early_stopping.min_delta:
                            early_stopping.best_value = cls_loss_value
                            early_stopping.counter = 0
                        else:
                            early_stopping.counter += 1
                            if early_stopping.counter > early_stopping.patience:
                                break
                    else:
                        if accuracy > early_stopping.best_value + early_stopping.min_delta:
                            early_stopping.best_value = accuracy
                            early_stopping.counter = 0
                        else:
                            early_stopping.counter += 1
                            if early_stopping.counter > early_stopping.patience:
                                break
                else:
                    raise NotImplementedError(f"stop_strategy={self.args.stop_strategy.name}")
            except torch.cuda.OutOfMemoryError as ex:
                self.logger.error(ex)
                sufficient_memory = False
                while not sufficient_memory:
                    user_input = SolverBase.__get_user_input_with_timeout(
                        "The minimal requirement of GPU is not yet satisfied. Press 'q' to finish the job (15s): ", 15)
                    if user_input and user_input.lower() == 'q':
                        raise SystemError("exit")
                    sufficient_memory = self.__check_available_memory()
                self.logger.info("The minimal requirement of GPU is satisfied, and continue training...")
        return True

    @abc.abstractmethod
    def _train(self, classifier: nw.NetworkBase, cls_optimizers: [too.Optimizer], generator, gen_optimizers: [too.Optimizer],
               epoch_No: int) -> (t.Optional[float], t.Optional[float]):
        raise NotImplementedError

    def __check_available_memory(self) -> bool:
        torch.cuda.empty_cache()  # Clear GPU cache.
        mem_info = torch.cuda.mem_get_info()
        available_memory = mem_info[0] / 1024 ** 3
        print(f"Currently available GPU: {available_memory:.2f} GB.")
        return available_memory >= self.args.min_required_memory

    @staticmethod
    def __get_user_input_with_timeout(prompt, timeout) -> t.Optional[str]:
        """
        Wait for user input. Return None if timeout.
        """
        print(prompt, flush=True)
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                return sys.stdin.readline().strip()
        except OSError as ex:
            print(ex)
            jm.JobManager.finish_job(1)
        return None

    def __get_best_fold(self) -> [int]:
        """
        :return: an ndarray
        """
        # self.min_losses = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3])
        # self.max_accuracies = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.3])
        if self.args.best_strategy == es.EBestStrategy.Loss:
            best_folds = np.where(self.min_losses == np.min(self.min_losses))[0]
        else:
            best_folds = np.where(self.max_accuracies == np.max(self.max_accuracies))[0]

        return best_folds

    @property
    def best_folds(self) -> [int]:
        if len(self.__best_folds) == 0:
            self.__best_folds = self.__get_best_fold()
        return self.__best_folds

    def __get_worst_fold(self) -> [int]:
        """
        :return: an ndarray
        """
        # self.min_losses = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3])
        # self.max_accuracies = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.3])
        if self.args.best_strategy == es.EBestStrategy.Loss:
            worst_folds = np.where(self.min_losses == np.max(self.min_losses))[0]
        else:
            worst_folds = np.where(self.max_accuracies == np.min(self.max_accuracies))[0]

        return worst_folds

    @property
    def worst_folds(self) -> [int]:
        if len(self.__worst_folds) == 0:
            self.__worst_folds = self.__get_worst_fold()
        return self.__worst_folds

    def test(self, fold_index: int, test_indices: [int], classifier: nn.Module = None) -> nn.Module:
        if classifier is None:
            classifier, _, _ = self.params.create_cls_factory.create_network()
            net_state = self.best_states[fold_index]
            classifier.load_state_dict(net_state)
            classifier.to(ag.Arguments.device)
        classifier.current_indices = test_indices

        ground_truth_labels, predicted_labels, predicted_scores = self._predict(classifier)
        self.ground_truth_labels[fold_index] = ground_truth_labels
        self.predicted_labels[fold_index] = predicted_labels
        self.predicted_scores[fold_index] = predicted_scores

        return classifier

    def save_all_parameters(self):
        for i in range(len(self.best_states)):
            best_state = self.best_states[i]
            classifier, _, _ = self.params.create_cls_factory.create_network()
            classifier.load_state_dict(best_state)

            if self.args.net_state_file is not None and self.args.net_state_file != "":
                net_state_file = self.args.net_state_file.replace(".pth", f"-{i+1:0>2}.pth")
                torch.save(best_state, net_state_file)

            if isinstance(classifier, nw.NetworkBase):
                classifier.save_parameters(i+1)

    @abc.abstractmethod
    def _predict(self, classifier: nw.NetworkBase) -> ([int], [int], [[float]]):
        raise NotImplementedError

    def __validate(self, classifier: nw.NetworkBase, vldtn_indices: [int], fold: int) -> (float, str):
        classifier.current_indices = vldtn_indices

        ground_truth_labels, predicted_labels, predicted_scores = self._predict(classifier)
        self.ground_truth_labels[fold] = ground_truth_labels
        self.predicted_labels[fold] = predicted_labels
        self.predicted_scores[fold] = predicted_scores

        accuracy = skm.accuracy_score(ground_truth_labels, predicted_labels)

        if self.args.display_F1_scores:
            f1_scores = []
            for cls in range(self.args.class_num):
                f1 = skm.f1_score(ground_truth_labels, predicted_labels, labels=[cls], average="macro", zero_division=0)
                f1_scores.append(f"{f1 * 100:.2f}%")
            str_f1_scores = ",".join(f1_scores)
            str_f1_scores = ", f1_scores=" + str_f1_scores
        else:
            str_f1_scores = ""

        return accuracy, str_f1_scores

    def evaluate(self, fold: int, test_indices: [int]) -> (float, float, float, float, float, float, float, float, int):
        if fold >= 0:
            ground_truth_labels = self.ground_truth_labels[fold]
            predicted_labels = self.predicted_labels[fold]
            predicted_scores = self.predicted_scores[fold]
        else:
            ground_truth_labels = SolverBase.__concat_arrays(self.ground_truth_labels)
            predicted_labels = SolverBase.__concat_arrays(self.predicted_labels)
            predicted_scores = SolverBase.__concat_arrays(self.predicted_scores)
        indices = ground_truth_labels != predicted_labels
        error_indices = test_indices[indices]
        error_count = len(error_indices)

        if fold >= 0:
            self.logger.info(f"ground_truth_labels({len(ground_truth_labels)}):\n{ground_truth_labels.tolist()}")
            self.logger.info(f"   predicted_labels({len(predicted_labels)}):\n{predicted_labels.tolist()}")
            printing_predicted_scores = [[np.round(v, 3) for v in row] for row in predicted_scores]
            self.logger.info(f"predicted_scores:\n{printing_predicted_scores}")
            if self.params.all_labels.shape[0] == len(self.params.all_samples):
                error_samples = [str(error_sample) for error_sample in self.params.all_samples[error_indices]]
                self.logger.info(f"error samples({len(error_indices)}):\n{error_samples}")
            else:
                self.logger.info(f"error count: {error_count}")
        else:
            self.logger.info(f"error count: {error_count}")

        accuracy = skm.accuracy_score(ground_truth_labels, predicted_labels) * 100
        balanced_accuracy = skm.balanced_accuracy_score(ground_truth_labels, predicted_labels) * 100
        classes = list(range(self.args.class_num))
        confusion = skm.confusion_matrix(ground_truth_labels, predicted_labels, labels=classes, normalize="true")

        if self.args.class_num == 2:
            """
            Cohen's Kappa: an statistical indicator to evaluate a classifier, whose range is between [-1, 1].
            When it is 1, it means the predicted results of the classifier are perfectly same with the factual situation without any bias.
            When it is 0, it means the coherence between the predicted results of the classifier and the factual situation is same as random guess.
            When it is negative, it means the coherence between the predicted results of the classifier and the factual situation is worse than random guess,
            and there may be systematic errors.
            """
            kappa_value = skm.cohen_kappa_score(ground_truth_labels, predicted_labels) * 100
            area_under_curve = skm.roc_auc_score(ground_truth_labels, predicted_scores[:, 1]) * 100
            tn, fp, fn, tp = confusion.ravel()
            # Specificity: the percent of correctly predicted cases in real negative cases (correctness among true negatives), namely, TN/(TN+FP).
            specificity = tn / (tn + fp) * 100
            if np.isnan(specificity):
                specificity = 0.0
            # Precision: the percent of correctly predicted cases in cases predicted as positive (correctness among all positives), namely, TP/(TP+FP).
            precision = tp / (tp + fp) * 100
            if np.isnan(precision):
                precision = 0.0
            # Recall or sensitivity: the percent of correctly predicted cases in real positive cases (correctness among true positives), namely, TP/(TP+FN).
            # The former is for machine learning, whereas the latter is for the medical field.
            recall = tp / (tp + fn) * 100
            if np.isnan(recall):
                recall = 0.0
            # F1 score: it indicates the capability of class balance, whose range is [0, 1]. And it's better when its value is greater.
            f1_score = 2 * precision * recall / (precision + recall)
            if np.isnan(f1_score):
                f1_score = 0.0
        else:
            # weights=None: all classes are equal; weights=linear: classes are sequential, with different severe extents.
            kappa_value = skm.cohen_kappa_score(ground_truth_labels, predicted_labels, weights="linear") * 100
            area_under_curve = SolverBase.__calculate_AUC(ground_truth_labels, predicted_scores, self.args.class_num) * 100
            specificity = SolverBase.__calculate_specificity(confusion, self.args.class_num) * 100
            # average=macro: simple arithmetic mean; average=weighted: weighted mean, consider class imbalance.
            precision = skm.precision_score(ground_truth_labels, predicted_labels, labels=classes, average="weighted") * 100
            recall = skm.recall_score(ground_truth_labels, predicted_labels, average="weighted") * 100
            f1_score = skm.f1_score(ground_truth_labels, predicted_labels, average="weighted") * 100

        confusion_matrix = []
        value_max_len = 0
        for i in range(confusion.shape[0]):
            confusion_row = []
            for j in range(confusion.shape[1]):
                confusion_value = f"{confusion[i][j] * 100:.2f}"
                if len(confusion_value) > value_max_len:
                    value_max_len = len(confusion_value)
                confusion_row.append(confusion_value)
            confusion_matrix.append(confusion_row)

        str_confusion = []
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                confusion_matrix[i][j] = confusion_matrix[i][j].rjust(value_max_len, ' ')

            str_confusion_row = ", ".join(confusion_matrix[i])
            if i == 0:
                str_confusion_row = "┌" + str_confusion_row + "┐"
            elif i == confusion.shape[0] - 1:
                str_confusion_row = "└" + str_confusion_row + "┘"
            else:
                str_confusion_row = "│" + str_confusion_row + "│"
            str_confusion.append(str_confusion_row)
        confusion_text = "\n".join(str_confusion)

        strReport = f"\naccuracy(%)={accuracy:.2f}, balanced_accuracy(%)={balanced_accuracy:.2f}" \
                    f"\nkappa_value(%)={kappa_value:.2f}, area_under_curve(%)={area_under_curve:.2f}," \
                    f"\nf1_score(%)={f1_score:.2f}, precision(%)={precision:.2f}," \
                    f"\nrecall(%)={recall:.2f}, specificity(%)={specificity:.2f}," \
                    f"\nconfusion_matrix(%)=\n{confusion_text}"
        self.logger.info(strReport)

        return accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count

    @staticmethod
    def __calculate_specificity(confusion: [[float]], class_num: int) -> float:
        """
        :param confusion: the column headers are ground truths, and the row headers are predictions.
        """
        specificities = []
        for i in range(class_num):
            TN = sum([confusion[j][k] for j in range(class_num) if j != i for k in range(class_num) if k != i])
            FP = sum([confusion[j][i] for j in range(class_num) if j != i])
            specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
            specificities.append(specificity)

        avg_specificity = np.mean(specificities)
        return float(avg_specificity)

    @staticmethod
    def __calculate_AUC(ground_truth_labels: [int], predicted_scores: [[float]], class_num: int) -> float:
        auc_scores = []

        for i in range(class_num):
            results = ground_truth_labels == i
            assert isinstance(results, np.ndarray), "Dismiss a warning."
            y_true_one_hot = results.astype(int)
            auc_score = skm.roc_auc_score(y_true_one_hot, predicted_scores[:, i])
            auc_scores.append(auc_score)

        area_under_curve = np.mean(auc_scores)
        return float(area_under_curve)

    def evaluate_k_folds(self) -> (float, float, float, float, float, float, float, float, int):
        assert self.args.vldtn_ratio_OR_k_fold > 1
        accuracies = []
        balanced_accuracies = []
        kappa_values = []
        are_under_curves = []
        f1_scores = []
        precisions = []
        recalls = []
        specificities = []
        total_error_count = 0

        k_folds = int(self.args.vldtn_ratio_OR_k_fold)
        self.logger.info(f"--------------------------------Evaluate {k_folds} Folds--------------------------------")

        for fold in range(k_folds):
            if fold in self.best_folds:
                best_or_worst = "best"
            elif fold in self.worst_folds:
                best_or_worst = "worst"
            else:
                best_or_worst = ""

            self.analyse_training_histories(fold, best_or_worst)
            self.plot_training_histories(fold, best_or_worst)

            if self.args.check_training:
                self.logger.info("Evaluate on a training set:")
                train_vldtn_indices = set(self.params.train_vldtn_indices.tolist())
                validation_indices = set(self.vldtn_label_indices[fold])
                training_indices = np.array(list(train_vldtn_indices - validation_indices))
                classifier = self.test(fold, training_indices)
                self.evaluate(fold, training_indices)
            else:
                classifier = None

            self.logger.info("Evaluate on a validation set:")
            self.test(fold, self.vldtn_label_indices[fold], classifier)
            accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count \
                = self.evaluate(fold, self.vldtn_label_indices[fold])

            accuracies.append(accuracy)
            balanced_accuracies.append(balanced_accuracy)
            kappa_values.append(kappa_value)
            are_under_curves.append(area_under_curve)
            f1_scores.append(f1_score)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            total_error_count += error_count

            if self.args.net_state_file is not None and self.args.net_state_file != "":
                best_state = self.best_states[fold]
                net_state_file = self.args.net_state_file.replace(".pth", f"-{fold + 1:0>2}.pth")
                torch.save(best_state, net_state_file)

            if isinstance(classifier, nw.NetworkBase):
                classifier.save_parameters(fold+1)

        avg_accuracy = np.mean(accuracies)
        avg_balanced_accuracy = np.mean(balanced_accuracies)
        avg_kappa_value = np.mean(kappa_values)
        avg_are_under_curve = np.mean(are_under_curves)
        avg_f1_score = np.mean(f1_scores)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_specificity = np.mean(specificities)

        std_accuracy = np.std(accuracies)
        std_balanced_accuracy = np.std(balanced_accuracies)
        std_kappa_value = np.std(kappa_values)
        std_are_under_curve = np.std(are_under_curves)
        std_f1_score = np.std(f1_scores)
        std_precision = np.std(precisions)
        std_recall = np.std(recalls)
        std_specificity = np.std(specificities)

        self.logger.info(f"--------------------------AVERAGE SCORES ({k_folds} folds)-------------------------")
        self.logger.info(f"error_count: {total_error_count}")
        self.logger.info(
            f"accuracy(%): {avg_accuracy:.2f}±{std_accuracy:.2f}, balanced_accuracy(%): {avg_balanced_accuracy:.2f}±{std_balanced_accuracy:.2f}")
        self.logger.info(
            f"kappa_value(%): {avg_kappa_value:.2f}±{std_kappa_value:.2f}, area_under_curve(%): {avg_are_under_curve:.2f}±{std_are_under_curve:.2f}")
        self.logger.info(
            f"f1_score(%): {avg_f1_score:.2f}±{std_f1_score:.2f}, precision(%): {avg_precision:.2f}±{std_precision:.2f}")
        self.logger.info(
            f"recall(%): {avg_recall:.2f}±{std_recall:.2f}, specificity(%): {avg_specificity:.2f}±{std_specificity:.2f}")
        self.logger.info(
            f"{total_error_count}\t{avg_accuracy:.2f}±{std_accuracy:.2f}\t{avg_balanced_accuracy:.2f}±{std_balanced_accuracy:.2f}\t{avg_kappa_value:.2f}±{std_kappa_value:.2f}\t{avg_are_under_curve:.2f}±{std_are_under_curve:.2f}\t{avg_f1_score:.2f}±{std_f1_score:.2f}\t{avg_precision:.2f}±{std_precision:.2f}\t{avg_recall:.2f}±{std_recall:.2f}\t{avg_specificity:.2f}±{std_specificity:.2f}")

        self.logger.info(f"---------------------------TOTAL SCORES ({k_folds} folds)--------------------------")
        strTrain_times1 = list(map(str, self.train_times))
        strTrain_times2 = "+".join(strTrain_times1)
        avg_train_times = np.mean(self.train_times)
        self.logger.info(f"Train times: ({strTrain_times2})/{k_folds}={avg_train_times}")

        lengths = [len(history) for history in self.loss_histories]
        strLengths = list(map(str, lengths))
        strItems = "+".join(strLengths)
        self.logger.info(f"Total epochs: {strItems}={sum(lengths)}.")
        self.evaluate(-1, SolverBase.__concat_arrays(self.vldtn_label_indices))

        return avg_accuracy, avg_balanced_accuracy, avg_kappa_value, avg_are_under_curve, avg_f1_score, avg_precision, avg_recall, avg_specificity, total_error_count

    def analyse_training_histories(self, fold_index: int, best_or_worst: str = ""):
        best_epoch = self.best_epochs[fold_index]
        if fold_index != -1:
            if best_or_worst != "":
                strBest_OR_worst = f"the {best_or_worst} fold, "
            else:
                strBest_OR_worst = ""
            report0 = f"Fold {fold_index + 1} ({strBest_OR_worst}best epoch on {best_epoch}/{len(self.loss_histories[fold_index])}):"
        else:
            report0 = f"Best epoch on {best_epoch}/{len(self.loss_histories[fold_index])}:"
        self.logger.info(report0)

        min_loss = np.min(self.loss_histories[fold_index])
        max_loss = np.max(self.loss_histories[fold_index])
        avg_loss = float(np.average(self.loss_histories[fold_index]))
        # sample standard deviation
        std_loss = float(np.std(self.loss_histories[fold_index], ddof=1))
        report1 = "Training min_loss=%.6f, max_loss=%.6f, avg_loss=%.6f, std_loss=%.6f" % (min_loss, max_loss, avg_loss, std_loss)
        self.logger.info(report1)

        avg_acc = 0
        std_acc = 0
        if len(self.train_acc_histories) > 0:
            min_acc = np.min(self.train_acc_histories[fold_index])
            max_acc = np.max(self.train_acc_histories[fold_index])
            avg_train_acc = np.average(self.train_acc_histories[fold_index])
            std_train_acc = float(np.std(self.train_acc_histories[fold_index], ddof=1))
            report2 = "Training min_acc=%.4f, max_acc=%.4f, avg_acc=%.4f, std_acc=%.4f" % (min_acc, max_acc, avg_train_acc, std_acc)
            self.logger.info(report2)
            avg_acc += avg_train_acc
            std_acc += std_train_acc

        if len(self.val_acc_histories) > 0:
            min_acc = np.min(self.val_acc_histories[fold_index])
            max_acc = np.max(self.val_acc_histories[fold_index])
            avg_val_acc = np.average(self.val_acc_histories[fold_index])
            std_val_acc = float(np.std(self.val_acc_histories[fold_index], ddof=1))
            report2 = "Validation min_acc=%.4f, max_acc=%.4f, avg_acc=%.4f, std_acc=%.4f" % (min_acc, max_acc, avg_val_acc, std_acc)
            self.logger.info(report2)
            avg_acc += avg_val_acc
            std_acc += std_val_acc

        if len(self.train_acc_histories) > 0 and len(self.val_acc_histories) > 0:
            avg_acc /= 2
            std_acc /= 2

        length = len(self.loss_histories[fold_index])
        if length < ag.Arguments.history_splitting_epoch:
            self.history_split_epochs[fold_index] = length + 1
        else:
            for i in range(length):
                loss_value = self.loss_histories[fold_index][i]
                if self.args.accuracy_standard == es.EAccuracy.Train:
                    accuracy = self.train_acc_histories[fold_index][i]
                elif self.args.accuracy_standard == es.EAccuracy.Validate:
                    accuracy = self.val_acc_histories[fold_index][i]
                else:
                    accuracy = (self.train_acc_histories[fold_index][i] + self.val_acc_histories[fold_index][i]) / 2

                if loss_value <= avg_loss and accuracy >= avg_acc:
                    history_split_epoch = i + 1 + 1
                    self.history_split_epochs[fold_index] = history_split_epoch
                    break

    def plot_training_histories(self, fold_index: int, best_OR_worst: str = ""):
        """
        :param fold_index: zero-based
        :param best_OR_worst: an empty string, `best`, `worst`
        """
        history_kinds = []
        if self.params.create_gen_factory is not None and self.args.cls_gen_train_ratio > 0:
            history_kinds.append("Generator")
        if len(self.train_acc_histories) > 0:
            history_kinds.append("Training")
        if len(self.val_acc_histories) > 0:
            history_kinds.append("Validation")

        plt.axis("off")
        fig = plt.figure()
        kind_size = len(history_kinds)
        heights = [10, 16, 24]
        height = heights[kind_size - 1]
        fig.set_figheight(height)

        loss_history = self.loss_histories[fold_index]
        history_split_epoch = self.history_split_epochs[fold_index]
        col_num = 2 if history_split_epoch < len(loss_history) + 1 else 1
        fig.set_figwidth(10 * col_num)
        gs = fig.add_gridspec(1 + kind_size, col_num)
        # hspace: vertical, wspace: horizontal
        gs.update(hspace=0.25, wspace=0.125)

        for c in range(col_num):
            total_length = len(loss_history)
            if c == 0:
                loss_history2 = loss_history[:history_split_epoch - 1]
            else:
                loss_history2 = loss_history[history_split_epoch - 1:]
            pic_loss = fig.add_subplot(gs[0, c])

            sequence_no = cu.get_sequence_no(fold_index + 1)
            if fold_index != -1:
                strSequence_No = f" (the {sequence_no} fold)"
            else:
                strSequence_No = ""

            strClassifier = "Classifier " if self.params.create_gen_factory is not None and self.args.cls_gen_train_ratio > 0 else ""
            pic_loss.set_title(f"{strClassifier}Training Loss{strSequence_No}", fontsize=14)
            from_epoch = 1 if c == 0 else history_split_epoch
            pic_loss.set_xlabel(f"from the {cu.get_sequence_no(from_epoch)} epoch, total {total_length} epochs", fontsize=14)
            pic_loss.grid()

            length = len(loss_history2)
            if length == 0:
                continue

            x_ticker_num = 25.0 if length <= 100 else 20.0
            y_ticker_num = 9.0
            unit_x = length / x_ticker_num
            if unit_x < 1.0:
                unit_x = 1
            else:
                unit_x = np.ceil(unit_x)
            min_y = min(loss_history2)
            max_y = max(loss_history2)
            unit_y = (max_y - min_y) / y_ticker_num
            if unit_x != 0:
                pic_loss.set_xticks(np.arange(0, length + 1, unit_x))
            if unit_y != 0:
                pic_loss.set_yticks(np.arange(min_y, max_y + unit_y / 10, unit_y))
            pic_loss.plot(loss_history2)
            pic_loss.tick_params(axis="x", rotation=45)

            if best_OR_worst != "":
                strBest_OR_worst = f" (the {best_OR_worst} fold)"
            else:
                strBest_OR_worst = ""

            for i, history_kind in enumerate(history_kinds):
                if history_kind == "Training":
                    history = self.train_acc_histories[fold_index]
                elif history_kind == "Validation":
                    history = self.val_acc_histories[fold_index]
                else:
                    history = self.gen_loss_histories[fold_index]

                if c == 0:
                    history2 = history[:history_split_epoch - 1]
                else:
                    history2 = history[history_split_epoch - 1:]
                total_length = len(history)

                pic = fig.add_subplot(gs[1 + i, c])
                if history_kind != "Generator":
                    pic.set_title(f"{history_kind} Accuracy{strBest_OR_worst}", fontsize=14)
                else:
                    pic.set_title(f"Generator Training Loss{strSequence_No}", fontsize=14)
                from_epoch = 1 if c == 0 else history_split_epoch
                pic.set_xlabel(f"from the {cu.get_sequence_no(from_epoch)} epoch, total {total_length} epochs", fontsize=14)
                pic.grid()

                part_length = len(history2)
                unit_x = part_length / x_ticker_num
                if unit_x < 1.0:
                    unit_x = 1
                else:
                    unit_x = np.ceil(unit_x)
                min_y = min(history2)
                max_y = max(history2)
                unit_y = (max_y - min_y) / y_ticker_num
                if unit_x != 0:
                    pic.set_xticks(np.arange(0, length + 1, unit_x))
                if unit_y != 0:
                    pic.set_yticks(np.arange(min_y, max_y + unit_y / 10, unit_y))
                pic.plot(history2)
                plt.xticks(rotation=45)

        left_margin = 0.045 if col_num == 2 else 0.07
        if kind_size == 1 and (self.params.create_gen_factory is None or self.args.cls_gen_train_ratio <= 0):
            plt.subplots_adjust(left_margin, bottom=0.075, right=0.98, top=0.960)
        else:
            plt.subplots_adjust(left_margin, bottom=0.045, right=0.98, top=0.975)

        fold_no = f"{fold_index + 1}-"
        strBest_OR_worst = f"-{best_OR_worst}" if best_OR_worst != "" else ""
        picture_path = os.path.join(self.args.log_dir_path, f"{fold_no}loss_AND_acc{strBest_OR_worst}.png")
        plt.savefig(picture_path)
        plt.close()

    def plot_results(self, back_title: str = ""):
        self.__plot_ROC(back_title)
        self.__plot_t_SNE(back_title)

    def __plot_ROC(self, back_title: str = ""):
        plt.figure(figsize=(10, 10))

        predicted_scores = SolverBase.__concat_arrays(self.predicted_scores)
        ground_truth_labels = SolverBase.__concat_arrays(self.ground_truth_labels)

        for i in range(self.args.class_num):
            y_true_one_hot = np.array(ground_truth_labels == i, dtype=np.int32)
            fpr, tpr, _ = skm.roc_curve(y_true_one_hot, predicted_scores[:, i])
            auc_value = skm.auc(fpr, tpr)

            if self.args.class_num == 2:
                # color = "blue" if i == 0 else "red"
                color = "#3b4cc0" if i == 0 else "#b40426"
            else:
                color = None
            plt.plot(fpr, tpr, color=color, alpha=0.7, lw=2, label="Class {} (AUC = {:.4f})".format(i, auc_value))

        plt.plot([0, 1], [0, 1], color="grey", alpha=0.7, lw=2, linestyle="--")

        plt.xlabel("Specificity", fontsize=14)
        plt.ylabel("Sensitivity", fontsize=14)
        plt.title(f"Receiver Operating Characteristic Curve of {self.args.classes} Classification", fontsize=14)
        plt.legend(loc="lower right")
        plt.subplots_adjust(left=0.09, bottom=0.07, right=0.96, top=0.94)

        if back_title != "":
            back_title = "-" + back_title
        picture_path = os.path.join(self.args.log_dir_path, f"ROC{back_title}.png")
        plt.savefig(picture_path)
        plt.close()

    def __plot_t_SNE(self, back_title: str = ""):
        predicted_scores = SolverBase.__concat_arrays(self.predicted_scores)
        predicted_labels = SolverBase.__concat_arrays(self.predicted_labels)

        N = predicted_scores.shape[0]
        perplexity = N // 2 if N <= 30 else 30
        n_iter = 500 if N <= 30 else 1000
        tsne = slm.TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=ag.Arguments.random_seed)
        h_tsne = tsne.fit_transform(predicted_scores)
        tx = cu.min_max_scale(h_tsne[:, 0], False)
        ty = cu.min_max_scale(h_tsne[:, 1], False)
        # tx = h_tsne[:, 0]
        # ty = h_tsne[:, 1]

        plt.figure(figsize=(10, 9.5))
        scatter = None
        if self.args.class_num == 2:
            cmap = plt.cm.get_cmap("bwr", self.args.class_num)
        elif self.args.class_num == 3:
            colors = plt.cm.viridis(np.linspace(0, 0.9, self.args.class_num))
            cmap = mc.ListedColormap(colors)
        else:
            cmap = plt.cm.get_cmap("tab10", self.args.class_num)

        for i in range(self.args.class_num):
            indices = [j for j, label in enumerate(predicted_labels) if label == i]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            scatter = plt.scatter(current_tx, current_ty, c=[i] * len(current_tx), cmap=cmap, vmin=0,
                                  vmax=self.args.class_num - 1, alpha=0.5)

        ticks = list(range(self.args.class_num))
        cbar = plt.colorbar(scatter, ticks=ticks, label="Class Label", fraction=0.1, pad=0.01)
        cbar.set_label("Class Labels", labelpad=0)

        plt.xlabel("Component X", fontsize=16)
        plt.ylabel("Component Y", fontsize=16)
        plt.title(f"t-SNE Visualization of {self.args.classes}-Classification", fontsize=18)
        plt.subplots_adjust(left=0.07, bottom=0.07, right=1, top=0.95)

        if back_title != "":
            back_title = "-" + back_title
        picture_path = os.path.join(self.args.log_dir_path, f"t-SNE{back_title}.png")
        plt.savefig(picture_path)
        plt.close()

    @staticmethod
    def __concat_arrays(arrays: list[np.ndarray]) -> np.ndarray:
        outputs = []
        for array in arrays:
            if isinstance(array, np.ndarray):
                outputs += array.tolist()
            else:
                raise ValueError(f"array is not an ndarray:\n{array}")
        outputs = np.array(outputs)
        return outputs
