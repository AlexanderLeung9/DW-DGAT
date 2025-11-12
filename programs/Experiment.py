import os
import copy
import time
import platform
import typing as t
import numpy as np
import pathlib as pl
import logging as l
import torch
import importlib
import arguments as ag
import utils as u
import solvers as sol
import datasets as ds
import networks as nw


class Experiment(object):
    __framework_version: str = "3.3.4"

    def __init__(self, args: ag.Arguments):
        self.args: ag.Arguments = args
        self.logger: t.Optional[l.Logger] = None
        self.sample_set: t.Optional[ds.SampleSet] = None

        self.final_accuracy: float = 0
        self.final_loss: float = 0

    def initialize(self):
        u.cu.set_random(ag.Arguments.random_seed)
        if not self.args.has_initialized:
            self.args.initialize()
        if self.args.stop_value != 0:
            self.logger = self.__create_logger(l.INFO)
        self.__log_args()

    def run(self):
        assert self.sample_set is not None, "BDExperiment.initialize() has not been called!"

        accuracy: float = 0.0
        total_error_count: int = 0
        accuracies: [float] = []
        balanced_accuracies: [float] = []
        kappa_values: [float] = []
        are_under_curves: [float] = []
        f1_scores: [float] = []
        precisions: [float] = []
        recalls: [float] = []
        specificities: [float] = []
        min_losses: [float] = []
        start_time = time.time()

        for test_index in range(self.args.test_times):
            test_time = test_index + 1
            if self.args.test_times > 1:
                self.logger.info(f"-------------------------------ROUND {test_time}-------------------------------")

            solver = self.__create_solver(test_time)
            if solver is None:
                return

            if self.args.vldtn_ratio_OR_k_fold != 1:
                result = solver.train()
                if not result:
                    self.__rename_log_dir_title(" error")
                    return

                if self.args.check_validation or self.args.test_percent is not None:
                    accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count = self.__test_and_evaluate_after_training(solver)

                    total_error_count += error_count
                    accuracies.append(accuracy)
                    balanced_accuracies.append(balanced_accuracy)
                    kappa_values.append(kappa_value)
                    are_under_curves.append(area_under_curve)
                    f1_scores.append(f1_score)
                    recalls.append(recall)
                    precisions.append(precision)
                    specificities.append(specificity)
                else:
                    # self-supervision
                    solver.save_all_parameters()

                    mean_loss = np.mean(solver.min_losses)
                    min_losses.append(mean_loss)
            else:
                accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count = self.__test_and_evaluate(solver)

        if self.args.test_times > 1:
            if len(accuracies) > 0:
                self.final_accuracy = self.__report_total_test_results(total_error_count, accuracies, balanced_accuracies, kappa_values, are_under_curves, f1_scores, recalls, precisions, specificities)
            else:
                self.final_loss = np.mean(np.array(min_losses))
        else:
            self.final_accuracy = accuracy

        self.__report_used_time(start_time)
        if self.final_accuracy > 0:
            str_final_value = f" {self.final_accuracy:.2f}%"
        elif self.final_loss > 0:
            str_final_value = f" {self.final_loss}"
        else:
            raise NotImplementedError
        self.__rename_log_dir_title(str_final_value)

    def __test_and_evaluate(self, solver: sol.SolverBase) -> (float, float, float, float, float, float, float, float, int):
        solver.best_states[-1] = torch.load(self.args.net_state_file, map_location=ag.Arguments.device)
        solver.test(-1, solver.params.test_indices)

        accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count = solver.evaluate(-1, solver.params.test_indices)

        file_title, extension = os.path.splitext(os.path.basename(self.args.net_state_file))
        solver.plot_results(file_title)

        return accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count

    def __test_and_evaluate_after_training(self, solver: sol.SolverBase) -> (float, float, float, float, float, float, float, float, int):
        if self.args.vldtn_ratio_OR_k_fold > 1:
            accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count = solver.evaluate_k_folds()

            k_fold = int(self.args.vldtn_ratio_OR_k_fold)
            title = f"{k_fold}_folds"
            solver.plot_results(title)

            if self.args.re_test_all_data:
                self.logger.info("Re-test on all data...")
                if solver.args.re_test_all_data:
                    solver.args.batch_size = len(solver.params.train_vldtn_indices)

                for best_fold in solver.best_folds:
                    title = f"{u.cu.get_sequence_no(best_fold + 1)}_fold"
                    self.logger.info(f"---------------------The {title.replace('_', ' ')} evaluation on all data---------------------")

                    solver.clear_test_results()
                    solver.test(best_fold, solver.params.train_vldtn_indices)
                    solver.evaluate(best_fold, solver.params.train_vldtn_indices)
                    solver.plot_results(title)

            if solver.params.test_indices is not None:
                self.logger.info("Validate the test set...")
                for best_fold in solver.best_folds:
                    title = f"{u.cu.get_sequence_no(best_fold + 1)}_fold"
                    self.logger.info(f"---------------------The {title.replace('_', ' ')} evaluation on a test set---------------------")

                    solver.clear_test_results()
                    solver.test(best_fold, solver.params.test_indices)
                    solver.evaluate(best_fold, solver.params.test_indices)
                    solver.plot_results(title)

        elif self.args.vldtn_ratio_OR_k_fold < 1:
            solver.analyse_training_histories(-1)
            solver.plot_training_histories(-1)

            title = f"{u.cu.get_sequence_no(solver.params.test_time)}_time"
            test_indices = np.concatenate(solver.vldtn_label_indices if solver.params.test_indices is None else solver.vldtn_label_indices)

            if self.args.check_training:
                self.logger.info(f"--------------------The {title.replace('_', ' ')} evaluation on a training set--------------------")
                train_vldtn_indices = set(solver.params.train_vldtn_indices)
                validation_indices = set(test_indices)
                train_indices = np.array(list(train_vldtn_indices - validation_indices))
                classifier = solver.test(-1, train_indices)
                solver.evaluate(-1, train_indices)
            else:
                classifier = None

            self.logger.info(f"--------------------The {title.replace('_', ' ')} evaluation on a test set--------------------")
            solver.test(-1, test_indices, classifier)
            accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count = solver.evaluate(-1, test_indices)

            solver.plot_results(title)

            if self.args.net_state_file is not None and self.args.net_state_file != "":
                if self.args.test_times > 1:
                    net_state_file = self.args.net_state_file.replace(".pth", f"-{solver.params.test_time:0>2}.pth")
                else:
                    net_state_file = self.args.net_state_file
                torch.save(solver.best_states[-1], net_state_file)

            if isinstance(classifier, nw.NetworkBase):
                classifier.save_parameters(solver.params.test_time)
        else:
            raise ValueError("Dismiss a warning.")

        return accuracy, balanced_accuracy, kappa_value, area_under_curve, f1_score, precision, recall, specificity, error_count

    def __report_total_test_results(
            self, total_error_count: int, accuracies: [float], balanced_accuracies: [float], kappa_values: [float], are_under_curves: [float],
            f1_scores: [float], precisions: [float], recalls: [float], specificities: [float]) -> float:
        num = len(accuracies)
        avg_error_count = total_error_count / num
        avg_accuracy = float(np.mean(accuracies))
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

        self.logger.info(f"-----------------------------------AVERAGE SCORES ({self.args.test_times} times)-----------------------------------")
        self.logger.info(f"error_count: {avg_error_count:.2f}")
        self.logger.info(f"accuracy(%): {avg_accuracy:.2f}±{std_accuracy:.2f}, balanced_accuracy(%): {avg_balanced_accuracy:.2f}±{std_balanced_accuracy:.2f}")
        self.logger.info(f"kappa_value(%): {avg_kappa_value:.2f}±{std_kappa_value:.2f}, area_under_curve(%): {avg_are_under_curve:.2f}±{std_are_under_curve:.2f}")
        self.logger.info(f"f1_score(%): {avg_f1_score:.2f}±{std_f1_score:.2f}, precision(%): {avg_precision:.2f}±{std_precision:.2f}")
        self.logger.info(f"recall(%): {avg_recall:.2f}±{std_recall:.2f}, specificity(%): {avg_specificity:.2f}±{std_specificity:.2f}")
        self.logger.info(f"{avg_error_count:.2f}\t{avg_accuracy:.2f}±{std_accuracy:.2f}\t{avg_balanced_accuracy:.2f}±{std_balanced_accuracy:.2f}\t{avg_kappa_value:.2f}±{std_kappa_value:.2f}\t{avg_are_under_curve:.2f}±{std_are_under_curve:.2f}\t{avg_f1_score:.2f}±{std_f1_score:.2f}\t{avg_precision:.2f}±{std_precision:.2f}\t{avg_recall:.2f}±{std_recall:.2f}\t{avg_specificity:.2f}±{std_specificity:.2f}")

        return avg_accuracy

    def __report_used_time(self, start_time: float):
        end_time = time.time()
        total_time = int(np.round(end_time - start_time, 0))
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        strUsed_time = f"Total used time of {self.args.net_name}: {hours}:{minutes}:{seconds}."
        self.logger.info(strUsed_time)

    def __rename_log_dir_title(self, final_value: str):
        file_handler = self.logger.handlers[1]
        file_handler.close()
        new_dir_path = self.args.log_dir_path + f"{final_value}{self.args.log_dir_comment}"

        system = platform.system()
        if system == "Linux":
            try:
                os.rename(self.args.log_dir_path, new_dir_path)
            except PermissionError as e:
                print(e)
        elif system == "Windows":
            WAIT_SECONDS = 3
            print(f"Wait for {WAIT_SECONDS} seconds to rename the folder...")
            vbs_file_path = r".\utils\MoveFolder.vbs"
            arguments = f"{WAIT_SECONDS} \"{self.args.log_dir_path}\" \"{new_dir_path}\""
            os.system(f"start WScript {vbs_file_path} {arguments}")
        else:
            raise NotImplementedError(f"system={system}")

    def __create_solver(self, test_time: int) -> t.Optional[sol.SolverBase]:
        index = self.args.__class__.__name__.index("Args")
        name_part1 = self.args.__class__.__name__[:index]
        solver_param_name = name_part1 + "SolParams"
        module = importlib.import_module("solvers")
        parm_constructor = getattr(module, solver_param_name)
        solver_params = parm_constructor(self.sample_set.all_samples, self.logger, self.args)

        assert isinstance(solver_params, sol.SolverBaseParams)
        solver_params.test_time = test_time
        solver_params.initialize()
        if self.args.stop_value == 0:
            print("Preprocessing completed!")
            return None

        sol_constructor = getattr(module, solver_params.solver_name)
        solver = sol_constructor(solver_params)
        return solver

    def __create_logger(self, log_level: int) -> l.Logger:
        log_file_path = os.path.join(self.args.log_dir_path, f"{self.args.net_name}.log")
        path = pl.Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = path.name.split(".")[0]
        logger = l.Logger(timestamp, log_level)
        format_string = "%(asctime)s [%(levelname)s] %(message)s"
        l.basicConfig(level=l.INFO, format=format_string)
        formatter = l.Formatter(format_string)

        file_handler = l.FileHandler(log_file_path, encoding="UTF-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = l.StreamHandler()
        # stream_handler.setLevel(l.WARN)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.info(f"Hundred-net Framework v{Experiment.__framework_version} © 2024-2025 Chengjia Liang. Licensed under the Apache License 2.0.")
        return logger

    def __log_args(self):
        tArgs = copy.deepcopy(self.args)
        del tArgs.log_dir_path
        del tArgs.log_dir_comment
        del tArgs.test_times
        del tArgs.vldtn_ratio_OR_k_fold

        deleted_keys = []
        for key in tArgs.__dict__.keys():
            if getattr(tArgs, key) is None and key not in ag.Arguments.highlighted_arguments:
                if key != "load_OR_save_data":
                    deleted_keys.append(key)
        for key in deleted_keys:
            delattr(tArgs, key)

        ablation_pairs = []
        for key in ag.Arguments.highlighted_arguments:
            if key in tArgs.__dict__.keys():
                ablation_pairs.append((key + " *", tArgs.__dict__[key]))

        current_dir = os.getcwd()
        project_name = os.path.basename(current_dir)
        pairs = [("host_name", ag.Arguments.host_name), ("user_name", ag.Arguments.user_name), ("local_IP", ag.Arguments.local_IP),
                 ("project_name", project_name), ("device", ag.Arguments.device), ("net_name", tArgs.net_name)]
        pairs2 = [pair for pair in sorted(tArgs.__dict__.items(), key=lambda pair: pair[0])
                  if pair[0] not in ag.Arguments.highlighted_arguments and pair[0][0] != '_']
        pairs.extend(pairs2)
        pairs.extend(ablation_pairs)

        key_max_len = 0
        val_max_len = 0
        item_size = 0
        for key, value in pairs:
            if len(key) > key_max_len:
                key_max_len = len(key)
            if len(str(value)) > val_max_len:
                val_max_len = len(str(value))
            item_size += 1

        no_len = 1 if item_size < 10 else 2
        table = ["┌" + "─" * (no_len + 2) + "┬" + "─" * (key_max_len + 2) + "┬" + "─" * (val_max_len + 2) + "┐", f"│ {'No'.center(no_len)} │ {'Key'.center(key_max_len)} │ {'Value'.center(val_max_len)} │", "├" + "─" * (no_len + 2) + "┼" + "─" * (key_max_len + 2) + "┼" + "─" * (val_max_len + 2) + "┤", ]
        for i, (key, value) in enumerate(pairs):
            table.append(f"│ {str(int(i) + 1).rjust(no_len)} │ {key.ljust(key_max_len)} │ {str(value).ljust(val_max_len)} │")
        table.append("└" + "─" * (no_len + 2) + "┴" + "─" * (key_max_len + 2) + "┴" + "─" * (val_max_len + 2) + "┘")

        table_text = "\n".join(table)
        if self.logger is None:
            print(f"{tArgs.__class__.__name__}:\n{table_text}")
        else:
            self.logger.info(f"{tArgs.__class__.__name__}:\n{table_text}")

        if ag.Arguments.comments is not None and ag.Arguments.comments != "":
            if self.logger is None:
                print(f"Comments: {ag.Arguments.comments}")
            else:
                self.logger.info(f"Comments: {ag.Arguments.comments}")
        if self.args.log_dir_comment != "":
            if self.logger is None:
                print(f"log_dir_comment:{self.args.log_dir_comment}")
            else:
                self.logger.info(f"log_dir_comment:{self.args.log_dir_comment}")

    @property
    def auto_job_name(self) -> str:
        assert self.args.has_initialized, "You have not yet initialized the instance."
        strClasses = "-".join(list(map(str, self.args.classes)))
        job_name = f"{strClasses} {self.args.net_name}"
        return job_name
