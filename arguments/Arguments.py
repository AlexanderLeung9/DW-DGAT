import os
import subprocess
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import typing as t
import socket
import enums as es
import optimizers as om
import schedulers as sd


class Arguments(object):
    """
    Class fields will not be recorded automatically in logs.
    """
    # region directory paths
    # input
    document_dir: str = os.path.join(".", "documents")
    # output
    log_root_dir: str = None
    # endregion
    comments: str = """"""
    # Segment a plotting history into two parts if it's longer than or equal to this value.
    history_splitting_epoch: int = 51
    verbose_logs: bool = False
    # Useful only for BatchSolver. Take effect when it's greater than 0.
    print_every_iterations: int = 0
    # A subject has multiple samples. When enabled, SolverBase will split subjects instead of samples for training, validating, and testing.
    multi_samples: bool = True
    # Highlight these arguments in the argument list.
    highlighted_arguments: [str] = ["txt_indices", "single_graph_module", "adj_graph_type", "cls_gen_train_ratio"]
    random_seed: int = 231
    device: torch.device = None
    user_name: str = None
    host_name: str = None
    local_IP: str = None
    business: es.EBusiness = None
    classes: [int] = None

    def __init__(self):
        """
        Instance fields will be recorded automatically in logs.
        """
        self.business: t.Optional[es.EBusiness] = None
        self.classes: list[int] = None
        self.balance_dataset: bool = False
        self.dataset_dir: str = ""
        self.preprocessed_data_dir: str = ""
        # Validate the training set.
        self.check_training: bool = False
        # Validate the non-training set then print while training.
        self.check_validation: bool = True
        self.accuracy_standard: es.EAccuracy = es.EAccuracy.Validate
        # A strategy to choose the best state.
        self.best_strategy: es.EBestStrategy = es.EBestStrategy.Accuracy
        # A strategy to stop training.
        self.stop_strategy: es.EStopStrategy = es.EStopStrategy.FixedEpochs
        # It can be a float, an integer, or an EarlyStopping object.
        self.stop_value = 0
        # region set in the set_validation_mode().
        self.test_times: int = 0
        # The percent of test samples among all samples. None: no test at last.
        self.test_percent: t.Optional[float] = None
        """
        It's recommended to set the value in set_validation_mode().
        When it's greater than 1, it means to use cross-validation.
        When it's less than 1, its value denotes the percent (0~1) of validation samples among training samples.
        When it's equal to 1, it indicates to predict without training.
        """
        self.vldtn_ratio_OR_k_fold: float = 0
        # For logging.
        self.validation_mode: str = ""
        # endregion
        # Take effect when `vldtn_ratio_OR_k_fold` > 1. All data includes the three sets.
        self.re_test_all_data: bool = False
        # If the validation accuracy while training is lower than it, the training stops.
        self.lowest_accuracy: float = 0
        self.NaN_error_continue: bool = False
        # For indicating which network to be used and, saving a log and figures.
        self.log_dir_path: str = ""
        self._net_name: str = ""
        # The file to save or load network state.
        # Empty string: do not save or load; None: locates at the `preprocessed_data_dir` with the `net_name` being the file title.
        self.net_state_file: t.Optional[str] = None
        # False: load original data then preprocess; True: preprocess data then save; None: preprocess data then save if data does not exist, else load it.
        self.load_OR_save_data: t.Optional[bool] = False
        # Preprocessed data
        self.data_file_name: t.Optional[str] = None
        self.optimizer: t.Optional[om.OptimizerParams] = None
        self.scheduler: t.Optional[sd.SchedulerParams] = None
        # For inductive training networks.
        self.batch_size: int = 0
        # Display F1 scores in the validation of training phase.
        self.display_F1_scores: bool = False
        # For GAN-like networks. Take effect when it's greater than 0.
        self.cls_gen_train_ratio: int = 0
        # Unit: GB.
        self.min_required_memory: float = 4
        # False: inductive; True: transductive.
        self.learning_mode: bool = False
        self.max_train_times: int = 5
        # Please prepend a space before your comment if any.
        self.log_dir_comment: str = ""

    @staticmethod
    def initialize_globally(GPU_No: int):
        if Arguments.device is None:
            if torch.cuda.is_available() and GPU_No >= 0:
                Arguments.device = torch.device(f"cuda:{GPU_No}")
            else:
                Arguments.device = torch.device("cpu")
        cudnn.enabled = True

        if Arguments.log_root_dir is None:
            # Linux/macOS
            if os.name == "posix":
                Arguments.log_root_dir = os.path.join("..", "logs")
            # Windows
            else:
                Arguments.log_root_dir = os.path.join(".", "logs")

        if not os.path.isdir(Arguments.log_root_dir):
            os.makedirs(Arguments.log_root_dir)

        if Arguments.user_name is None:
            if os.name == "posix":
                try:
                    Arguments.user_name = os.environ["USERNAME"]
                except KeyError:
                    Arguments.user_name = subprocess.getoutput("whoami")
            else:
                Arguments.user_name = os.getlogin()
        if Arguments.host_name is None:
            Arguments.host_name = socket.gethostname()
        if Arguments.local_IP is None:
            Arguments.local_IP = socket.gethostbyname(Arguments.host_name)

    def initialize(self):
        if self.business is None:
            self.business = Arguments.business

        if self.classes is None:
            self.classes = Arguments.classes

    def _check_arguments(self):
        assert self.business is not None
        assert self.classes is not None
        assert len(self.classes) >= 2

        if self.test_percent is not None:
            assert 0 < self.test_percent < 1
        assert 0 < self.vldtn_ratio_OR_k_fold
        assert 0 <= self.print_every_iterations
        assert 1 <= self.test_times
        if self.vldtn_ratio_OR_k_fold > 1:
            assert self.test_times == 1
        assert 0 <= self.cls_gen_train_ratio
        assert self.check_training or self.check_validation
        assert 0 <= self.lowest_accuracy
        assert self.batch_size != 1

        if self.load_OR_save_data is None or self.load_OR_save_data:
            assert self.data_file_name is not None

    def set_validation_mode(self, vldtn_ratio_OR_k_fold: float, test_times: t.Optional[int] = None, test_percent: t.Optional[float] = None):
        if vldtn_ratio_OR_k_fold > 1:
            self.vldtn_ratio_OR_k_fold = int(vldtn_ratio_OR_k_fold)
            self.test_times = 1
            validation_mode = f"{self.vldtn_ratio_OR_k_fold}-fold cross-validation"
        elif vldtn_ratio_OR_k_fold < 1:
            self.vldtn_ratio_OR_k_fold = float(vldtn_ratio_OR_k_fold)
            assert test_times is not None
            self.test_times = test_times
            self.re_test_all_data = False
            validation_mode = f"{self.test_times}-time random validation"
        else:
            self.vldtn_ratio_OR_k_fold = int(vldtn_ratio_OR_k_fold)
            self.test_times = 1
            self.re_test_all_data = False
            validation_mode = f"Solely testing"
        self.test_percent = test_percent

        if test_percent is not None:
            validation_mode += f" with {test_percent*100}% test percent"

        self.validation_mode = validation_mode

    @property
    def has_initialized(self) -> bool:
        test = self.log_dir_path != ""
        return test

    @property
    def class_num(self) -> int:
        return len(self.classes)

    @property
    def net_name(self) -> str:
        if self._net_name == "":
            index = self.__class__.__name__.index("Args")
            self._net_name = self.__class__.__name__[:index]
        return self._net_name
