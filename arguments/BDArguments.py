import os
import time
import typing as t
import numpy as np
import arguments as ag
import enums as es


class BDArguments(ag.Arguments):
    """
    For brain disorder diagnosis.
    """
    # only months
    EVENTS: list[str] = ["00m", "12m", "24m", "48m"]
    # add part path if any
    event_parts: list[str] = None
    ROI_NUM: int = 90
    NII_FILE_NAMES: list[str] = [
        # region 3D images
        r"T1/co_{0}_swap_bet_crop_resample_2MNI152.nii.gz",  # 0
        r"standard_space/{0}_06LDHs_4normalize_to_target_2mm.nii.gz",  # 1
        r"standard_space/{0}_07LDHk_4normalize_to_target_2mm.nii.gz",  # 2
        r"standard_space/{0}_FA_4normalize_to_target_2mm.nii.gz",  # 3
        r"standard_space/{0}_L1_4normalize_to_target_2mm.nii.gz",  # 4
        r"standard_space/{0}_L23m_4normalize_to_target_2mm.nii.gz",  # 5
        r"standard_space/{0}_MD_4normalize_to_target_2mm.nii.gz",  # 6
        # endregion
    ]
    TXT_FILE_NAMES: list[str] = [
        # region 2D networks
        # 3; (90, 90)
        "Network/Deterministic/{0}_dti_FACT_45_02_1_0_Matrix_FA_AAL_Contract_90_2MM_90.txt",  # 0
        "Network/Deterministic/{0}_dti_FACT_45_02_1_0_Matrix_FN_AAL_Contract_90_2MM_90.txt",  # 1
        "Network/Deterministic/{0}_dti_FACT_45_02_1_0_Matrix_Length_AAL_Contract_90_2MM_90.txt",  # 2
        # endregion
        # region 1D vectors
        # 2 / 2 = 1; (90, 1)
        # quantity of voxels in which the fibers terminate in each ROI.
        "Network/Deterministic/{0}_dti_FACT_45_02_1_0_ROISurfaceSize_AAL_Contract_90_2MM_90.txt",  # 3
        # quantity of voxels in each ROI.
        "Network/Deterministic/{0}_dti_FACT_45_02_1_0_ROIVoxelSize_AAL_Contract_90_2MM_90.txt",  # 4
        # endregion
        # region ROI weights
        # 3 * 7 = 21; (90, 1)
        # T1 MRI
        "ROI/{0}_T1-AvgValue_AAL-90-2mm.txt",  # 5
        "ROI/{0}_T1-MaxValue_AAL-90-2mm.txt",  # 6
        "ROI/{0}_T1-CentroidWeight_AAL-90-2mm.txt",  # 7
        # Spearman Local Diffusion Homogeneity
        "ROI/{0}_06LDHs-AvgValue_AAL-90-2mm.txt",  # 8
        "ROI/{0}_06LDHs-MaxValue_AAL-90-2mm.txt",  # 9
        "ROI/{0}_06LDHs-CentroidWeight_AAL-90-2mm.txt",  # 10
        # Kendall Local Diffusion Homogeneity
        "ROI/{0}_07LDHk-AvgValue_AAL-90-2mm.txt",  # 11
        "ROI/{0}_07LDHk-MaxValue_AAL-90-2mm.txt",  # 12
        "ROI/{0}_07LDHk-CentroidWeight_AAL-90-2mm.txt",  # 13
        # axial diffusivity
        "ROI/{0}_L1-AvgValue_AAL-90-2mm.txt",  # 14
        "ROI/{0}_L1-MaxValue_AAL-90-2mm.txt",  # 15
        "ROI/{0}_L1-CentroidWeight_AAL-90-2mm.txt",  # 16
        # radial diffusivity
        "ROI/{0}_L23m-AvgValue_AAL-90-2mm.txt",  # 17
        "ROI/{0}_L23m-MaxValue_AAL-90-2mm.txt",  # 18
        "ROI/{0}_L23m-CentroidWeight_AAL-90-2mm.txt",  # 19
        # mean diffusivity
        "ROI/{0}_MD-AvgValue_AAL-90-2mm.txt",  # 20
        "ROI/{0}_MD-MaxValue_AAL-90-2mm.txt",  # 21
        "ROI/{0}_MD-CentroidWeight_AAL-90-2mm.txt",  # 22
        # fractional anisotropy
        "ROI/{0}_FA-AvgValue_AAL-90-2mm.txt",  # 23
        "ROI/{0}_FA-MaxValue_AAL-90-2mm.txt",  # 24
        "ROI/{0}_FA-CentroidWeight_AAL-90-2mm.txt",  # 25
        # endregion
        # region centroids
        # 1 * 7 = 7; (90, 3)
        "ROI/{0}_T1-CentroidPosition_AAL-90-2mm.txt",  # 26
        "ROI/{0}_06LDHs-CentroidPosition_AAL-90-2mm.txt",  # 27
        "ROI/{0}_07LDHk-CentroidPosition_AAL-90-2mm.txt",  # 28
        "ROI/{0}_FA-CentroidPosition_AAL-90-2mm.txt",  # 29
        "ROI/{0}_L1-CentroidPosition_AAL-90-2mm.txt",  # 30
        "ROI/{0}_L23m-CentroidPosition_AAL-90-2mm.txt",  # 31
        "ROI/{0}_MD-CentroidPosition_AAL-90-2mm.txt",  # 32
        # 1 * 7 = 7; (90, 1)
        "ROI/{0}_T1-CentroidLength_AAL-90-2mm.txt",  # 33
        "ROI/{0}_06LDHs-CentroidLength_AAL-90-2mm.txt",  # 34
        "ROI/{0}_07LDHk-CentroidLength_AAL-90-2mm.txt",  # 35
        "ROI/{0}_FA-CentroidLength_AAL-90-2mm.txt",  # 36
        "ROI/{0}_L1-CentroidLength_AAL-90-2mm.txt",  # 37
        "ROI/{0}_L23m-CentroidLength_AAL-90-2mm.txt",  # 38
        "ROI/{0}_MD-CentroidLength_AAL-90-2mm.txt",  # 39
        # 1 * 7 = 7; (90, 3)
        "ROI/{0}_T1-CentroidCosAgl_AAL-90-2mm.txt",  # 40
        "ROI/{0}_06LDHs-CentroidCosAgl_AAL-90-2mm.txt",  # 41
        "ROI/{0}_07LDHk-CentroidCosAgl_AAL-90-2mm.txt",  # 42
        "ROI/{0}_FA-CentroidCosAgl_AAL-90-2mm.txt",  # 43
        "ROI/{0}_L1-CentroidCosAgl_AAL-90-2mm.txt",  # 44
        "ROI/{0}_L23m-CentroidCosAgl_AAL-90-2mm.txt",  # 45
        "ROI/{0}_MD-CentroidCosAgl_AAL-90-2mm.txt",  # 46
        # endregion
    ]

    def __init__(self):
        super().__init__()
        self.use_AD_bad2_data: bool = True
        self.nii_indices: list[int] = None
        self.txt_indices: list[int] = None
        self.networks_merge_mode: t.Optional[es.ENetworksMergeMode] = None

    def initialize(self):
        assert not self.has_initialized, "It is not supposed to reenter this method."
        super().initialize()

        if self.classes is None:
            if self.business == es.EBusiness.PD:
                self.classes = [0, 1, 2]
            elif self.business == es.EBusiness.AD:
                self.classes = [0, 2, 4]
            else:
                raise NotImplementedError(f"business={self.business}")

        if self.lowest_accuracy == 0:
            self.lowest_accuracy = np.round(1 / len(self.classes), 6)

        if self.lowest_accuracy == 0:
            self.lowest_accuracy = np.round(1 / len(self.classes), 6)

        strClasses = "-".join(list(map(str, self.classes)))
        if self.log_dir_path == "":
            if self.business == es.EBusiness.PD:
                if self.classes == [0, 1]:
                    tClasses = "(1) " + strClasses
                elif self.classes == [1, 2]:
                    tClasses = "(2) " + strClasses
                elif self.classes == [0, 2]:
                    tClasses = "(3) " + strClasses
                else:
                    tClasses = strClasses
            elif self.business == es.EBusiness.AD:
                if self.classes == [0, 2]:
                    tClasses = "(1) " + strClasses
                elif self.classes == [2, 4]:
                    tClasses = "(2) " + strClasses
                elif self.classes == [0, 4]:
                    tClasses = "(3) " + strClasses
                else:
                    tClasses = strClasses
            else:
                raise NotImplementedError(f"business={self.business}")

            caption = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.log_dir_path = os.path.join(ag.Arguments.log_root_dir, tClasses, self.net_name, caption)

        self.initialize_dataset()

        if self.preprocessed_data_dir == "":
            self.preprocessed_data_dir = os.path.join(self.dataset_dir, "preprocessed_data")
        if not os.path.isdir(self.preprocessed_data_dir):
            os.makedirs(self.preprocessed_data_dir)

        if self.net_state_file is None:
            net_state_dir = os.path.join(self.preprocessed_data_dir, self.net_name)
            if not os.path.isdir(net_state_dir):
                os.makedirs(net_state_dir)

            self.net_state_file = os.path.join(net_state_dir, f"{self.net_name}-{self.business.name}-{strClasses.replace('-', ',')}.pth")

        if self.load_OR_save_data is None or self.load_OR_save_data:
            self.data_file_name = f"{self.business.name}-{strClasses.replace('-', ',')}-{self.net_name}.dat"

        self._check_arguments()

    def initialize_dataset(self):
        if os.name == "posix":
            if self.dataset_dir == "":
                home_dir = os.path.expanduser("~")
                if home_dir == "/root":
                    dataset_root = "/hy-tmp"
                else:
                    index = ag.Arguments.host_name.index('-')
                    container_name = ag.Arguments.host_name[:index]
                    dataset_root = os.path.join(home_dir, container_name)

                if self.business == es.EBusiness.PD:
                    self.dataset_dir = rf"{dataset_root}/BDdata/PD_middle_data/MRI_DTI"
                else:
                    self.dataset_dir = rf"{dataset_root}/BDdata/AD_middle_data/MRI_DTI"

        else:
            if self.dataset_dir == "":
                if self.business == es.EBusiness.PD:
                    self.dataset_dir = r"E:\BDdata\PD_middle_data\MRI_DTI"
                else:
                    self.dataset_dir = r"E:\BDdata\AD_middle_data\MRI_DTI"

        if not os.path.isdir(self.dataset_dir):
            raise NotADirectoryError(f"dataset_dir={self.dataset_dir}")

        BDArguments.event_parts = BDArguments.EVENTS

    @staticmethod
    def initialize_globally(gpu_no: int):
        ag.Arguments.initialize_globally(gpu_no)

        txt_file_names = []
        for txt_file_name in BDArguments.TXT_FILE_NAMES:
            if os.name == "posix":
                txt_file_name2 = txt_file_name.replace('\\', '/')
            else:
                txt_file_name2 = txt_file_name.replace('/', '\\')
            txt_file_names.append(txt_file_name2)
        BDArguments.TXT_FILE_NAMES = txt_file_names

        nii_file_names = []
        for nii_file_name in BDArguments.NII_FILE_NAMES:
            if os.name == "posix":
                nii_file_name2 = nii_file_name.replace('\\', '/')
            else:
                nii_file_name2 = nii_file_name.replace('/', '\\')
            nii_file_names.append(nii_file_name2)
        BDArguments.NII_FILE_NAMES = nii_file_names
